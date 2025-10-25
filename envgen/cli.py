# envgen/cli.py
import argparse
import os
import time
import json
import numpy as np

from .config import load_config
from .sampling import sample_grid_size, sample_p_obs, sample_n_pois
from .obstacles import generate_obstacles
from .metrics import summarize_obstacles
from .base import sample_base_on_perimeter
from .gridsearch import bfs_dist, INF
from .pois import place_pois, assign_attributes
from .viz import *
from .energy import energy_dist_map, check_energy_feasibility
from .qa import qa_connectivity, qa_viability, qa_distributions, qa_summary
from .persist import save_instance_npz, save_instance_json, append_index_row
from .sim_engine import simulate_episode, UAV, POI, BaseStation

try:
    from .viz import plot_distance_heatmap
    HAS_DIST_HEATMAP = True
except Exception:
    HAS_DIST_HEATMAP = False


def main():
    ap = argparse.ArgumentParser("envgen")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default="instances")
    ap.add_argument("--n-train", type=int, default=1)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=8,
                    help="Reintentos por instancia hasta pasar QA.")
    ap.add_argument("--qa-viab-thresh", type=float, default=95.0,
                    help="Umbral mínimo de viabilidad energética (%).")
    ap.add_argument("--perim-width", type=int, default=1)
    ap.add_argument("--plot", action="store_true",
                    help="Guarda PNG del mapa/base/POIs dentro de la carpeta del escenario.")
    ap.add_argument("--plot-dist", action="store_true",
                    help="Guarda heatmap de distancias (si está disponible).")
    ap.add_argument("--plot-energy", action="store_true",
                    help="Gráficos energéticos (si implementado).")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--run-seed", type=int, default=None)
    ap.add_argument("--qa-only", action="store_true",
                    help="Oculta todos los logs excepto el resumen QA (Paso 7).")
    ap.add_argument("--simulate", action="store_true",
                    help="Tras aceptar una instancia (ALL_OK), ejecuta una simulación simple y reporta métricas.")
    ap.add_argument("--mission-report", action="store_true",
                    help="Guarda un JSON con métricas de la misión simulada (--simulate).")
    args = ap.parse_args()

    def vprint(*a, **k):
        if not args.qa_only:
            print(*a, **k)

    # === Cargar configuración ===
    cfg = load_config(args.config)
    vprint("[OK] config cargado.")

    # === Parámetros de configuración ===
    seeds         = cfg["simulation_environment"]["seeds"]
    gs            = cfg["simulation_environment"]["grid_size"]
    genr          = cfg["generation_rules"]
    ncfg          = genr["n_pois"]
    limits        = tuple(ncfg["n_pois_size"])
    profiles_root = ncfg["profiles"]
    dens_range    = tuple(genr["obstacles"]["density_range"])
    delta_t_s     = float(cfg["simulation_environment"]["delta_t_s"])
    horizon_ticks = int(cfg["simulation_environment"]["horizon_ticks"])
    speed_ms      = float(cfg["uav_specs"]["cruise_speed_ms"])
    L_o           = float(cfg["routes"]["cell_distances_m"]["L_o"])
    L_d           = float(cfg["routes"]["cell_distances_m"]["L_d"])

    os.makedirs(args.out, exist_ok=True)

    # === Semillas maestras ===
    run_seed = args.run_seed if args.run_seed is not None else (time.time_ns() & 0xFFFFFFFF)
    obs_parent  = np.random.SeedSequence([seeds["obstacles"] + args.seed_offset, run_seed, 0])
    pois_parent = np.random.SeedSequence([seeds["pois"]       + args.seed_offset, run_seed, 1])

    print(f"[seeds] run_seed={run_seed} | obs_parent={obs_parent.entropy}")

    # === Loop por split ===
    for split_i, (split, Ninst) in enumerate([("train", args.n_train), ("val", args.n_val)]):
        obs_children  = obs_parent.spawn(Ninst + 1)
        pois_children = pois_parent.spawn(Ninst + 1)

        for k in range(Ninst):
            print(f"\n=== split={split} | inst={k:03d} ===")
            passed = False
            tries_used = 0

            obs_k = obs_children[k]
            pois_k = pois_children[k]

            attempt_obs = obs_k.spawn(args.max_retries)
            attempt_poi = pois_k.spawn(args.max_retries)

            # === Reintentos ===
            for attempt in range(args.max_retries):
                tries_used = attempt + 1
                rng_obs  = np.random.default_rng(attempt_obs[attempt])
                rng_pois = np.random.default_rng(attempt_poi[attempt])

                # Paso 1: muestreo de escenario
                H, W = sample_grid_size(gs[split], rng_obs)
                pobs = sample_p_obs(dens_range, rng_obs)
                Np   = sample_n_pois(H, W, ncfg[split], limits, profiles_root, rng_pois)

                # Paso 2: obstáculos
                grid = generate_obstacles(H, W, pobs, rng_obs,
                                          clear_perim=True,
                                          perim_width=args.perim_width,
                                          min_free_perim_ratio=0.05,
                                          max_tries=50)

                # Paso 3: base
                base_xy = sample_base_on_perimeter(grid, rng_obs)

                # Paso 5: conectividad (BFS)
                distmap = bfs_dist(grid, base_xy)

                # Paso 4: POIs (colocación + filtrado por alcanzables)
                pois_xy = place_pois(grid, Np, rng_pois, forbid_xy=base_xy)
                pois_xy = [(y, x) for (y, x) in pois_xy if distmap[y, x] < INF]

                # Helper ETA
                def eta_ticks_fn(xy):
                    y, x = xy
                    d_cells = int(distmap[y, x])
                    if d_cells >= INF:
                        return INF
                    t_s = (d_cells * L_o) / max(speed_ms, 1e-9)
                    return int(np.ceil(t_s / max(delta_t_s, 1e-9)))

                # Atributos POIs
                pois = assign_attributes(
                    pois_xy,
                    cfg_pois=cfg["pois"],
                    eta_fn=eta_ticks_fn,
                    delta_t_s=delta_t_s,
                    horizon_ticks=horizon_ticks,
                    rng=rng_pois
                )

                # Paso 6: energía (campo + chequeo)
                E_max      = float(cfg["uav_specs"]["E_max"][split])
                E_reserve  = float(cfg["uav_specs"]["E_reserve"][split])
                e_ortho    = float(cfg["uav_specs"]["energy_model"]["e_move_ortho"])
                e_diag_all = cfg["uav_specs"]["energy_model"]["e_move_diag"]
                e_diag     = float(e_diag_all[split]) if isinstance(e_diag_all, dict) else float(e_diag_all)
                e_wait     = float(cfg["uav_specs"]["energy_model"].get("e_wait", 0.0))

                energy_map = energy_dist_map(grid, base_xy, e_move_ortho=e_ortho, e_move_diag=e_diag)
                pois_eval = check_energy_feasibility(
                    pois, energy_map, E_max=E_max, E_reserve=E_reserve,
                    include_dwell_energy=True, e_wait=e_wait
                )

                # Paso 7: QA consolidado
                q_conn = qa_connectivity([(p["y"], p["x"]) for p in pois], distmap)
                q_viab = qa_viability(pois_eval, min_pct_ok=args.qa_viab_thresh)
                q_dist = qa_distributions(pois)
                q = qa_summary(q_conn, q_viab, q_dist)

                print(f"QA.summary -> conn={q['connectivity_ok']} | viab≥{args.qa_viab_thresh:.0f}%={q['viability_ok']} "
                      f"| distrib={q['distributions_ok']} | ALL_OK={q['ALL_OK']} "
                      f"(try {tries_used}/{args.max_retries})")

                if q["ALL_OK"]:
                    passed = True

                    # Métricas de obstáculos
                    obst_stats = summarize_obstacles(grid)

                    # Resumen energía roundtrip desde energy_map
                    Er = np.array([p["energy_cost"] for p in pois_eval if np.isfinite(p["energy_cost"])], dtype=float)
                    E_min = float(Er.min()) if Er.size else np.nan
                    E_med = float(np.median(Er)) if Er.size else np.nan
                    E_mean = float(Er.mean()) if Er.size else np.nan
                    E_max_obs = float(Er.max()) if Er.size else np.nan

                    # TW stats
                    widths = [(p["tw"]["tmax"] - p["tw"]["tmin"]) for p in pois if p.get("tw") is not None]
                    tw_any = len(widths) > 0
                    tw_min = int(min(widths)) if tw_any else None
                    tw_max = int(max(widths)) if tw_any else None

                    # Prioridades (robusto a vacío)
                    prios = [p["priority"] for p in pois]
                    if prios:
                        uniq, cnt = np.unique(prios, return_counts=True)
                    else:
                        uniq, cnt = np.array([]), np.array([])
                    prio_levels = int(uniq.size) if uniq.size > 0 else 0
                    prio_mode_ratio = float(cnt.max() / cnt.sum()) if cnt.size > 0 else 0.0

                    # Dwell
                    dw = [p["dwell_ticks_eff"] for p in pois]
                    dwell_min = int(min(dw)) if dw else None
                    dwell_max = int(max(dw)) if dw else None

                    # === Persistencia en carpeta por escenario ===
                    stem = f"inst_{split}_{k:03d}_H{H}W{W}_run{run_seed}_try{tries_used}"
                    inst_dir = os.path.join(args.out, stem)
                    os.makedirs(inst_dir, exist_ok=True)

                    # Guardar NPZ/JSON dentro de la carpeta del escenario
                    npz_path = save_instance_npz(inst_dir, stem, grid, distmap, energy_map)

                    # === [NEW] Estadísticas de personas por POI ===
                    persons = [p.get("n_persons", 0) for p in pois]
                    has_any_person = any(pp > 0 for pp in persons)
                    p_mean = float(np.mean(persons)) if persons else 0.0
                    p_max  = int(np.max(persons)) if persons else 0
                    p_pos  = 100.0 * float(np.mean(np.array(persons) > 0)) if persons else 0.0


                    meta = {
                        "run_seed": int(run_seed), "split": split, "k": int(k), "tries": int(tries_used),
                        "H": int(H), "W": int(W), "p_obs": float(pobs), "base_xy": list(base_xy),
                        "uav_energy": {"E_max": E_max, "E_reserve": E_reserve,
                                       "e_move_ortho": e_ortho, "e_move_diag": e_diag, "e_wait": e_wait},
                        "metrics": {
                            "density_real": float(obst_stats["density_real"]),
                            "perimeter_free": float(obst_stats["perimeter_free_ratio"]),
                            "pct_reachable": float(q_conn["pct_reachable"]),
                            "pct_viable": float(q_viab["pct_ok"]),
                            "E_round": {"min": E_min, "med": E_med, "mean": E_mean, "max": E_max_obs},
                            "tw_any": bool(tw_any), "tw_min": tw_min, "tw_max": tw_max
                        },

                        "pois": pois,
                        "persons_any": bool(has_any_person),
                        "persons_mean": p_mean,
                        "persons_max": p_max,
                        "persons_p_gt0": p_pos,
                    }
                    json_path = save_instance_json(inst_dir, stem, meta)

                    # Index por escenario (dentro de la carpeta del escenario)
                    append_index_row(inst_dir, {
                        "run_seed": run_seed, "split": split, "k": k, "tries": tries_used,
                        "H": H, "W": W, "p_obs": pobs,
                        "density_real": obst_stats["density_real"],
                        "perimeter_free": obst_stats["perimeter_free_ratio"],
                        "N_pois": len(pois), "pct_reachable": q_conn["pct_reachable"],
                        "pct_viable": q_viab["pct_ok"],
                        "prio_levels": prio_levels, "prio_mode_ratio": prio_mode_ratio,
                        "dwell_min": dwell_min, "dwell_max": dwell_max,
                        "tw_any": tw_any, "tw_min": tw_min, "tw_max": tw_max,
                        "E_round_min": E_min, "E_round_med": E_med,
                        "E_round_mean": E_mean, "E_round_max": E_max_obs,
                        "E_max": E_max, "E_reserve": E_reserve, "stem": stem,
                        "persons_any": has_any_person,
                        "persons_mean": p_mean,
                        "persons_max": p_max,
                        "persons_p_gt0": p_pos
                    })

                    print(f"[SAVE] ok → {npz_path} | {json_path}")

                    # Mapa y heatmap dentro de la carpeta del escenario
                    if args.plot:
                        map_img = os.path.join(inst_dir, f"map_{stem}.png")
                        title = f"{split} | {H}x{W} | base={base_xy} | POIs={len(pois)}"
                        highlight_idx = int(np.argmax([p["priority"] for p in pois])) if pois else None
                        plot_grid_with_base_pois(grid, base_xy, pois, highlight_idx=highlight_idx,
                                                 title=title, savepath=map_img)
                        print(f"[plot] guardado → {map_img}")

                    if args.plot_dist and HAS_DIST_HEATMAP:
                        dist_img = os.path.join(inst_dir, f"dist_{stem}.png")
                        plot_distance_heatmap(grid, distmap, base_xy, pois_xy,
                                              savepath=dist_img,
                                              title=f"{split} | distancia (celdas) desde la base")
                        print(f"[plot-dist] guardado → {dist_img}")

                    # === Simulación temporal ===
                    if args.simulate:
                        # número de UAVs (mínimo del rango del split)
                        nu_min, nu_max = cfg["uav_specs"]["n_uavs"][split]
                        Emax_cfg = float(cfg["uav_specs"]["E_max"][split])
                        uavs_list = [UAV(uid=i, pos=tuple(base_xy), E=Emax_cfg) for i in range(nu_min)]

                        pois_objs = [
                            POI(
                                y=p["y"], x=p["x"],
                                dwell_ticks=p["dwell_ticks"],                     # base
                                dwell_ticks_eff=p.get("dwell_ticks_eff", None),   # [NEW]
                                n_persons=p.get("n_persons", 0),                  # [NEW]
                                priority=p["priority"],
                                tw=p.get("tw")
                            )
                            for p in pois
                        ]

                        vprint(f"pois.persons     -> mean={p_mean:.2f} | max={p_max} | p>0={p_pos:.1f}%")

                        base_obj = BaseStation(
                            xy=tuple(base_xy),
                            capacity=int(cfg["base_station"]["base_capacity"]),
                            charge_model=str(cfg["base_station"]["charge_model"]),
                            turnaround_ticks=tuple(cfg["base_station"]["turnaround_ticks"])
                        )

                        # callback para snapshots (cada 10 ticks) 
                        def _save_snapshot(t: int, uavs_xy_uidyx):
                            fname = os.path.join(inst_dir, f"snap_t{t:03d}.png")
                            pois_xy_plot = [(p["y"], p["x"]) for p in pois]
                            plot_snapshot_frame(
                                grid, tuple(base_xy), pois_xy_plot, uavs_xy_uidyx,
                                title=f"t={t} ticks",
                                savepath=fname
                            )
                        
                        # callback final para la imagen con todo el recorrido
                        def _save_final_paths(paths_dict):
                            pois_xy_plot = [(p["y"], p["x"]) for p in pois]
                            full_img = os.path.join(inst_dir, f"traj_full_{stem}.png")
                            plot_trajectories_full(
                                grid, tuple(base_xy), pois_xy_plot, paths_dict,
                                title=f"Full trajectories | total ticks={horizon_ticks}",
                                savepath=full_img
                            )
                            print(f"[plot-traj-full] guardado → {full_img}")


                        res = simulate_episode(
                            grid=grid, base_xy=tuple(base_xy), pois=pois_objs, uavs=uavs_list,
                            delta_t_s=delta_t_s, horizon_ticks=horizon_ticks,
                            L_o=L_o, L_d=L_d, speed_ms=speed_ms,
                            e_move_ortho=e_ortho, e_move_diag=e_diag, e_wait=e_wait,
                            base=base_obj,
                            snapshot_every=100,
                            snapshot_cb=_save_snapshot,
                            final_traj_cb=_save_final_paths
                        )
                        print(f"[simulate] ticks={res['ticks_used']} | served={res['served']}/{res['total']} "
                              f"| violations={res['violations']} | RTB={res['n_rtb']}")

                        if args.mission_report:
                            rep_path = os.path.join(inst_dir, f"mission_{stem}.json")
                            with open(rep_path, "w", encoding="utf-8") as f:
                                json.dump(res, f, ensure_ascii=False, indent=2)
                            print(f"[report] guardado → {rep_path}")

                    break  # instancia pasó QA

            # fin x attempt

            if not passed:
                print(f"[FAIL] {split} inst={k:03d} descartada tras {args.max_retries} intentos.")

    print(f"\n[INFO] Directorio de salida asegurado: {args.out}")


if __name__ == "__main__":
    main()
