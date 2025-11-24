# -*- coding: utf-8 -*-
"""
Corrida única usando exclusivamente la política del checkpoint (sin heurísticas extra).
- Opciones para usar/omitir obstáculos.
- Log cada tick (ajustable), snapshots cada N ticks y GIF final.
- Heatmap de visitas con fondo satelital si se proporciona.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from envgen.config import load_config
from envgen.gridsearch import bfs_dist
from marl.envs import (
    build_lurigancho_fixed_episode,
    load_lurigancho_fixed_data,
    load_lurigancho_map,
)
from marl.eval_marl import clone_instance, GuidedMarlPolicy, GreedyBaselinePolicy, GeneticBaselinePolicy
from marl.networks import GraphActor, CentralCritic
from marl.ppo_agent import PPOAgent
from marl.train_marl import RewardWeights, MarlEnv


def render_env_frame(
    env: MarlEnv,
    trajectories: Dict[int, List[Tuple[int, int]]],
    step_idx: int,
    visit_counts: np.ndarray,
    sat_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    status = compute_status(env)
    H, W = env.grid.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    if sat_img is not None:
        ax.imshow(sat_img, extent=[0, W, H, 0], alpha=0.6, zorder=0)
    bg = np.where(env.grid, 0.25, 0.95)
    ax.imshow(bg, cmap="gray", origin="upper", alpha=0.08, zorder=1)
    nonzero = visit_counts[visit_counts > 0]
    vmax = max(1, float(np.percentile(nonzero, 90)) if nonzero.size else visit_counts.max())
    ax.imshow(visit_counts, cmap="OrRd", origin="upper", alpha=0.55, vmin=0, vmax=vmax, zorder=2)

    pending = [(p.x, p.y) for p in env.pois if not p.served]
    served = [(p.x, p.y) for p in env.pois if p.served]
    if pending:
        ax.scatter([x for x, _ in pending], [y for _, y in pending], s=20, c="tomato", edgecolors="black", linewidths=0.3, zorder=3)
    if served:
        ax.scatter([x for x, _ in served], [y for _, y in served], s=20, c="mediumseagreen", edgecolors="black", linewidths=0.3, zorder=3)
    by, bx = env.base_xy
    ax.scatter([bx], [by], marker="s", s=80, facecolors="none", edgecolors="deepskyblue", linewidths=2.0, zorder=4)
    colors = plt.cm.tab10.colors
    for idx, u in enumerate(env.uavs):
        y, x = u.pos
        ax.scatter([x], [y], marker="^", s=120, facecolors=colors[idx % len(colors)], edgecolors="black", linewidths=0.5, zorder=4)
        if trajectories and u.uid in trajectories:
            xs = [pos[1] for pos in trajectories[u.uid]]
            ys = [pos[0] for pos in trajectories[u.uid]]
            ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.0, alpha=0.6, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(f"Paso {step_idx} | tick {env.tick} | cobertura {status['coverage']:.1%}")
    fig.tight_layout()
    canvas = fig.canvas
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())
    frame = frame[:, :, :3].copy()
    plt.close(fig)
    return frame


def compute_status(env: MarlEnv) -> dict:
    total = len(env.pois)
    served = sum(1 for p in env.pois if p.served)
    coverage = served / max(total, 1)
    violations = sum(1 for p in env.pois if getattr(p, "violated", False))
    avg_energy = float(np.mean([u.E for u in env.uavs])) if env.uavs else 0.0
    return {
        "total_pois": total,
        "served": served,
        "coverage": coverage,
        "violations": violations,
        "avg_energy": avg_energy,
    }


def load_satellite_image(candidates: List[Path], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    for cand in candidates:
        if cand.exists():
            try:
                img = Image.open(cand).convert("RGB")
                img = img.resize((shape[1], shape[0]), Image.BILINEAR)
                print(f"[info] Usando imagen satelital: {cand}")
                return np.array(img)
            except Exception as e:
                print(f"[warn] No se pudo leer {cand}: {e}")
    print("[warn] No se encontró imagen satelital; se dibujará sin fondo.")
    return None


def build_policy(env: MarlEnv, policy_name: str, ckpt_path: Path, guidance_energy_margin: float, guidance_stagnation_steps: int, disable_guidance: bool):
    if policy_name == "greedy":
        return GreedyBaselinePolicy()
    if policy_name == "genetic":
        return GeneticBaselinePolicy()
    # default: marl
    sample_obs = next(iter(env.observations().values()))
    obs_dim = sample_obs.obs_vector.size
    node_dim = sample_obs.node_feats.shape[1]
    global_dim = env.global_state().size
    actor = GraphActor(obs_dim=obs_dim, node_feat_dim=node_dim, hidden_dim=128, n_actions=11)
    critic = CentralCritic(state_dim=global_dim, hidden_dim=128)
    agent = PPOAgent(actor, critic)
    state = torch.load(ckpt_path, map_location=agent.device)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    policy = GuidedMarlPolicy(
        agent,
        deterministic=True,
        energy_margin=guidance_energy_margin,
        stagnation_steps=guidance_stagnation_steps,
        disabled=disable_guidance,
    )
    return policy


def run_episode(
    env: MarlEnv,
    policy,
    log_interval: int = 0,
    snapshot_every: int = 50,
    snapshot_dir: Optional[Path] = None,
    sat_img: Optional[np.ndarray] = None,
    max_ticks: Optional[int] = None,
    log_fn=None,
):
    policy.reset()
    observations = env.observations()
    done = False
    step_idx = 0
    frames: list[np.ndarray] = []
    trajectories = {u.uid: [u.pos] for u in env.uavs}
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    visit_counts = np.zeros_like(env.grid, dtype=np.int32)
    while not done:
        frame = render_env_frame(env, trajectories, step_idx, visit_counts, sat_img=sat_img)
        frames.append(frame)
        for u in env.uavs:
            y, x = u.pos
            visit_counts[y, x] += 1
        actions = policy.select_actions(env, observations)
        observations, _, done, _ = env.step(actions)
        step_idx += 1
        for u in env.uavs:
            trajectories[u.uid].append(u.pos)
        if max_ticks is not None and step_idx >= max_ticks:
            done = True
        if snapshot_dir and snapshot_every > 0 and step_idx % snapshot_every == 0:
            snap_path = snapshot_dir / f"route_tick_{step_idx:04d}.png"
            imageio.imwrite(snap_path, frame)
        if log_interval and (step_idx % log_interval == 0 or done):
            status = compute_status(env)
            served_per_uav = {u.uid: env.served_counts.get(u.uid, 0) for u in env.uavs}
            coverage_per_uav = {uid: (cnt / max(status["total_pois"], 1)) for uid, cnt in served_per_uav.items()}
            if log_fn:
                log_fn(
                    f"[paso {step_idx:05d}] tick={env.tick:05d} "
                    f"serv={status['served']}/{status['total_pois']} "
                    f"cov={status['coverage']:.1%} energia_prom={status['avg_energy']:.1f} "
                    f"rtb={env.rtb_count} "
                    f"viol={status['violations']} "
                    f"serv_uav={served_per_uav} "
                    f"cov_uav={coverage_per_uav}"
                )
    final_status = compute_status(env)
    served_per_uav = {u.uid: env.served_counts.get(u.uid, 0) for u in env.uavs}
    coverage_per_uav = {uid: (cnt / max(final_status["total_pois"], 1)) for uid, cnt in served_per_uav.items()}
    return frames, visit_counts, final_status, served_per_uav, coverage_per_uav


def parse_args():
    ap = argparse.ArgumentParser("run_single_sim_capture")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--scenario-file", type=str, default="lurigancho_scenario.json")
    ap.add_argument("--pois-file", type=str, default="lurigancho_pois_val.json")
    ap.add_argument("--checkpoint", type=str, default="results/ckpt_step3_lurigancho_fixed.pt")
    ap.add_argument("--policy", type=str, choices=["marl", "greedy", "genetic"], default="marl")
    ap.add_argument("--no-obstacles", action="store_true", help="Ignorar obstáculos en la corrida")
    ap.add_argument("--max-ticks", type=int, default=None, help="Corta el episodio tras este número de ticks")
    ap.add_argument("--log-interval", type=int, default=0, help="0 para desactivar log por tick")
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--sat-path", type=str, default=None, help="Ruta explícita de imagen satelital")
    ap.add_argument("--output-dir", type=str, default="results/routes")
    ap.add_argument("--guidance-energy-margin", type=float, default=1.0)
    ap.add_argument("--guidance-stagnation-steps", type=int, default=0)
    ap.add_argument("--disable-guidance", action="store_true")
    ap.add_argument("--log-file", type=str, default=None, help="Ruta del archivo de log (por defecto output_dir/results.txt)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG_PATH = Path(args.config)
    SCENARIO_PATH = Path(args.scenario_file)
    POIS_PATH = Path(args.pois_file)
    CKPT_PATH = Path(args.checkpoint)
    SNAPSHOT_DIR = Path(args.output_dir)
    VIDEO_PATH = SNAPSHOT_DIR / "lurigancho_single_run.gif"

    cfg = load_config(CONFIG_PATH)
    map_data = load_lurigancho_map(SCENARIO_PATH)
    fixed_data = load_lurigancho_fixed_data(POIS_PATH)
    if args.no_obstacles:
        fixed_data.obstacles = []
    rng = np.random.default_rng(2025)
    episode, ei_map, hooks = build_lurigancho_fixed_episode(map_data, fixed_data, cfg, rng, split="val")

    if args.sat_path:
        sat_candidates = [Path(args.sat_path)]
    else:
        sat_candidates = [
            Path("results/lurigancho_sat.png"),
            Path("lurigancho_sat.png"),
            Path("data/lurigancho_sat.png"),
        ]
    sat_img = load_satellite_image(sat_candidates, episode["grid"].shape)

    clone = clone_instance(episode)
    env = MarlEnv(
        clone,
        RewardWeights(),
        global_obs=ei_map,
        hooks=hooks,
        env_mode="lurigancho_fixed",
        ignore_horizon=False,
    )

    policy = build_policy(
        env,
        args.policy,
        CKPT_PATH,
        guidance_energy_margin=args.guidance_energy_margin,
        guidance_stagnation_steps=args.guidance_stagnation_steps,
        disable_guidance=args.disable_guidance,
    )

    LOG_LINES: list[str] = []

    def log_fn(msg: str):
        LOG_LINES.append(msg)
        print(msg)

    frames, visit_counts, final_status, served_per_uav, coverage_per_uav = run_episode(
        env,
        policy,
        log_interval=args.log_interval,
        snapshot_every=args.snapshot_every,
        snapshot_dir=SNAPSHOT_DIR,
        sat_img=sat_img,
        max_ticks=args.max_ticks,
        log_fn=log_fn,
    )
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(VIDEO_PATH, frames, duration=0.45)
    print(f"[video] guardado en {VIDEO_PATH}")
    # Heatmap final
    try:
        H, W = env.grid.shape
        fig, ax = plt.subplots(figsize=(10, 4))
        if sat_img is not None:
            ax.imshow(sat_img, extent=[0, W, H, 0], alpha=0.6)
        bg = np.where(env.grid, 0.25, 0.95)
        ax.imshow(bg, cmap="gray", origin="upper", alpha=0.08)
        nonzero = visit_counts[visit_counts > 0]
        vmax = max(1, float(np.percentile(nonzero, 90)) if nonzero.size else visit_counts.max())
        hm = ax.imshow(visit_counts, cmap="OrRd", origin="upper", alpha=0.55, vmin=0, vmax=vmax)
        plt.colorbar(hm, ax=ax, shrink=0.8, pad=0.02, label="Visitas por celda")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        ax.set_title("Mapa de calor de trayectorias (frecuencia de visitas)")
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        heat_path = SNAPSHOT_DIR / "heatmap_routes.png"
        fig.tight_layout()
        fig.savefig(heat_path, dpi=150)
        plt.close(fig)
        print(f"[heatmap] guardado en {heat_path}")
    except Exception as e:
        print(f"[heatmap] error al generar heatmap: {e}")

    # Log final resumido
    final_line = ("[final] "
                  f"policy={args.policy} "
                  f"serv={final_status['served']}/{final_status['total_pois']} "
                  f"cov={final_status['coverage']:.1%} "
                  f"viol={final_status['violations']} "
                  f"rtb={env.rtb_count} "
                  f"serv_uav={served_per_uav} "
                  f"cov_uav={coverage_per_uav}")
    LOG_LINES.append(final_line)
    print(final_line)

    # Guardar log a archivo
    log_path = Path(args.log_file) if args.log_file else SNAPSHOT_DIR / "results.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"[log] guardado en {log_path}")
