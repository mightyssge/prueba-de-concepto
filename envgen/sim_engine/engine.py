# envgen/sim_engine/engine.py
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from .entities import UAV, POI, BaseStation
from .planner import greedy_follow_distmap
from ..gridsearch import bfs_dist, INF
from .utils import ticks_per_cell


def simulate_episode(
    grid: np.ndarray,
    base_xy: Tuple[int, int],
    pois: List[POI],
    uavs: List[UAV],
    *,
    delta_t_s: float,
    horizon_ticks: int,
    L_o: float, L_d: float,
    speed_ms: float,
    e_move_ortho: float, e_move_diag: float, e_wait: float,
    base: BaseStation,
    snapshot_every: int = 100,
    snapshot_cb: Optional[Callable[[int, List[Tuple[int, int, int]]], None]] = None,
    final_traj_cb: Optional[Callable[[Dict[int, List[Tuple[int, int]]]], None]] = None,
) -> Dict:
    """
    Simulación discreta 'tick a tick' con contabilidad por UAV/POI.
    - snapshot_cb(t, [(uid,y,x), ...]) se invoca cada 'snapshot_every' ticks y al final.
    - final_traj_cb(paths) se invoca una vez al final con paths: uid -> [(y,x), ...] (no se guarda en JSON).
    Devuelve un dict JSON-ready SIN trayectorias (no incluye fotogramas).
    """
    # Ticks por celda (ortogonal/diagonal)
    tpc_o = ticks_per_cell(L_o, speed_ms, delta_t_s)
    tpc_d = ticks_per_cell(L_d, speed_ms, delta_t_s)

    # Cola ingenua de objetivos: POIs no servidos en orden
    poi_queue = [(p.y, p.x) for p in pois if not p.served]

    def plan_path(u: UAV, goal: Tuple[int, int]):
        """Plan simple: seguir gradiente de distancias hacia 'goal'."""
        distmap = bfs_dist(grid, goal)
        u.path = greedy_follow_distmap(distmap, u.pos, goal)

    served_cnt = 0
    violations = 0
    n_rtb = 0

    # Tracking completo de trayectorias (solo en memoria)
    paths: Dict[int, List[Tuple[int, int]]] = {u.uid: [u.pos] for u in uavs}

    # Snapshot inicial
    if snapshot_cb is not None:
        snapshot_cb(0, [(u.uid, u.pos[0], u.pos[1]) for u in uavs])

    # Bucle temporal
    for t in range(horizon_ticks):
        # Asignación simple de objetivos a UAVs en idle
        for u in uavs:
            if u.state == "idle" and u.E > 0 and poi_queue:
                u.state = "en_route"
                u.target = poi_queue.pop(0)
                plan_path(u, u.target)

        # Actualización por UAV
        for u in uavs:
            # Servicio en curso
            if u.state == "servicing":
                if u.service_left > 0:
                    u.service_left -= 1
                    u.E -= e_wait
                    u.energy_spent += e_wait
                else:
                    u.state = "idle"
                continue

            # En ruta a POI o retorno a base
            if u.state in ("en_route", "rtb"):
                if u.move_cooldown > 0:
                    u.move_cooldown -= 1
                else:
                    if not u.path:
                        # Llegó al destino
                        if u.state == "rtb":
                            # Modelo simple: queda idle (recarga fuera del motor si aplica)
                            u.state = "idle"
                        else:
                            # Servicio del POI objetivo
                            py, px = u.target
                            the: Optional[POI] = next(
                                (p for p in pois if (p.y, p.x) == (py, px) and not p.served),
                                None
                            )
                            if the is not None:
                                # Ventanas temporales (tardanza si llega tarde)
                                if the.tw and t > the.tw["tmax"]:
                                    violations += 1
                                    the.violated = True
                                    the.tardiness += (t - the.tw["tmax"])
                                the.served = True
                                the.first_visit_t = t
                                served_cnt += 1
                                u.state = "servicing"
                                u.service_left = the.dwell_ticks
                            else:
                                u.state = "idle"
                            u.target = None
                        continue

                    # Avance de una celda
                    ny, nx = u.path.pop(0)
                    dy, dx = ny - u.pos[0], nx - u.pos[1]
                    diag = (abs(dy) == 1 and abs(dx) == 1)
                    step_cost = (e_move_diag if diag else e_move_ortho)
                    u.E -= step_cost
                    u.energy_spent += step_cost
                    if diag:
                        u.steps_diag += 1
                    else:
                        u.steps_ortho += 1
                    u.move_cooldown = (tpc_d if diag else tpc_o)
                    u.pos = (ny, nx)

                # Reglas de RTB por baja energía (simple)
                if u.state != "rtb":
                    min_step = min(e_move_ortho, e_move_diag)
                    if u.E <= (min_step + 1.0):
                        u.state = "rtb"
                        u.target = base_xy
                        plan_path(u, u.target)
                        n_rtb += 1

        # Snapshot periódico
        if snapshot_cb is not None and ((t + 1) % max(snapshot_every, 1) == 0):
            snapshot_cb(t + 1, [(u.uid, u.pos[0], u.pos[1]) for u in uavs])

        # Registrar posición al final de cada tick
        for u in uavs:
            paths[u.uid].append(u.pos)

        # Fin anticipado si todos los POIs fueron servidos
        if all(p.served for p in pois):
            # Snapshot final si no coincidió con múltiplo
            if snapshot_cb is not None and ((t + 1) % max(snapshot_every, 1) != 0):
                snapshot_cb(t + 1, [(u.uid, u.pos[0], u.pos[1]) for u in uavs])
            break

    # Callback de trayectoria completa (para la imagen final de recorridos)
    if final_traj_cb is not None:
        final_traj_cb(paths)

    # Reporte por UAV
    uav_reports = []
    for u in uavs:
        dist_m = u.steps_ortho * L_o + u.steps_diag * L_d
        uav_reports.append({
            "uid": u.uid,
            "energy_spent": round(u.energy_spent, 3),
            "steps_ortho": u.steps_ortho,
            "steps_diag": u.steps_diag,
            "distance_m": round(dist_m, 3)
        })

    # Reporte por POI
    poi_reports = [{
        "y": p.y, "x": p.x,
        "served": bool(p.served),
        "violated": bool(getattr(p, "violated", False)),
        "tardiness_ticks": int(getattr(p, "tardiness", 0))
    } for p in pois]

    return {
        "ticks_used": (t + 1),
        "served": sum(1 for p in pois if p.served),
        "total": len(pois),
        "violations": violations,
        "n_rtb": n_rtb,
        "uavs": uav_reports,
        "pois": poi_reports
    }
