# envgen/energy.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any
import heapq
import numpy as np

INF = 1e18

# 8 vecinos (dy, dx) y tipo de paso (0=ortho, 1=diag)
NEIGHBORS = [
    (-1,  0, 0), (1,  0, 0), (0, -1, 0), (0, 1, 0),   # ortogonales
    (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)    # diagonales
]

def energy_dist_map(
    grid: np.ndarray,
    start: Tuple[int, int],
    e_move_ortho: float,
    e_move_diag: float
) -> np.ndarray:
    """
    Dijkstra en grilla: costo mínimo de energía desde 'start' a cada celda libre.
    grid[True]=obstáculo (no transitable), grid[False]=libre.
    Retorna np.ndarray float64 con INF donde es inalcanzable.
    """
    H, W = grid.shape
    sy, sx = start
    dist = np.full((H, W), INF, dtype=np.float64)
    if grid[sy, sx]:
        return dist
    dist[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]
    while pq:
        cost, y, x = heapq.heappop(pq)
        if cost > dist[y, x]:
            continue
        for dy, dx, t in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx]:
                step = e_move_diag if t == 1 else e_move_ortho
                nc = cost + step
                if nc < dist[ny, nx]:
                    dist[ny, nx] = nc
                    heapq.heappush(pq, (nc, ny, nx))
    return dist

def check_energy_feasibility(
    pois: List[Dict[str, Any]],
    energy_map: np.ndarray,
    E_max: float,
    E_reserve: float,
    *,
    include_dwell_energy: bool = False,
    e_wait: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Marca cada POI con 'energy_cost', 'feasible' y 'reason' (si no lo es).
    Regla base: E_max >= 2*energy_map[y,x] + E_reserve (+ e_wait * dwell si se pide), viene de la fórmula del artículo.
    """
    out = []
    for p in pois:
        y, x = p["y"], p["x"]
        emap = float(energy_map[y, x])
        if not np.isfinite(emap) or emap >= INF/2:
            p2 = {**p, "energy_cost": float("inf"), "feasible": False, "reason": "unreachable"}
            out.append(p2); continue

        roundtrip = 2.0 * emap
        if include_dwell_energy:
            roundtrip += e_wait * float(p.get("dwell_ticks", 0))

        feasible = (E_max >= roundtrip + E_reserve)
        reason = "" if feasible else "insufficient_energy"
        out.append({**p, "energy_cost": roundtrip, "feasible": feasible, "reason": reason})
    return out
