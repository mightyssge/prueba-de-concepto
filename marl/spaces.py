from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envgen.sim_engine.entities import POI, UAV
from envgen.sim_engine.utils import ticks_per_cell

# Discrete actions (0-10)
# 0-7 => 8-neighbor moves, 8=service, 9=hover, 10=RTB
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (-1, -1),  # up-left
    5: (-1, 1),   # up-right
    6: (1, -1),   # down-left
    7: (1, 1),    # down-right
    8: (0, 0),    # service
    9: (0, 0),    # hover
    10: (0, 0),   # rtb (handled by policy logic)
}


@dataclass
class LocalObservation:
    obs_vector: np.ndarray
    node_feats: np.ndarray
    adj_matrix: np.ndarray
    action_mask: np.ndarray


def _norm(v: float, denom: float) -> float:
    return float(v) / float(max(denom, 1e-6))


def _flight_mode_onehot(state: str) -> np.ndarray:
    modes = ["idle", "en_route", "servicing", "rtb"]
    out = np.zeros(len(modes), dtype=np.float32)
    try:
        out[modes.index(state)] = 1.0
    except ValueError:
        pass
    return out


def _poi_lookup_dict(pois: Sequence[POI]) -> Dict[Tuple[int, int], POI]:
    return {(p.y, p.x): p for p in pois}


def build_action_mask(
    uav: UAV,
    grid: np.ndarray,
    *,
    e_move_ortho: float,
    e_move_diag: float,
    e_wait: float,
    E_reserve: float,
    base_xy: Tuple[int, int],
    pois: Sequence[POI],
    rtb_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Returns a boolean mask of shape (11,) where False marks forbidden actions.
    Rules:
      - forbid moving outside grid or into obstacles
      - forbid moves with insufficient energy (respecting reserve)
      - forbid service if not on top of an unserved POI
      - if RTB is mandatory by energy threshold, only allow RTB
    """
    H, W = grid.shape
    mask = np.ones(len(ACTIONS), dtype=bool)
    poi_lookup = _poi_lookup_dict(pois)

    low_energy_margin = min(e_move_ortho, e_move_diag) + 1.0
    threshold = rtb_threshold if rtb_threshold is not None else max(E_reserve, low_energy_margin)
    rtb_required = (uav.E <= threshold) and (uav.pos != base_xy)

    for a in range(8):
        dy, dx = ACTIONS[a]
        ny, nx = uav.pos[0] + dy, uav.pos[1] + dx
        diag = (dy != 0 and dx != 0)
        step_cost = e_move_diag if diag else e_move_ortho
        if not (0 <= ny < H and 0 <= nx < W):
            mask[a] = False
            continue
        if grid[ny, nx]:
            mask[a] = False
            continue
        if (uav.E - step_cost) < E_reserve:
            mask[a] = False

    # Service only if on top of an unserved POI
    poi_here = poi_lookup.get(uav.pos)
    mask[8] = bool(poi_here and not poi_here.served)

    # Hover allowed only if energy allows the wait cost
    mask[9] = (uav.E - e_wait) >= 0.0

    if rtb_required:
        forced = np.zeros_like(mask, dtype=bool)
        forced[10] = True
        return forced

    # RTB always allowed
    mask[10] = True
    return mask


def build_graph_representation(
    uav: UAV,
    uavs: Sequence[UAV],
    pois: Sequence[POI],
    grid: np.ndarray,
    base_xy: Tuple[int, int],
    distmap_base: np.ndarray,
    energy_map: np.ndarray,
    *,
    horizon_ticks: int,
    E_max: float,
    E_reserve: float,
    ticks_per_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = grid.shape
    nodes: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []
    tps = ticks_per_step if ticks_per_step is not None else ticks_per_cell(1.0, 1.0, 1.0)

    # Base node
    by, bx = base_xy
    nodes.append(
        np.array(
            [
                _norm(by, H),
                _norm(bx, W),
                0.0,  # priority
                0.0,  # eta
                0.0,  # has_tw
                0.0, 0.0,  # tmin/tmax
                0.0,  # dwell
                1.0,  # energy feasible
                1.0,  # deadline feasible
                1.0,  # energy frac
                0.0,  # node type (base)
            ],
            dtype=np.float32,
        )
    )
    coords.append(base_xy)

    # POI nodes
    for p in pois:
        eta_ticks = float(distmap_base[p.y, p.x] if np.isfinite(distmap_base[p.y, p.x]) else 0.0) * float(tps)
        tw = getattr(p, "tw", None)
        has_tw = 1.0 if tw else 0.0
        tmin = float(tw["tmin"]) if tw else 0.0
        tmax = float(tw["tmax"]) if tw else 0.0
        dwell = float(getattr(p, "dwell_ticks_eff", getattr(p, "dwell_ticks", 0)))
        energy_cost = float(energy_map[p.y, p.x]) if np.isfinite(energy_map[p.y, p.x]) else np.inf
        feasible_energy = float((2.0 * energy_cost + dwell) <= (E_max - E_reserve)) if np.isfinite(energy_cost) else 0.0
        deadline_feasible = 1.0
        if tw:
            deadline_feasible = 1.0 if (eta_ticks <= tw["tmax"]) else 0.0

        nodes.append(
            np.array(
                [
                    _norm(p.y, H),
                    _norm(p.x, W),
                    float(getattr(p, "priority", 0)),
                    _norm(eta_ticks, horizon_ticks),
                    has_tw,
                    _norm(tmin, horizon_ticks),
                    _norm(tmax, horizon_ticks),
                    _norm(dwell, horizon_ticks),
                    feasible_energy,
                    float(deadline_feasible),
                    0.0,  # energy frac (not used for POI)
                    1.0,  # node type (poi)
                ],
                dtype=np.float32,
            )
        )
        coords.append((p.y, p.x))

    # Other UAVs (positions + energy fraction)
    for other in uavs:
        if other.uid == uav.uid:
            continue
        ey, ex = other.pos
        nodes.append(
            np.array(
                [
                    _norm(ey, H),
                    _norm(ex, W),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    _norm(other.E, E_max),
                    2.0,  # node type (other uav)
                ],
                dtype=np.float32,
            )
        )
        coords.append((ey, ex))

    node_feats = np.vstack(nodes).astype(np.float32)

    # Adjacency by 8-connectivity plus base fully connected for context
    N = node_feats.shape[0]
    adj = np.eye(N, dtype=bool)
    for i, (y_i, x_i) in enumerate(coords):
        for j, (y_j, x_j) in enumerate(coords):
            if i == j:
                continue
            dy, dx = abs(y_i - y_j), abs(x_i - x_j)
            if max(dy, dx) <= 1:
                adj[i, j] = True
        # Connect base node (0) to everyone to keep the graph coherent
        adj[0, i] = True
        adj[i, 0] = True

    return node_feats, adj.astype(np.float32)


def build_local_observation(
    uav: UAV,
    uavs: Sequence[UAV],
    pois: Sequence[POI],
    grid: np.ndarray,
    base_xy: Tuple[int, int],
    distmap_base: np.ndarray,
    energy_map: np.ndarray,
    *,
    tick: int,
    horizon_ticks: int,
    E_max: float,
    E_reserve: float,
    e_move_ortho: float,
    e_move_diag: float,
    e_wait: float,
    ticks_per_step: Optional[int] = None,
) -> LocalObservation:
    """Constructs the local observation tuple for one UAV."""
    H, W = grid.shape
    dist_base = distmap_base[uav.pos[0], uav.pos[1]]
    dist_norm = _norm(dist_base, max(H, W))
    obs_vec = np.concatenate(
        [
            np.array(
                [
                    _norm(uav.pos[0], H),
                    _norm(uav.pos[1], W),
                    _norm(uav.E, E_max),
                    _norm(tick, horizon_ticks),
                    dist_norm,
                ],
                dtype=np.float32,
            ),
            _flight_mode_onehot(uav.state),
        ]
    )

    node_feats, adj = build_graph_representation(
        uav,
        uavs,
        pois,
        grid,
        base_xy,
        distmap_base,
        energy_map,
        horizon_ticks=horizon_ticks,
        E_max=E_max,
        E_reserve=E_reserve,
        ticks_per_step=ticks_per_step,
    )

    action_mask = build_action_mask(
        uav,
        grid,
        e_move_ortho=e_move_ortho,
        e_move_diag=e_move_diag,
        e_wait=e_wait,
        E_reserve=E_reserve,
        base_xy=base_xy,
        pois=pois,
    )
    return LocalObservation(
        obs_vector=obs_vec,
        node_feats=node_feats,
        adj_matrix=adj,
        action_mask=action_mask,
    )


def build_global_state_vector(
    uavs: Sequence[UAV],
    pois: Sequence[POI],
    *,
    tick: int,
    horizon_ticks: int,
    E_max: float,
) -> np.ndarray:
    """Aggregated global state for the centralized critic (CTDE)."""
    energy_fracs = np.array([_norm(u.E, E_max) for u in uavs], dtype=np.float32) if uavs else np.zeros(1, dtype=np.float32)
    served = np.array([1.0 if p.served else 0.0 for p in pois], dtype=np.float32) if pois else np.zeros(1, dtype=np.float32)
    prios = np.array([float(getattr(p, "priority", 0)) for p in pois], dtype=np.float32) if pois else np.zeros(1, dtype=np.float32)
    tw_viol = np.array([1.0 if getattr(p, "violated", False) else 0.0 for p in pois], dtype=np.float32) if pois else np.zeros(1, dtype=np.float32)
    dwell = np.array([float(getattr(p, "dwell_ticks_eff", getattr(p, "dwell_ticks", 0))) for p in pois], dtype=np.float32) if pois else np.zeros(1, dtype=np.float32)

    return np.array(
        [
            np.mean(energy_fracs),
            np.min(energy_fracs) if energy_fracs.size else 0.0,
            np.max(energy_fracs) if energy_fracs.size else 0.0,
            np.mean(served),
            np.sum(prios),
            np.mean(prios) if prios.size else 0.0,
            np.mean(tw_viol) if tw_viol.size else 0.0,
            np.mean(dwell) if dwell.size else 0.0,
            _norm(tick, horizon_ticks),
        ],
        dtype=np.float32,
    )
