from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from envgen.energy import energy_dist_map
from envgen.gridsearch import bfs_dist, INF
from envgen.sim_engine.entities import POI, UAV
from envgen.sim_engine.utils import ticks_per_cell

TERRAIN_TO_INT = {"residential": 0, "hill": 1, "river": 2}
RISK_LABELS = {0: "low", 1: "medium", 2: "high"}
RISK_VALUE_MAP = {label: val for val, label in RISK_LABELS.items()}
RISK_PRIORITY = {"low": 1, "medium": 2, "high": 3}
POI_WEIGHTS = {
    ("residential", "high"): 1.0,
    ("residential", "medium"): 0.6,
    ("residential", "low"): 0.25,
    ("hill", "high"): 0.18,
    ("hill", "medium"): 0.05,
    ("hill", "low"): 0.0,
    ("river", "high"): 0.12,
    ("river", "medium"): 0.02,
    ("river", "low"): 0.0,
}
TW_PROB = {"low": 0.15, "medium": 0.35, "high": 0.60}
DWELL_RANGES = {"low": (1, 2), "medium": (2, 3), "high": (3, 4)}
N_PERSONS_VALUES = np.array([0, 1, 2, 3, 4], dtype=int)
N_PERSONS_PROBS = np.array([0.5, 0.30, 0.12, 0.05, 0.03], dtype=float)




def load_lurigancho_map(path: Path | str) -> LuriganchoMapData:
    return LuriganchoMapData.from_json(Path(path))
@dataclass
class LuriganchoMapData:
    rows: int
    cols: int
    base_cell: int
    base_xy: Tuple[int, int]
    terrain_grid: np.ndarray
    risk_grid: np.ndarray

    @classmethod
    def from_json(cls, path: Path) -> "LuriganchoMapData":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data.get("grid_meta") or data.get("meta")
        if meta is None:
            raise KeyError("missing grid meta info")
        rows = int(meta["rows"])
        cols = int(meta["cols"])
        base_cell = int(meta.get("base_cell") or data.get("base_cell") or 0)
        base_xy = (base_cell // cols, base_cell % cols)
        terrain_grid = np.full((rows, cols), TERRAIN_TO_INT["residential"], dtype=np.int8)
        risk_grid = np.zeros((rows, cols), dtype=np.int8)
        terrain_map: Dict[str, str] = dict(data.get("cell_terrain", {}))
        risk_map: Dict[str, int] = dict(data.get("cell_priority", {}))
        cells_data = data.get("cells")
        if isinstance(cells_data, dict):
            for cell_info in cells_data.values():
                cell_idx = int(cell_info.get("cell_id") or cell_info.get("id", 0))
                terrain_map[str(cell_idx)] = cell_info.get("terrain", terrain_map.get(str(cell_idx), "residential"))
                risk_level = cell_info.get("risk_level")
                if risk_level is None:
                    risk_level = cell_info.get("priority_level")
                if risk_level is not None:
                    risk_map[str(cell_idx)] = risk_level
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            terr = terrain_map.get(str(idx), "residential")
            pri = risk_map.get(str(idx), 0)
            terrain_grid[r, c] = TERRAIN_TO_INT.get(terr, 0)
            if isinstance(pri, str):
                pri = RISK_VALUE_MAP.get(pri.lower(), 0)
            elif pri is None:
                pri = 0
            risk_grid[r, c] = int(pri)
        return cls(rows=rows, cols=cols, base_cell=base_cell, base_xy=base_xy, terrain_grid=terrain_grid, risk_grid=risk_grid)

    def cell_info(self, cell_id: int) -> Tuple[int, int, str, str]:
        row, col = divmod(cell_id, self.cols)
        terrain = {v: k for k, v in TERRAIN_TO_INT.items()}[int(self.terrain_grid[row, col])]
        risk = RISK_LABELS[int(self.risk_grid[row, col])]
        return row, col, terrain, risk

    def iter_cells(self) -> Sequence[int]:
        return range(self.rows * self.cols)


class LuriganchoEIMap:
    def __init__(self, map_data: LuriganchoMapData, obstacles: Sequence[int], poi_cells: Sequence[int]):
        self.map = map_data
        H, W = map_data.rows, map_data.cols
        static_channels = []
        terrain_one_hot = np.zeros((H, W, len(TERRAIN_TO_INT)), dtype=np.float32)
        for terr_str, terr_idx in TERRAIN_TO_INT.items():
            terrain_one_hot[:, :, terr_idx] = (map_data.terrain_grid == terr_idx).astype(np.float32)
        static_channels.append(terrain_one_hot)
        risk_one_hot = np.zeros((H, W, len(RISK_LABELS)), dtype=np.float32)
        for risk_val in RISK_LABELS.keys():
            risk_one_hot[:, :, risk_val] = (map_data.risk_grid == risk_val).astype(np.float32)
        static_channels.append(risk_one_hot)
        base_mask = np.zeros((H, W, 1), dtype=np.float32)
        base_mask[map_data.base_xy[0], map_data.base_xy[1], 0] = 1.0
        static_channels.append(base_mask)
        obstacle_mask = np.zeros((H, W, 1), dtype=np.float32)
        for cell in obstacles:
            r, c = divmod(cell, map_data.cols)
            obstacle_mask[r, c, 0] = 1.0
        static_channels.append(obstacle_mask)
        self.static = np.concatenate(static_channels, axis=-1)
        self.static_flat = self.static.flatten()
        self.dynamic = np.zeros((H, W, 3), dtype=np.float32)
        self.reset_dynamic(poi_cells)

    def reset_dynamic(self, poi_cells: Sequence[int]) -> None:
        self.dynamic.fill(0.0)
        for cell in poi_cells:
            r, c = divmod(cell, self.map.cols)
            self.dynamic[r, c, 1] = 1.0  # pending
        # mark base as visited at start
        self.mark_visit(*self.map.base_xy)

    def mark_visit(self, row: int, col: int) -> None:
        self.dynamic[row, col, 0] = 1.0

    def mark_served(self, row: int, col: int) -> None:
        self.dynamic[row, col, 1] = 0.0
        self.dynamic[row, col, 2] = 1.0

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.static_flat, self.dynamic.flatten()], dtype=np.float32)


def _sample_dwell_base(rng: np.random.Generator, risk: str) -> int:
    lo, hi = DWELL_RANGES[risk]
    return int(rng.integers(lo, hi + 1))


def _service_ticks(dwell_base: int, n_persons: int) -> int:
    return int(dwell_base * max(1, int(n_persons)))


def _sample_time_window(
    rng: np.random.Generator,
    risk: str,
    eta_ticks: Optional[float],
    horizon_ticks: int,
) -> Optional[Dict[str, int]]:
    prob = TW_PROB[risk]
    if rng.random() > prob or eta_ticks is None or eta_ticks >= INF:
        return None
    start = int(max(0, eta_ticks))
    width = int(rng.integers(60, 240))
    end = min(horizon_ticks, start + width)
    if start >= horizon_ticks:
        return None
    return {"tmin": start, "tmax": end}


def _build_grid(rows: int, cols: int, obstacles: Sequence[int]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=bool)
    for cell in obstacles:
        r, c = divmod(cell, cols)
        grid[r, c] = True
    return grid


def build_lurigancho_random_episode(
    map_data: LuriganchoMapData,
    cfg: Dict,
    rng: np.random.Generator,
    *,
    split: str,
) -> Tuple[dict, LuriganchoEIMap, Dict[str, callable]]:
    rows, cols = map_data.rows, map_data.cols
    base_cell = map_data.base_cell
    horizon_ticks = int(cfg["simulation_environment"]["horizon_ticks"])
    delta_t_s = float(cfg["simulation_environment"]["delta_t_s"])
    # sample obstacles
    allowed_obstacles = [
        cell
        for cell in map_data.iter_cells()
        if cell != base_cell
        and (
            map_data.terrain_grid[cell // cols, cell % cols] == TERRAIN_TO_INT["hill"]
            or (
                map_data.terrain_grid[cell // cols, cell % cols] == TERRAIN_TO_INT["residential"]
                and map_data.risk_grid[cell // cols, cell % cols] == 2
            )
        )
    ]
    n_obstacles = int(rng.integers(10, 19))
    obstacles = set(rng.choice(allowed_obstacles, size=min(n_obstacles, len(allowed_obstacles)), replace=False).tolist()) if allowed_obstacles else set()
    grid = _build_grid(rows, cols, obstacles)
    distmap = bfs_dist(grid, (base_cell // cols, base_cell % cols))
    # sample POI cells
    eligible_cells: List[int] = []
    weights: List[float] = []
    for cell in map_data.iter_cells():
        if cell == base_cell or cell in obstacles:
            continue
        row, col, terrain, risk = map_data.cell_info(cell)
        w = POI_WEIGHTS.get((terrain, risk), 0.0)
        if w <= 0:
            continue
        eligible_cells.append(cell)
        weights.append(w)
    weights_arr = np.array(weights, dtype=float)
    poi_cells: List[int] = []
    if eligible_cells and weights_arr.sum() > 0:
        probs = weights_arr / weights_arr.sum()
        sample_size = min(50, len(eligible_cells))
        poi_cells = rng.choice(eligible_cells, size=sample_size, replace=False, p=probs).tolist()
    if len(poi_cells) < 50:
        remaining = [
            cell
            for cell in map_data.iter_cells()
            if cell != base_cell and cell not in obstacles and cell not in poi_cells
        ]
        remaining.sort(
            key=lambda c: (
                -map_data.risk_grid[c // cols, c % cols],
                TERRAIN_TO_INT["residential"] - map_data.terrain_grid[c // cols, c % cols],
            )
        )
        needed = 50 - len(poi_cells)
        poi_cells.extend(remaining[:needed])
    elif len(poi_cells) > 50:
        poi_cells = rng.choice(poi_cells, size=50, replace=False).tolist()
    # build POIs
    pois: List[POI] = []
    for idx, cell in enumerate(poi_cells, start=1):
        row, col, terrain, risk = map_data.cell_info(cell)
        priority = RISK_PRIORITY[risk]
        n_persons = int(rng.choice(N_PERSONS_VALUES, p=N_PERSONS_PROBS))
        dwell_base = _sample_dwell_base(rng, risk)
        eta = distmap[row, col]
        ticks_per_step = ticks_per_cell(
            cfg["routes"]["cell_distances_m"]["L_o"],
            cfg["uav_specs"]["cruise_speed_ms"],
            delta_t_s,
        )
        eta_ticks = eta * ticks_per_step if eta < INF else None
        tw = _sample_time_window(rng, risk, eta_ticks, horizon_ticks)
        poi = POI(y=row, x=col, dwell_ticks=dwell_base, priority=priority, tw=tw)
        poi.dwell_ticks_eff = _service_ticks(dwell_base, n_persons)
        poi.n_persons = n_persons
        poi.poi_id = idx
        poi.terrain = terrain
        poi.risk_level = risk
        pois.append(poi)
    # energy parameters
    split_key = split if split in cfg["uav_specs"]["E_max"] else "train"
    E_max = float(cfg["uav_specs"]["E_max"][split_key])
    E_reserve = float(cfg["uav_specs"]["E_reserve"][split_key])
    e_ortho = float(cfg["uav_specs"]["energy_model"]["e_move_ortho"])
    e_diag_all = cfg["uav_specs"]["energy_model"]["e_move_diag"]
    e_diag = float(e_diag_all[split_key]) if isinstance(e_diag_all, dict) else float(e_diag_all)
    e_wait = float(cfg["uav_specs"]["energy_model"].get("e_wait", 0.0))
    energy_map = energy_dist_map(grid, (map_data.base_xy[0], map_data.base_xy[1]), e_move_ortho=e_ortho, e_move_diag=e_diag)
    n_uav_cfg = cfg["uav_specs"]["n_uavs"][split]
    n_uavs = int(rng.integers(int(n_uav_cfg[0]), int(n_uav_cfg[1]) + 1))
    uavs = [UAV(uid=i, pos=map_data.base_xy, E=E_max) for i in range(n_uavs)]
    instance = {
        "grid": grid,
        "base_xy": map_data.base_xy,
        "pois": pois,
        "uavs": uavs,
        "distmap": distmap,
        "energy_map": energy_map,
        "E_max": E_max,
        "E_reserve": E_reserve,
        "e_move_ortho": e_ortho,
        "e_move_diag": e_diag,
        "e_wait": e_wait,
        "horizon_ticks": horizon_ticks,
        "ticks_per_step": int(ticks_per_cell(cfg["routes"]["cell_distances_m"]["L_o"], cfg["uav_specs"]["cruise_speed_ms"], delta_t_s)),
        "L_o": float(cfg["routes"]["cell_distances_m"]["L_o"]),
        "L_d": float(cfg["routes"]["cell_distances_m"]["L_d"]),
    }
    eimap = LuriganchoEIMap(map_data, obstacles, poi_cells)
    hooks = {
        "on_visit": lambda y, x, m=eimap: m.mark_visit(y, x),
        "on_service": lambda y, x, m=eimap: m.mark_served(y, x),
    }
    return instance, eimap, hooks


@dataclass
class LuriganchoFixedData:
    pois: List[Dict]
    obstacles: List[int]


def load_lurigancho_fixed_data(path: Path | str) -> LuriganchoFixedData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pois_dict = data.get("pois", {})
    pois = []
    for _, entry in sorted(pois_dict.items(), key=lambda kv: int(kv[0])):
        pois.append(entry)
    obstacles = data.get("obstacles", [])
    return LuriganchoFixedData(pois=pois, obstacles=obstacles)


def build_lurigancho_fixed_episode(
    map_data: LuriganchoMapData,
    fixed_data: LuriganchoFixedData,
    cfg: Dict,
    rng: np.random.Generator,
    *,
    split: str,
) -> Tuple[dict, LuriganchoEIMap, Dict[str, callable]]:
    rows, cols = map_data.rows, map_data.cols
    horizon_ticks = int(cfg["simulation_environment"]["horizon_ticks"])
    delta_t_s = float(cfg["simulation_environment"]["delta_t_s"])
    obstacles = set(int(cell) for cell in fixed_data.obstacles)
    grid = _build_grid(rows, cols, obstacles)
    distmap = bfs_dist(grid, map_data.base_xy)
    pois: List[POI] = []
    poi_cells: List[int] = []
    for entry in fixed_data.pois:
        cell = int(entry["cell_id"])
        row, col = divmod(cell, cols)
        terrain = entry.get("terrain") or map_data.cell_info(cell)[2]
        risk = entry.get("risk_level") or map_data.cell_info(cell)[3]
        priority = entry.get("priority_level") or RISK_PRIORITY[risk]
        n_persons = int(entry.get("n_persons", int(rng.choice(N_PERSONS_VALUES, p=N_PERSONS_PROBS))))
        dwell_base = int(entry.get("dwell_base", _sample_dwell_base(rng, risk)))
        tw_json = entry.get("time_window_s")
        tw = None
        if tw_json:
            start = int(round(tw_json["start"] / delta_t_s))
            end = int(round(tw_json["end"] / delta_t_s))
            tw = {"tmin": max(0, start), "tmax": min(horizon_ticks, end)}
        poi = POI(y=row, x=col, dwell_ticks=dwell_base, priority=priority, tw=tw)
        poi.dwell_ticks_eff = _service_ticks(dwell_base, n_persons)
        poi.n_persons = n_persons
        poi.poi_id = entry.get("poi_id", len(pois) + 1)
        poi.terrain = terrain
        poi.risk_level = risk
        pois.append(poi)
        poi_cells.append(cell)
    split_key = split if split in cfg["uav_specs"]["E_max"] else "train"
    E_max = float(cfg["uav_specs"]["E_max"][split_key])
    E_reserve = float(cfg["uav_specs"]["E_reserve"][split_key])
    e_ortho = float(cfg["uav_specs"]["energy_model"]["e_move_ortho"])
    e_diag_all = cfg["uav_specs"]["energy_model"]["e_move_diag"]
    e_diag = float(e_diag_all[split_key]) if isinstance(e_diag_all, dict) else float(e_diag_all)
    e_wait = float(cfg["uav_specs"]["energy_model"].get("e_wait", 0.0))
    energy_map = energy_dist_map(grid, map_data.base_xy, e_move_ortho=e_ortho, e_move_diag=e_diag)
    n_uav_cfg = cfg["uav_specs"]["n_uavs"][split]
    default_n_uavs = int(rng.integers(int(n_uav_cfg[0]), int(n_uav_cfg[1]) + 1))
    fixed_overrides = cfg.get("lurigancho_fixed_overrides", {})
    horizon_ticks = int(fixed_overrides.get("horizon_ticks", horizon_ticks))
    n_uavs = int(fixed_overrides.get("n_uavs", default_n_uavs))
    uavs = [UAV(uid=i, pos=map_data.base_xy, E=E_max) for i in range(n_uavs)]
    instance = {
        "grid": grid,
        "base_xy": map_data.base_xy,
        "pois": pois,
        "uavs": uavs,
        "distmap": distmap,
        "energy_map": energy_map,
        "E_max": E_max,
        "E_reserve": E_reserve,
        "e_move_ortho": e_ortho,
        "e_move_diag": e_diag,
        "e_wait": e_wait,
        "horizon_ticks": horizon_ticks,
        "ticks_per_step": int(ticks_per_cell(cfg["routes"]["cell_distances_m"]["L_o"], cfg["uav_specs"]["cruise_speed_ms"], delta_t_s)),
        "L_o": float(cfg["routes"]["cell_distances_m"]["L_o"]),
        "L_d": float(cfg["routes"]["cell_distances_m"]["L_d"]),
    }
    eimap = LuriganchoEIMap(map_data, obstacles, poi_cells)
    hooks = {
        "on_visit": lambda y, x, m=eimap: m.mark_visit(y, x),
        "on_service": lambda y, x, m=eimap: m.mark_served(y, x),
    }
    return instance, eimap, hooks
