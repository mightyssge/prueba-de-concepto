# envgen/persist.py
from __future__ import annotations
import os, json, csv
from typing import Dict, Any
import numpy as np

def save_instance_npz(outdir: str, stem: str, grid: np.ndarray, distmap: np.ndarray | None = None,
                      energy_map: np.ndarray | None = None):
    path = os.path.join(outdir, f"{stem}.npz")
    np.savez_compressed(path, grid=grid, distmap=distmap, energy_map=energy_map)
    return path

def save_instance_json(outdir: str, stem: str, meta: Dict[str, Any]):
    path = os.path.join(outdir, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

INDEX_FIELDS = [
    "run_seed","split","k","tries","H","W","p_obs","density_real","perimeter_free",
    "N_pois","pct_reachable","pct_viable",
    "prio_levels","prio_mode_ratio","dwell_min","dwell_max","tw_any","tw_min","tw_max",
    "E_round_min","E_round_med","E_round_mean","E_round_max","E_max","E_reserve",
    "stem"
]

def append_index_row(outdir: str, row: Dict[str, Any]):
    csv_path = os.path.join(outdir, "index.csv")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INDEX_FIELDS})
    return csv_path
