# envgen/pois.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any

def place_pois(
    grid: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    forbid_xy: Tuple[int,int] | None = None
) -> List[Tuple[int,int]]:
    """
    Selecciona 'n' celdas libres (False) sin solape.
    - grid[True]=obstáculo, grid[False]=libre
    - forbid_xy: coord (y,x) a excluir (e.g., base)
    """
    H, W = grid.shape
    libres = np.column_stack(np.where(~grid))  # [ [y,x], ... ]
    if forbid_xy is not None:
        fy, fx = forbid_xy
        mask = ~((libres[:, 0] == fy) & (libres[:, 1] == fx))
        libres = libres[mask]

    if len(libres) < n:
        raise RuntimeError(f"No hay celdas libres suficientes para {n} POIs (quedan {len(libres)}).")

    idx = rng.choice(len(libres), size=n, replace=False)
    puntos = libres[idx]
    return [(int(y), int(x)) for y, x in puntos]

def _sample_priority(levels: List[int], weights: List[float], rng: np.random.Generator) -> int:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    i = int(rng.choice(len(levels), p=w))
    return int(levels[i])

def assign_attributes(
    pois_xy: List[Tuple[int,int]],
    cfg_pois: Dict[str, Any],
    *,
    eta_fn,
    delta_t_s: float,
    horizon_ticks: int,
    rng: np.random.Generator
) -> List[Dict[str, Any]]:

    out = []

    # --- nuevos parámetros desde config.json ---
    n_max  = int(cfg_pois.get("n_persons_max", 4))
    probs  = np.asarray(cfg_pois.get("n_persons_probs",
                     [0.55, 0.25, 0.10, 0.06, 0.04]), dtype=float)
    k_tick = int(cfg_pois.get("extra_tick_per_person", 1))

    if probs.size != (n_max + 1) or probs.sum() <= 0:
        probs = np.ones(n_max + 1, dtype=float) / (n_max + 1)
    else:
        probs = probs / probs.sum()

    for (y, x) in pois_xy:
        # prioridad
        prio = int(rng.choice(cfg_pois["priority_levels"]))

        # dwell base
        dmin, dmax = cfg_pois["dwell_ticks_range"]
        dwell = int(rng.integers(dmin, dmax + 1))

        # ventana temporal
        tw_lo_s, tw_hi_s = cfg_pois["time_window_s"]
        rho_min, rho_max = cfg_pois["tightness_rho"]
        rho = float(rng.random() * (rho_max - rho_min) + rho_min)
        eta = eta_fn((y, x))
        if np.isfinite(eta):
            W_s = tw_hi_s - rho * (tw_hi_s - tw_lo_s)
            W_ticks = max(1, int(np.round(W_s / max(delta_t_s, 1e-9))))
            tmin = int(eta)
            tmax = int(min(horizon_ticks, tmin + W_ticks))
            tw = {"tmin": tmin, "tmax": tmax}
        else:
            tw = None

        # --- NUEVO: personas y dwell efectivo ---
        n_persons = int(rng.choice(np.arange(n_max + 1), p=probs))
        dwell_eff = int(dwell + k_tick * n_persons)

        out.append({
            "y": y, "x": x,
            "priority": prio,
            "dwell_ticks": dwell,
            "dwell_ticks_eff": dwell_eff,   # para simulación
            "n_persons": n_persons,
            "tw": tw
        })

    return out