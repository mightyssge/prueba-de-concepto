# envgen/sampling.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import math
import numpy as np

# --- helpers de mapeo para perfiles en inglés/español ---

_DENSITY_ALIAS = {
    "low": "baja", "baja": "baja",
    "medium": "media", "media": "media",
    "high": "alta", "alta": "alta",
}


def _aspect_ok(h: int, w: int, amin: float, amax: float) -> bool:
    a = w / max(h, 1)
    return (a >= amin) and (a <= amax)

def _nearest_w_for_target(h: int, w_min: int, w_max: int, target: float) -> int:
    """Fallback: si falla el muestreo, elige el W más cercano al target."""
    w_t = int(round(target * h))
    candidates = [w_min, w_max, w_t, max(w_min, w_t-1), min(w_max, w_t+1)]
    cand = [w for w in candidates if w_min <= w <= w_max]
    return min(cand, key=lambda w: abs((w / max(h,1)) - target))

def sample_grid_size(cfg_split: Dict[str, Any], rng: np.random.Generator) -> Tuple[int, int]:
    """
    Muestrea (H,W) de TODOS los pares factibles (por aspecto) con sesgo hacia el target.
    Soporta enforce='filter' (sesgo hacia target) o modo laxo (uniforme).
    """
    h_min, w_min = cfg_split["min"]
    h_max, w_max = cfg_split["max"]
    ar = cfg_split.get("aspect_ratio", {})
    a_target = float(ar.get("target", 2.73))
    a_min    = float(ar.get("min", 2.2))
    a_max    = float(ar.get("max", 3.2))
    enforce  = ar.get("enforce", "filter").lower()

    # 1) Construye TODOS los pares (H,W) factibles por relación de aspecto
    pairs, weights = [], []
    eps = 1e-9
    for H in range(h_min, h_max + 1):
        # Pre-filtra Ws por aspecto para este H
        feasible_ws = [W for W in range(w_min, w_max + 1) if _aspect_ok(H, W, a_min, a_max)]
        for W in feasible_ws:
            pairs.append((H, W))
            if enforce == "filter":
                a = W / H
                # peso inverso a la distancia al target (más cerca ⇒ más prob)
                w = 1.0 / (abs(a - a_target) + eps)
            else:
                # modo laxo: uniforme
                w = 1.0
            weights.append(w)

    if pairs:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(pairs), p=weights))
        return pairs[idx]

    # 2) Fallback robusto si no hay pares factibles (debería ser raro)
    H = int(rng.integers(h_min, h_max + 1))
    W = _nearest_w_for_target(H, w_min, w_max, a_target)
    return H, W

def sample_p_obs(density_range: tuple[float, float], rng: np.random.Generator) -> float:
    lo, hi = density_range
    return float(rng.uniform(lo, hi))

def _resolve_profile_cfg(n_pois_cfg_split: Dict[str, Any],
                         profiles_root: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    n_pois_cfg_split: p.ej. {"density_level": "medium"}
    profiles_root: dict con "baja", "media", "alta": {"d":..., "jitter":...}
    """
    lvl = n_pois_cfg_split.get("density_level", "media")
    lvl = _DENSITY_ALIAS.get(lvl, "media")  # normaliza idioma, x si acaso lo pongo en inglés
    prof = profiles_root.get(lvl)
    if not prof:
        # fallback seguro
        prof = profiles_root.get("media") or {"d": 0.08, "jitter": 0.01}
    return prof

def sample_n_pois(H: int, W: int,
                  n_pois_cfg_split: Dict[str, Any],
                  limits: tuple[int,int],
                  profiles_root: Dict[str, Dict[str, float]],
                  rng: np.random.Generator) -> int:
    """
    Calcula N_pois usando densidad d ± jitter, redondeo y recorte a límites.
    No verifica celdas libres aún (eso se valida después de generar obstáculos).
    """
    prof = _resolve_profile_cfg(n_pois_cfg_split, profiles_root)
    d      = float(prof.get("d", 0.08))
    jitter = float(prof.get("jitter", 0.01))
    # densidad efectiva con ruido uniforme en [-jitter, +jitter]
    d_eff = d + rng.uniform(-jitter, +jitter)
    d_eff = float(np.clip(d_eff, 0.0, 1.0))
    # N = d_eff * área
    area = H * W
    N = int(round(d_eff * area))
    N = int(np.clip(N, limits[0], limits[1]))
    return N
