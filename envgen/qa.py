# envgen/qa.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

INF = 10**9

def qa_connectivity(pois_xy: List[Tuple[int,int]], distmap: np.ndarray) -> Dict[str, Any]:
    d = [int(distmap[y, x]) for (y, x) in pois_xy]
    unreachable = sum(1 for v in d if v >= INF)
    reachable = len(d) - unreachable
    return {
        "n_total": len(d),
        "n_reachable": reachable,
        "pct_reachable": 100.0 * reachable / max(len(d), 1),
        "ok": (unreachable == 0),
        "detail": {"unreachable": unreachable}
    }

def qa_viability(pois_eval: List[Dict[str, Any]], min_pct_ok: float = 95.0) -> Dict[str, Any]:
    n_total = len(pois_eval)
    n_ok = sum(1 for p in pois_eval if p.get("feasible", False))
    pct = 100.0 * n_ok / max(n_total, 1)
    return {"n_total": n_total, "n_ok": n_ok, "pct_ok": pct, "ok": (pct >= min_pct_ok)}

def qa_distributions(
    pois: List[Dict[str, Any]],
    *,
    max_priority_mode_ratio: float = 0.80
) -> Dict[str, Any]:
    # prioridad
    pr = np.array([p["priority"] for p in pois], dtype=int) if pois else np.array([], dtype=int)
    uniq_pr, counts_pr = np.unique(pr, return_counts=True) if pr.size > 0 else (np.array([]), np.array([]))
    pr_ok = (uniq_pr.size >= 2) and ((counts_pr.max()/max(counts_pr.sum(),1)) <= max_priority_mode_ratio)

    # dwell
    dw = np.array([p["dwell_ticks"] for p in pois], dtype=int) if pois else np.array([], dtype=int)
    dw_ok = (dw.size == 0) or (dw.max() - dw.min() >= 1)

    # ventanas (si existen)
    widths = np.array([(p["tw"]["tmax"] - p["tw"]["tmin"]) for p in pois if p.get("tw") is not None], dtype=int)
    has_tw = (widths.size > 0)
    tw_ok = True  # no lo usamos para bloquear, solo informativo

    return {
        "priority": {"unique_levels": int(uniq_pr.size), "mode_ratio": float(counts_pr.max()/max(counts_pr.sum(),1) if counts_pr.size else 0.0), "ok": bool(pr_ok)},
        "dwell": {"min": int(dw.min()) if dw.size else None, "max": int(dw.max()) if dw.size else None, "ok": bool(dw_ok)},
        "tw": {"has_any": bool(has_tw), "min_width": int(widths.min()) if has_tw else None, "max_width": int(widths.max()) if has_tw else None, "ok": bool(tw_ok)},
        "ok": bool(pr_ok and dw_ok and tw_ok)
    }

def qa_summary(connect, energy, dists) -> Dict[str, Any]:
    """Resumen final y bandera global."""
    return {
        "connectivity_ok": bool(connect["ok"]),
        "viability_ok": bool(energy["ok"]),
        "distributions_ok": bool(dists["ok"]),
        "ALL_OK": bool(connect["ok"] and energy["ok"] and dists["ok"]),
        "sections": {"connectivity": connect, "viability": energy, "distributions": dists}
    }
