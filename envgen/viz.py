# envgen/viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def plot_grid_with_base(grid: np.ndarray, base_xy: tuple[int,int], *, title: str = "", savepath: str | None = None):
    """
    Muestra la grilla (True=obstáculo, False=libre) y marca la base en verde.
    Si savepath se proporciona, guarda el PNG en disco (no bloquea).
    """
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(W/5, H/5))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")  # obstáculo=negro, libre=blanco
    by, bx = base_xy
    ax.scatter([bx], [by], s=60, marker="s", edgecolors="black", facecolors="none", linewidths=1.5, label="Base")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_grid_with_base_pois(
    grid: np.ndarray,
    base_xy: tuple[int,int],
    pois: list[dict] | None = None,
    *,
    highlight_idx: int | None = None,  # índice del POI a resaltar
    title: str = "",
    savepath: str | None = None
):
    """
    Dibuja la grilla (True=obstáculo, False=libre), la base (verde) y POIs.
    - Obstáculos: gris/negro (cmap="gray_r")
    - Base: cuadrado borde verde
    - POIs: círculos rojos
    - POI destacado: azul/cian y anotación con atributos (priority, dwell, TW si existe)
    """
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(max(W/5, 4), max(H/5, 4)))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    # Base (cuadrado con borde verde)
    by, bx = base_xy
    ax.scatter([bx], [by], s=70, marker="s",
               facecolors="none", edgecolors="limegreen", linewidths=2, label="Base")

    # POIs (todos en rojo)
    if pois:
        xs = [p["x"] for p in pois]
        ys = [p["y"] for p in pois]
        ax.scatter(xs, ys, s=35, marker="o",
                   facecolors="red", edgecolors="black", linewidths=0.5, label="POIs")

        # POI destacado (si se indicó)
        if highlight_idx is not None and 0 <= highlight_idx < len(pois):
            p = pois[highlight_idx]
            hx, hy = p["x"], p["y"]
            ax.scatter([hx], [hy], s=80, marker="o",
                       facecolors="deepskyblue", edgecolors="black", linewidths=0.8,
                       label=f"POI* #{highlight_idx}")

            # Construir texto con atributos
            pr = p.get("priority", "?")
            dw = p.get("dwell_ticks", "?")
            tw = p.get("tw", None)
            if tw is None:
                tw_str = "TW: —"
            else:
                tw_str = f"TW: [{tw.get('tmin','?')},{tw.get('tmax','?')}]"

            info = f"POI* #{highlight_idx}  (y={hy}, x={hx})\nprio={pr} | dwell={dw} | {tw_str}"

            # Anotación (cuadro blanco semitransparente)
            ax.annotate(
                info,
                xy=(hx, hy),
                xytext=(10, -10),
                textcoords="offset points",
                ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
            )

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def plot_distance_heatmap(grid, distmap, base_xy, pois_xy=None, savepath=None, title=""):
    """Unreachable en gris, mayor distancia en más intenso."""
    H, W = grid.shape
    INF = 10**9
    # mascaras
    m = distmap.astype(float)
    m[m >= INF] = np.nan  # NaN = inalcanzable
    fig, ax = plt.subplots(figsize=(max(W/5,4), max(H/5,4)))
    im = ax.imshow(m, cmap="viridis", interpolation="nearest")
    # fondo gris para obstáculos e inalcanzables
    obst_or_nan = grid | np.isnan(m)
    ax.imshow(obst_or_nan, cmap="gray", alpha=0.35, interpolation="nearest")
    # overlays
    by, bx = base_xy
    ax.scatter([bx], [by], s=70, marker="s",
               facecolors="none", edgecolors="limegreen", linewidths=2, label="Base")
    if pois_xy:
        xs = [x for (y,x) in pois_xy]
        ys = [y for (y,x) in pois_xy]
        ax.scatter(xs, ys, s=20, facecolors="red", edgecolors="black", linewidths=0.3, label="POIs")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True)
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Distancia (celdas)", rotation=270, labelpad=10)
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()

def plot_energy_hist_cdf(E_round: np.ndarray, E_max: float, savepath: str | None = None, title: str = ""):
    """Histograma y CDF de energía ida+vuelta; traza E_max para referencia."""
    if E_round.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6,4))
    # Histograma
    ax.hist(E_round, bins=20, density=True, alpha=0.55, label="hist(E_round)")
    # CDF
    xs = np.sort(E_round)
    cdf = np.arange(1, len(xs)+1) / len(xs)
    ax2 = ax.twinx()
    ax2.plot(xs, cdf, lw=2, label="CDF(E_round)")
    # Límites y líneas de referencia
    ax.axvline(E_max, color="red", lw=1.5, ls="--", label=f"E_max={E_max:.1f}")
    ax.set_xlabel("E_round (unidades de energía)")
    ax.set_ylabel("densidad")
    ax2.set_ylabel("CDF")
    ax.set_title(title or "Distribución de E_round")
    # Leyendas
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc="lower right")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()

def plot_energy_vs_eta(E_round: np.ndarray, ETA_ticks: np.ndarray, E_max: float,
                       savepath: str | None = None, title: str = ""):
    """Dispersión E_round vs ETA_ticks con línea de referencia en E_max."""
    if E_round.size == 0 or ETA_ticks.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(ETA_ticks, E_round, s=20, edgecolors="black", linewidths=0.3)
    ax.axhline(E_max, color="red", lw=1.5, ls="--", label=f"E_max={E_max:.1f}")
    ax.set_xlabel("ETA (ticks)")
    ax.set_ylabel("E_round (unidades de energía)")
    ax.set_title(title or "E_round vs ETA")
    ax.legend(loc="best")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()

def plot_feasibility_bars(n_feas: int, n_total: int, savepath: str | None = None, title: str = ""):
    """Bar chart simple de factibilidad (feasible vs no feasible)."""
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    vals = [n_feas, max(n_total - n_feas, 0)]
    labels = ["feasible", "no feasible"]
    ax.bar(labels, vals)
    ax.set_ylim(0, max(vals)+1)
    ax.set_title(title or "Feasibility")
    for i,v in enumerate(vals):
        ax.text(i, v+0.2, str(v), ha="center", va="bottom")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()



def plot_trajectories(grid: np.ndarray,
                      base_xy: Tuple[int,int],
                      pois_xy: List[Tuple[int,int]],
                      traj: List[Dict],
                      savepath: str | None = None,
                      title: str = "Trayectorias (cada N ticks)"):
    """
    Dibuja la grilla y las posiciones de los UAVs registradas en 'traj'.
    'traj' es la lista devuelta por simulate_episode['traj'].
    """
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(max(W/5, 4), max(H/5, 4)))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    # Base
    by, bx = base_xy
    ax.scatter([bx], [by], s=80, marker="s", facecolors="none",
               edgecolors="limegreen", linewidths=2, label="Base")

    # POIs
    if pois_xy:
        xs = [x for (y, x) in pois_xy]
        ys = [y for (y, x) in pois_xy]
        ax.scatter(xs, ys, s=28, marker="o", facecolors="red",
                   edgecolors="black", linewidths=0.5, label="POIs")

    # Trayectorias: para cada UAV, unir sus puntos en el tiempo
    # Construimos un dict uid -> listas de (x_t, y_t)
    paths = {}
    for snap in traj:
        for uid, y, x in snap["uavs"]:
            paths.setdefault(uid, {"x": [], "y": []})
            paths[uid]["x"].append(x)
            paths[uid]["y"].append(y)

    # Plot de líneas por UAV
    for uid, p in sorted(paths.items()):
        ax.plot(p["x"], p["y"], linewidth=1.5, label=f"UAV {uid}")

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150); plt.close(fig)
    else:
        plt.show()

def plot_snapshot_frame(
    grid: np.ndarray,
    base_xy: Tuple[int,int],
    pois_xy: List[Tuple[int,int]],
    uavs_xy: List[Tuple[int,int,int]],  # [(uid,y,x), ...]
    *,
    title: str,
    savepath: str
):
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(max(W/5, 4), max(H/5, 4)))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    # base
    by, bx = base_xy
    ax.scatter([bx], [by], s=80, marker="s", facecolors="none",
               edgecolors="limegreen", linewidths=2, label="Base")

    # POIs
    if pois_xy:
        xs = [x for (y, x) in pois_xy]
        ys = [y for (y, x) in pois_xy]
        ax.scatter(xs, ys, s=28, marker="o", facecolors="red",
                   edgecolors="black", linewidths=0.5, label="POIs")

    # UAVs
    if uavs_xy:
        xs = [x for (_, y, x) in uavs_xy]
        ys = [y for (_, y, x) in uavs_xy]
        ax.scatter(xs, ys, s=36, marker="^", facecolors="deepskyblue",
                   edgecolors="black", linewidths=0.5, label="UAVs")

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

def plot_trajectories_full(
    grid: np.ndarray,
    base_xy: Tuple[int,int],
    pois_xy: List[Tuple[int,int]],
    paths: Dict[int, List[Tuple[int,int]]],
    *,
    title: str,
    savepath: str
):
    """
    Dibuja el mapa y, para cada UAV, la polilínea de TODO su recorrido (tick a tick).
    'paths' es un dict: uid -> [(y,x), (y,x), ...] a lo largo del tiempo.
    """
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(max(W/5, 4), max(H/5, 4)))
    ax.imshow(grid, cmap="gray_r", interpolation="nearest")

    # Base
    by, bx = base_xy
    ax.scatter([bx], [by], s=80, marker="s", facecolors="none",
               edgecolors="limegreen", linewidths=2, label="Base")

    # POIs
    if pois_xy:
        xs = [x for (y, x) in pois_xy]
        ys = [y for (y, x) in pois_xy]
        ax.scatter(xs, ys, s=28, marker="o", facecolors="red",
                   edgecolors="black", linewidths=0.5, label="POIs")

    # Polilíneas de cada UAV
    for uid, seq in sorted(paths.items()):
        if not seq: continue
        xs = [x for (y, x) in seq]
        ys = [y for (y, x) in seq]
        ax.plot(xs, ys, linewidth=1.8, label=f"UAV {uid}")
        # Marca final
        ax.scatter([xs[-1]], [ys[-1]], s=40, marker="^", edgecolors="black", facecolors="deepskyblue")

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)