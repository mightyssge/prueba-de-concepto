from __future__ import annotations
import numpy as np

def _perimeter_mask(H: int, W: int, width: int = 1) -> np.ndarray:
    """Máscara booleana del perímetro (banda de 'width' celdas)."""
    m = np.zeros((H, W), dtype=bool)
    w = max(1, int(width))
    m[:w, :]  = True
    m[-w:, :] = True
    m[:, :w]  = True
    m[:, -w:] = True
    return m

def realized_density(grid: np.ndarray) -> float:
    """Proporción de celdas ocupadas (True)."""
    H, W = grid.shape
    return float(grid.sum() / (H * W))

def perimeter_free_ratio(grid: np.ndarray, width: int = 1) -> float:
    """Proporción de celdas libres en el perímetro."""
    per = _perimeter_mask(*grid.shape, width=width)
    total = int(per.sum())
    libres = int((~grid & per).sum())
    return libres / max(total, 1)

def generate_obstacles(
    H: int,
    W: int,
    p_obs: float,
    rng: np.random.Generator,
    *,
    clear_perim: bool = True,
    perim_width: int = 1,
    min_free_perim_ratio: float = 0.05,
    max_tries: int = 50
) -> np.ndarray:
    """
    Genera una grilla booleana con obstáculos ~ Bernoulli(p_obs).
    - Si clear_perim=True, limpia una banda de perímetro (base en contorno).
    - Reintenta hasta 'max_tries' si el perímetro queda 100% bloqueado.
    - Retorna grid[True]=obstáculo, grid[False]=libre.
    """
    per_mask = _perimeter_mask(H, W, perim_width)

    for _ in range(max_tries):
        grid = rng.random((H, W)) < p_obs
        if clear_perim:
            grid = grid.copy()
            grid[per_mask] = False  # limpiar perímetro
        # chequear que haya al menos algo de perímetro libre
        if perimeter_free_ratio(grid, perim_width) >= min_free_perim_ratio:
            return grid

    # último intento que acepta lo que haya, pero limpia perímetro si se pidió
    grid = rng.random((H, W)) < p_obs
    if clear_perim:
        grid = grid.copy()
        grid[per_mask] = False
    return grid
