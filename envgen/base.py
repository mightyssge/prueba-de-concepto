import numpy as np

def sample_base_on_perimeter(grid: np.ndarray, rng: np.random.Generator) -> tuple[int,int]:
    H, W = grid.shape
    # Buscar todas las celdas libres (False) del perímetro
    candidates = []
    for x in range(W):
        if not grid[0, x]:   candidates.append((0, x))
        if not grid[H-1, x]: candidates.append((H-1, x))
    for y in range(H):
        if not grid[y, 0]:   candidates.append((y, 0))
        if not grid[y, W-1]: candidates.append((y, W-1))

    if not candidates:
        raise RuntimeError("No hay celdas libres en el perímetro. Revisa p_obs o clear_perim.")

    # Elegir una aleatoriamente (reproducible por semilla)
    return candidates[rng.integers(0, len(candidates))]
