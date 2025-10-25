from collections import deque
import numpy as np

INF = 10**9
OFFS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 vecinos

def bfs_dist(grid: np.ndarray, start: tuple[int,int]) -> np.ndarray:
    """
    Calcula distancias mínimas en celdas desde 'start' a todas las celdas libres (False).
    Obstáculos (True) = no transitables.
    Retorna matriz distancias (int32), INF donde no se puede llegar.
    """
    H, W = grid.shape
    dist = np.full((H, W), INF, dtype=np.int32)
    sy, sx = start
    if grid[sy, sx]:
        return dist  # base sobre obstáculo (debe evitarse antes)
    dist[sy, sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        for dy, dx in OFFS8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx] and dist[ny, nx] == INF:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return dist
