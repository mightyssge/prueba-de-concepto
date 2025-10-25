from typing import List, Tuple
import numpy as np

def greedy_follow_distmap(distmap: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    if np.isinf(distmap[goal]): return []
    path = []
    y, x = start
    H, W = distmap.shape
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    visited = set()
    for _ in range(H*W):
        if (y,x) == goal: break
        visited.add((y,x))
        best = None; bestd = distmap[y,x]
        for dy,dx in nbrs:
            ny,nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W and (ny,nx) not in visited:
                d = distmap[ny,nx]
                if d < bestd: bestd, best = d, (ny,nx)
        if best is None: break
        path.append(best); y,x = best
    return path
