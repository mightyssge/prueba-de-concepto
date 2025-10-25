import numpy as np
from collections import Counter
from math import log2
from .obstacles import realized_density, perimeter_free_ratio


def entropy_counts(counts: dict[int,int]) -> float:
    total = sum(counts.values()) or 1
    H = 0.0
    for c in counts.values():
        p = c/total
        if p > 0: H -= p*log2(p)
    return H


def summarize_obstacles(grid):
    return {
        "density_real": realized_density(grid),
        "perimeter_free_ratio": perimeter_free_ratio(grid),
    }