import math
def ticks_per_cell(L_m: float, speed_ms: float, delta_t_s: float) -> int:
    return max(1, math.ceil(L_m / max(speed_ms, 1e-9)))
