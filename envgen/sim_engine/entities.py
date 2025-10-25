from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal, Dict

UavState = Literal["idle","en_route","servicing","rtb","charging","queued"]

@dataclass
class UAV:
    uid: int
    pos: Tuple[int,int]
    E: float
    state: UavState = "idle"
    target: Optional[Tuple[int,int]] = None
    path: List[Tuple[int,int]] = field(default_factory=list)
    move_cooldown: int = 0
    service_left: int = 0
    energy_spent: float = 0.0
    steps_ortho: int = 0
    steps_diag: int = 0

@dataclass
class POI:
    y: int; x: int
    dwell_ticks: int
    priority: int
    tw: Optional[Dict[str,int]] = None
    served: bool = False
    first_visit_t: Optional[int] = None
    # --- NUEVO ---
    n_persons: int = 0                 # 0..4
    dwell_ticks_eff: Optional[int] = None  # base + extra por personas
    tardiness: int = 0           
    violated: bool = False       

@dataclass
class BaseStation:
    xy: Tuple[int,int]
    capacity: int
    charge_model: str
    turnaround_ticks: Tuple[int,int]
