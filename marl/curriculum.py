from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CurriculumPhase:
    name: str
    obstacles: bool
    time_windows: bool
    energy: bool
    poi_multiplier: float = 1.0
    obstacle_scale: float = 1.0
    threshold: float = 0.8
    min_episodes: int = 5
    retain_ratio: float = 0.1  # fraction of old buffer to keep

    def overrides(self) -> Dict[str, float]:
        return {
            "poi_multiplier": self.poi_multiplier,
            "obstacle_scale": self.obstacle_scale,
        }


class CurriculumManager:
    def __init__(self, phases: List[CurriculumPhase], window: int = 5):
        if not phases:
            raise ValueError("At least one curriculum phase is required.")
        self.phases = phases
        self.window = max(1, window)
        self.idx = 0
        self.history: List[float] = []
        self.episodes_in_phase = 0

    @property
    def current(self) -> CurriculumPhase:
        return self.phases[self.idx]

    def update(self, metric: float) -> bool:
        """
        Updates moving average performance; advances phase if threshold reached.
        Returns True if a transition occurred.
        """
        self.history.append(metric)
        self.episodes_in_phase += 1
        recent = self.history[-self.window :]
        mean_perf = float(np.mean(recent))
        advanced = False
        if (
            self.episodes_in_phase >= self.current.min_episodes
            and mean_perf >= self.current.threshold
            and self.idx < (len(self.phases) - 1)
        ):
            self.idx += 1
            self.history.clear()
            self.episodes_in_phase = 0
            advanced = True
        return advanced

    def reset(self) -> None:
        self.idx = 0
        self.history.clear()
        self.episodes_in_phase = 0
