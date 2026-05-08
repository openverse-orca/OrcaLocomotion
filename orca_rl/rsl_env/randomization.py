from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool = True
    friction_range: tuple[float, float] = (0.7, 1.3)
    base_mass_delta_range: tuple[float, float] = (-0.5, 1.5)


@dataclass(frozen=True)
class RandomizationState:
    friction_scale: float = 1.0
    base_mass_delta: float = 0.0


class DomainRandomizer:
    def __init__(self, cfg: RandomizationConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> RandomizationState:
        if not self.cfg.enabled:
            return RandomizationState()
        friction_scale = float(self.rng.uniform(*self.cfg.friction_range))
        base_mass_delta = float(self.rng.uniform(*self.cfg.base_mass_delta_range))
        return RandomizationState(friction_scale=friction_scale, base_mass_delta=base_mass_delta)

