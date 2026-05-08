from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommandConfig:
    lin_vel_x: tuple[float, float] = (-0.8, 1.0)
    lin_vel_y: tuple[float, float] = (-0.25, 0.25)
    yaw_vel: tuple[float, float] = (-0.8, 0.8)
    resample_time_s: float = 4.0


class FlatVelocityCommandSampler:
    def __init__(self, cfg: CommandConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(*self.cfg.lin_vel_x),
                self.rng.uniform(*self.cfg.lin_vel_y),
                self.rng.uniform(*self.cfg.yaw_vel),
            ],
            dtype=np.float64,
        )

