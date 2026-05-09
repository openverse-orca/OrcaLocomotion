from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quat_wxyz_to_rotmat
from .obs_builder import LocomotionTaskState


@dataclass(frozen=True)
class TerminationConfig:
    min_base_height: float = 0.18
    max_base_height: float = 0.75
    max_tilt_rad: float = 0.95
    terminate_on_base_contact: bool = True
    terminate_on_illegal_contact: bool = False


class TerminationManager:
    def __init__(self, cfg: TerminationConfig) -> None:
        self.cfg = cfg

    def check(
        self,
        state: LocomotionTaskState,
        base_contact: bool,
        illegal_contact: bool = False,
    ) -> tuple[bool, dict[str, float]]:
        rot = quat_wxyz_to_rotmat(state.base_quat)
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        max_tilt_cos = np.cos(float(self.cfg.max_tilt_rad))

        too_low = bool(state.base_pos[2] < self.cfg.min_base_height)
        too_high = bool(state.base_pos[2] > self.cfg.max_base_height)
        too_tilted = bool(projected_gravity[2] > -max_tilt_cos)
        invalid = bool(
            not np.all(np.isfinite(state.base_pos))
            or not np.all(np.isfinite(state.base_quat))
            or not np.all(np.isfinite(state.qpos))
            or not np.all(np.isfinite(state.qvel))
        )
        base_hit = bool(base_contact and self.cfg.terminate_on_base_contact)
        illegal_hit = bool(illegal_contact and self.cfg.terminate_on_illegal_contact)
        terminated = too_low or too_high or too_tilted or invalid or base_hit or illegal_hit
        return terminated, {
            "/termination/too_low": float(too_low),
            "/termination/too_high": float(too_high),
            "/termination/too_tilted": float(too_tilted),
            "/termination/base_contact": float(base_hit),
            "/termination/illegal_contact": float(illegal_hit),
            "/termination/invalid": float(invalid),
        }
