from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import safe_clip


@dataclass(frozen=True)
class ActionMapperConfig:
    safety_scale: float = 0.85
    max_delta: list[float] | float | None = None
    action_clip: float = 1.0


class ResidualJointTargetActionMapper:
    """Map bounded policy actions to residual joint-position targets and PD torques."""

    def __init__(
        self,
        nominal_qpos: np.ndarray,
        joint_limits: np.ndarray,
        torque_limits: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        cfg: ActionMapperConfig,
    ) -> None:
        self.nominal_qpos = np.asarray(nominal_qpos, dtype=np.float64).reshape(-1)
        self.joint_limits = np.asarray(joint_limits, dtype=np.float64).reshape(-1, 2)
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1, 2)
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        self.cfg = cfg

        if self.joint_limits.shape[0] != self.nominal_qpos.size:
            raise ValueError("joint_limits and nominal_qpos dimensions do not match.")
        if self.torque_limits.shape[0] != self.nominal_qpos.size:
            raise ValueError("torque_limits and nominal_qpos dimensions do not match.")

        delta_low = self.joint_limits[:, 0] - self.nominal_qpos
        delta_high = self.joint_limits[:, 1] - self.nominal_qpos
        delta_low *= float(cfg.safety_scale)
        delta_high *= float(cfg.safety_scale)

        if cfg.max_delta is not None:
            max_delta = np.asarray(cfg.max_delta, dtype=np.float64)
            if max_delta.ndim == 0:
                max_delta = np.full_like(self.nominal_qpos, float(max_delta))
            if max_delta.shape != self.nominal_qpos.shape:
                raise ValueError("control.max_delta must be a scalar or one value per action.")
            delta_low = np.maximum(delta_low, -np.abs(max_delta))
            delta_high = np.minimum(delta_high, np.abs(max_delta))

        self.delta_low = delta_low
        self.delta_high = delta_high

    @property
    def num_actions(self) -> int:
        return int(self.nominal_qpos.size)

    def action_to_target_qpos(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(self.num_actions)
        action = np.clip(action, -float(self.cfg.action_clip), float(self.cfg.action_clip))
        alpha = 0.5 * (action + 1.0)
        delta_q = self.delta_low + alpha * (self.delta_high - self.delta_low)
        target_qpos = self.nominal_qpos + delta_q
        return safe_clip(target_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def compute_torque(self, target_qpos: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        target_qpos = np.asarray(target_qpos, dtype=np.float64).reshape(self.num_actions)
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.num_actions)
        qvel = np.asarray(qvel, dtype=np.float64).reshape(self.num_actions)
        torque = self.kp * (target_qpos - qpos) - self.kd * qvel
        return safe_clip(torque, self.torque_limits[:, 0], self.torque_limits[:, 1])

