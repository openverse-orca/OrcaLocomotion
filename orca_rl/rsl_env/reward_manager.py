from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quat_wxyz_to_rotmat
from .obs_builder import LocomotionTaskState


@dataclass(frozen=True)
class RewardConfig:
    tracking_lin_vel: float = 1.5
    tracking_ang_vel: float = 0.75
    lin_vel_sigma: float = 0.25
    ang_vel_sigma: float = 0.25
    z_vel: float = -1.0
    orientation: float = -2.0
    height: float = -1.0
    torque: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -1.0
    foot_slip: float = -0.1
    termination: float = -2.0
    target_height: float = 0.34


class FlatVelocityReward:
    def __init__(self, joint_limits: np.ndarray, cfg: RewardConfig) -> None:
        self.joint_limits = np.asarray(joint_limits, dtype=np.float64).reshape(-1, 2)
        self.cfg = cfg

    def compute(
        self,
        state: LocomotionTaskState,
        action: np.ndarray,
        previous_action: np.ndarray,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        rot = quat_wxyz_to_rotmat(state.base_quat)
        lin_vel_body = rot.T @ state.base_lin_vel_world
        ang_vel_body = rot.T @ state.base_ang_vel_world
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

        lin_error = np.sum(np.square(state.command[:2] - lin_vel_body[:2]))
        yaw_error = float(np.square(state.command[2] - ang_vel_body[2]))
        rew_lin = self.cfg.tracking_lin_vel * np.exp(-lin_error / self.cfg.lin_vel_sigma)
        rew_yaw = self.cfg.tracking_ang_vel * np.exp(-yaw_error / self.cfg.ang_vel_sigma)

        pen_z = self.cfg.z_vel * float(np.square(lin_vel_body[2]))
        pen_orientation = self.cfg.orientation * float(np.sum(np.square(projected_gravity[:2])))
        pen_height = self.cfg.height * float(np.square(state.base_pos[2] - self.cfg.target_height))
        pen_torque = self.cfg.torque * float(np.sum(np.square(state.last_torque)))
        pen_action_rate = self.cfg.action_rate * float(np.sum(np.square(action - previous_action)))
        pen_joint_limit = self.cfg.joint_limit * self._joint_limit_violation(state.qpos)
        foot_speed_xy = np.linalg.norm(state.foot_vel_world[:, :2], axis=1)
        pen_slip = self.cfg.foot_slip * float(np.sum(foot_speed_xy * state.foot_contacts))
        pen_done = self.cfg.termination if terminated else 0.0

        terms = {
            "/reward/tracking_lin_vel": float(rew_lin),
            "/reward/tracking_ang_vel": float(rew_yaw),
            "/reward/z_vel": float(pen_z),
            "/reward/orientation": float(pen_orientation),
            "/reward/height": float(pen_height),
            "/reward/torque": float(pen_torque),
            "/reward/action_rate": float(pen_action_rate),
            "/reward/joint_limit": float(pen_joint_limit),
            "/reward/foot_slip": float(pen_slip),
            "/reward/termination": float(pen_done),
        }
        return float(sum(terms.values())), terms

    def _joint_limit_violation(self, qpos: np.ndarray) -> float:
        low_violation = np.maximum(self.joint_limits[:, 0] - qpos, 0.0)
        high_violation = np.maximum(qpos - self.joint_limits[:, 1], 0.0)
        return float(np.sum(low_violation + high_violation))

