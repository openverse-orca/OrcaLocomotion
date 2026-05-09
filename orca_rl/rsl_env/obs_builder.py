from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quat_wxyz_to_rotmat


@dataclass
class LocomotionTaskState:
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_lin_vel_world: np.ndarray
    base_ang_vel_world: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    command: np.ndarray
    last_action: np.ndarray
    last_torque: np.ndarray
    foot_pos_world: np.ndarray
    foot_vel_world: np.ndarray
    foot_contacts: np.ndarray
    friction_scale: float
    base_mass_delta: float


@dataclass(frozen=True)
class ObservationConfig:
    add_noise: bool = True
    noise_level: float = 1.0
    ang_vel_scale: float = 0.25
    lin_vel_scale: float = 2.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    command_scale: tuple[float, float, float] = (2.0, 2.0, 0.25)
    height_scale: float = 5.0
    height_scan_dim: int = 0
    height_scan_scale: float = 1.0


class LocomotionObservationBuilder:
    def __init__(
        self,
        nominal_qpos: np.ndarray,
        cfg: ObservationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.nominal_qpos = np.asarray(nominal_qpos, dtype=np.float64).reshape(-1)
        self.cfg = cfg
        self.rng = rng

    def build(self, state: LocomotionTaskState, noisy: bool) -> dict[str, np.ndarray]:
        rot = quat_wxyz_to_rotmat(state.base_quat)
        base_lin_vel_body = rot.T @ state.base_lin_vel_world
        base_ang_vel_body = rot.T @ state.base_ang_vel_world
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

        command_scale = np.asarray(self.cfg.command_scale, dtype=np.float64)
        height_scan = self._height_scan()
        policy_terms = [
            base_ang_vel_body * self.cfg.ang_vel_scale,
            projected_gravity,
            state.command * command_scale,
            (state.qpos - self.nominal_qpos) * self.cfg.dof_pos_scale,
            state.qvel * self.cfg.dof_vel_scale,
            state.last_action,
        ]
        if height_scan.size:
            policy_terms.append(height_scan)
        policy = np.concatenate(policy_terms).astype(np.float32)

        if noisy and self.cfg.add_noise:
            policy = policy + self._policy_noise(policy.shape)

        foot_heights = state.foot_pos_world[:, 2]
        foot_vel_body = (rot.T @ state.foot_vel_world.T).T
        privileged_terms = [
            base_lin_vel_body * self.cfg.lin_vel_scale,
            base_ang_vel_body * self.cfg.ang_vel_scale,
            projected_gravity,
            np.array([state.base_pos[2] * self.cfg.height_scale], dtype=np.float64),
            state.foot_contacts,
            foot_heights,
            foot_vel_body.reshape(-1),
            state.last_torque,
            np.array([state.friction_scale, state.base_mass_delta], dtype=np.float64),
        ]
        if height_scan.size:
            privileged_terms.append(height_scan)
        privileged = np.concatenate(privileged_terms).astype(np.float32)

        return {
            "policy": policy,
            "privileged": privileged,
        }

    def _policy_noise(self, shape: tuple[int, ...]) -> np.ndarray:
        scale = 0.01 * float(self.cfg.noise_level)
        return self.rng.uniform(-scale, scale, size=shape).astype(np.float32)

    def _height_scan(self) -> np.ndarray:
        dim = max(0, int(self.cfg.height_scan_dim))
        if dim == 0:
            return np.zeros(0, dtype=np.float64)
        return np.zeros(dim, dtype=np.float64) * float(self.cfg.height_scan_scale)
