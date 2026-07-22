"""Reward terms for Orca's G1 flat-velocity task.

Factories return closures over sensor/body names so the task config stays
declarative; every closure computes on batched GPU tensors only.
"""

from __future__ import annotations

import re

import torch

from .math_utils import quat_apply_inverse

##
# Velocity tracking.
##


def track_linear_velocity(command_name: str, std: float, lin_vel_sensor: str):
    def _term(env) -> torch.Tensor:
        command = env.get_command(command_name)
        actual = env.orca.sensor(lin_vel_sensor)
        xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
        z_error = torch.square(actual[:, 2])
        return torch.exp(-(xy_error + 2.0 * z_error) / std**2)

    return _term


def track_angular_velocity(command_name: str, std: float, ang_vel_sensor: str):
    def _term(env) -> torch.Tensor:
        command = env.get_command(command_name)
        actual = env.orca.sensor(ang_vel_sensor)
        z_error = torch.square(command[:, 2] - actual[:, 2])
        xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
        return torch.exp(-(z_error + 0.05 * xy_error) / std**2)

    return _term


##
# Posture / body attitude.
##


def body_orientation_l2(body_name: str):
    def _term(env) -> torch.Tensor:
        body_id = env.orca.body_id(body_name)
        body_quat = env.orca.state.xquat[:, body_id]
        gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        projected = quat_apply_inverse(body_quat, gravity)
        return torch.sum(torch.square(projected[:, :2]), dim=1)

    return _term


def body_angular_velocity_penalty(body_name: str):
    def _term(env) -> torch.Tensor:
        body_id = env.orca.body_id(body_name)
        ang_vel_xy = env.orca.state.cvel[:, body_id, 0:2]
        return torch.sum(torch.square(ang_vel_xy), dim=1)

    return _term


def angular_momentum_penalty(sensor_name: str):
    def _term(env) -> torch.Tensor:
        angmom = env.orca.sensor(sensor_name)
        magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
        env.extras["log"]["Metrics/angular_momentum_mean"] = torch.sqrt(magnitude_sq).mean()
        return magnitude_sq

    return _term


class variable_posture:
    """Speed-regime posture reward with per-joint standard-deviation maps."""

    def __init__(
        self,
        command_name: str,
        joint_names: tuple[str, ...],
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.1,
        running_threshold: float = 1.5,
    ):
        self.command_name = command_name
        self.joint_names = joint_names
        self.walking_threshold = walking_threshold
        self.running_threshold = running_threshold
        self._std_maps = (std_standing, std_walking, std_running)
        self._stds: tuple[torch.Tensor, ...] | None = None

    @staticmethod
    def _resolve(std_map: dict[str, float], joint_names: tuple[str, ...]) -> list[float]:
        values = []
        for joint_name in joint_names:
            for pattern, value in std_map.items():
                if re.fullmatch(pattern, joint_name):
                    values.append(float(value))
                    break
            else:
                raise ValueError(f"No posture std matches joint {joint_name!r}")
        return values

    def __call__(self, env) -> torch.Tensor:
        if self._stds is None:
            self._stds = tuple(
                torch.tensor(self._resolve(std_map, self.joint_names), device=env.device)
                for std_map in self._std_maps
            )
        command = env.get_command(self.command_name)
        total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = (total_speed < self.walking_threshold).float()
        running = (total_speed >= self.running_threshold).float()
        walking = 1.0 - standing - running
        std = (
            self._stds[0] * standing[:, None]
            + self._stds[1] * walking[:, None]
            + self._stds[2] * running[:, None]
        )
        error_sq = torch.square(env.orca.actuated_qpos() - env.orca.default_actuated_qpos())
        return torch.exp(-torch.mean(error_sq / (std**2), dim=1))


##
# Regularizers.
##


def is_terminated(env) -> torch.Tensor:
    return env.terminated_buf.float()


def joint_acc_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.orca.actuated_qacc()), dim=1)


def joint_vel_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.orca.actuated_qvel()), dim=1)


def joint_torques_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.orca.state.actuator_force), dim=1)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_action - env.prev_action), dim=1)


def action_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_action), dim=1)


class joint_pos_limits:
    """Penalize actuated joints beyond the soft limits (factor 0.9, Orca)."""

    def __init__(self, soft_limit_factor: float = 0.9):
        self.soft_limit_factor = soft_limit_factor
        self._limits: torch.Tensor | None = None

    def __call__(self, env) -> torch.Tensor:
        if self._limits is None:
            limits = env.orca.actuated_joint_range()
            mid = 0.5 * (limits[:, 0] + limits[:, 1])
            half = 0.5 * (limits[:, 1] - limits[:, 0]) * self.soft_limit_factor
            self._limits = torch.stack((mid - half, mid + half), dim=-1)
        qpos = env.orca.actuated_qpos()
        out_of_limits = -(qpos - self._limits[:, 0]).clip(max=0.0)
        out_of_limits += (qpos - self._limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)


##
# Feet terms (contact-sensor based).
##


def _command_active(env, command_name: str, threshold: float) -> torch.Tensor:
    command = env.get_command(command_name)
    total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (total > threshold).float()


def feet_gait(
    tracker,
    period: float,
    offset: tuple[float, ...],
    threshold: float,
    command_name: str,
    command_threshold: float = 0.1,
):
    def _term(env) -> torch.Tensor:
        tracker._ensure_state(env)
        is_contact = tracker.contact_time > 0
        global_phase = ((env.episode_length_buf.float() * env.step_dt) / period)[:, None]
        offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
        leg_phase = (global_phase + offsets) % 1.0
        is_stance = leg_phase < threshold
        reward = (is_stance == is_contact).float().mean(dim=1)
        return reward * _command_active(env, command_name, command_threshold)

    return _term


def feet_clearance(
    foot_pos_sensors: tuple[str, ...],
    foot_vel_sensors: tuple[str, ...],
    target_height: float,
    command_name: str,
    command_threshold: float = 0.1,
):
    def _term(env) -> torch.Tensor:
        foot_z = torch.cat(
            [env.orca.sensor(name)[:, 2:3] for name in foot_pos_sensors], dim=-1
        )
        vel_norm = torch.stack(
            [torch.norm(env.orca.sensor(name)[:, :2], dim=-1) for name in foot_vel_sensors],
            dim=-1,
        )
        cost = torch.sum(torch.abs(foot_z - target_height) * vel_norm, dim=1)
        return cost * _command_active(env, command_name, command_threshold)

    return _term


def feet_slip(
    tracker,
    foot_vel_sensors: tuple[str, ...],
    command_name: str,
    command_threshold: float = 0.1,
):
    def _term(env) -> torch.Tensor:
        in_contact = (tracker.found(env) > 0).float()
        vel_norm = torch.stack(
            [torch.norm(env.orca.sensor(name)[:, :2], dim=-1) for name in foot_vel_sensors],
            dim=-1,
        )
        cost = torch.sum(torch.square(vel_norm) * in_contact, dim=1)
        num_in_contact = torch.sum(in_contact)
        env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(vel_norm * in_contact) / torch.clamp(
            num_in_contact, min=1
        )
        return cost * _command_active(env, command_name, command_threshold)

    return _term


def soft_landing(
    tracker,
    foot_force_sensors: tuple[str, ...],
    command_name: str,
    command_threshold: float = 0.1,
):
    def _term(env) -> torch.Tensor:
        tracker._ensure_state(env)
        force_magnitude = torch.norm(tracker.force(env, foot_force_sensors), dim=-1)  # [B, F]
        landing_impact = force_magnitude * tracker.first_contact.float()
        cost = torch.sum(landing_impact, dim=1)
        num_landings = torch.sum(tracker.first_contact.float())
        env.extras["log"]["Metrics/landing_force_mean"] = torch.sum(landing_impact) / torch.clamp(
            num_landings, min=1
        )
        return cost * _command_active(env, command_name, command_threshold)

    return _term


def stand_still(command_name: str, command_threshold: float = 0.1):
    def _term(env) -> torch.Tensor:
        diff = env.orca.actuated_qpos() - env.orca.default_actuated_qpos()
        reward = torch.sum(torch.square(diff), dim=1)
        command = env.get_command(command_name)
        total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        return reward * (total <= command_threshold).float()

    return _term


def self_collision_cost(sensor_name: str):
    def _term(env) -> torch.Tensor:
        return env.orca.sensor(sensor_name).squeeze(-1)

    return _term


##
# Legacy simple terms (kept for the smoke task and older configs).
##


def track_lin_vel_x(env, command_name: str = "base_velocity", sigma: float = 0.25) -> torch.Tensor:
    error = env.orca.state.qvel[:, 0] - env.get_command(command_name)[:, 0]
    return torch.exp(-(error.square()) / sigma)


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright(env) -> torch.Tensor:
    quat_w = env.orca.state.qpos[:, 3].abs()
    return quat_w.clamp(0.0, 1.0)


def height_target(env, target: float = 0.74, sigma: float = 0.04) -> torch.Tensor:
    error = env.orca.state.qpos[:, 2] - target
    return torch.exp(-(error.square()) / sigma)
