"""Event terms ported from Orca's velocity task (reset / interval / startup)."""

from __future__ import annotations

import torch

from .math_utils import quat_from_yaw, quat_mul

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_VEL_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def reset_root_state_uniform(
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
):
    """Orca reset_root_state_uniform for a free-joint root at qpos[0:7].

    Positions are offset from the default (keyframe) root pose; only yaw
    rotation is applied (roll/pitch ranges of the G1 flat task are zero).
    """
    velocity_range = velocity_range or {}

    def _event(env, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        device = env.device
        pose_ranges = torch.tensor(
            [pose_range.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=device
        )
        pose_samples = (
            torch.rand(n, 6, device=device) * (pose_ranges[:, 1] - pose_ranges[:, 0])
            + pose_ranges[:, 0]
        )
        root_default = env.orca.default_root_qpos(env_ids)
        qpos = env.orca.state.qpos
        qpos[env_ids, 0:3] = root_default[:, 0:3] + pose_samples[:, 0:3]
        yaw_quat = quat_from_yaw(pose_samples[:, 5])
        qpos[env_ids, 3:7] = quat_mul(yaw_quat, root_default[:, 3:7])
        vel_ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in _VEL_KEYS], device=device
        )
        vel_samples = (
            torch.rand(n, 6, device=device) * (vel_ranges[:, 1] - vel_ranges[:, 0])
            + vel_ranges[:, 0]
        )
        env.orca.state.qvel[env_ids, 0:6] = vel_samples

    return _event


def reset_joints_by_offset(
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    soft_limit_factor: float = 0.9,
):
    """Offset actuated joints around the default pose, clamped to soft limits."""

    def _event(env, env_ids: torch.Tensor) -> None:
        orca = env.orca
        limits = orca.actuated_joint_range()
        mid = 0.5 * (limits[:, 0] + limits[:, 1])
        half = 0.5 * (limits[:, 1] - limits[:, 0]) * soft_limit_factor
        default = orca.default_actuated_qpos()[env_ids]
        n = env_ids.numel()
        joint_pos = default + torch.empty_like(default).uniform_(*position_range)
        joint_pos = joint_pos.clamp(mid - half, mid + half)
        joint_vel = torch.empty(n, limits.shape[0], device=env.device).uniform_(*velocity_range)
        orca.write_actuated_joint_state(joint_pos, joint_vel, env_ids)

    return _event


def push_by_setting_velocity(velocity_range: dict[str, tuple[float, float]]):
    """Interval event: add a random twist to the root velocity (Orca push_robot)."""

    def _event(env, mask: torch.Tensor) -> None:
        device = env.device
        ranges = torch.tensor(
            [velocity_range.get(key, (0.0, 0.0)) for key in _VEL_KEYS], device=device
        )
        samples = (
            torch.rand(env.num_envs, 6, device=device) * (ranges[:, 1] - ranges[:, 0])
            + ranges[:, 0]
        )
        qvel = env.orca.state.qvel
        qvel[:, 0:6] = torch.where(mask[:, None], qvel[:, 0:6] + samples, qvel[:, 0:6])

    return _event


def encoder_bias(bias_range: tuple[float, float]):
    """Startup event: per-env constant joint-position observation bias (Orca dr)."""

    def _event(env, _arg=None) -> None:
        nu = env.orca.state.ctrl.shape[-1]
        env.encoder_bias = torch.empty(env.num_envs, nu, device=env.device).uniform_(*bias_range)

    return _event
