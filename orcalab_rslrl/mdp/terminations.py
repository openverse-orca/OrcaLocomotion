from __future__ import annotations

import torch

from .math_utils import quat_apply_inverse


def bad_orientation(limit_angle: float):
    """Terminate when the base tilts beyond ``limit_angle`` radians (Orca)."""

    def _term(env) -> torch.Tensor:
        gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        projected = quat_apply_inverse(env.orca.state.qpos[:, 3:7], gravity)
        return torch.acos(-projected[:, 2].clamp(-1.0, 1.0)).abs() > limit_angle

    return _term


def time_out(env) -> torch.Tensor:
    """Episode length termination; pair with TerminationTermCfg(time_out=True)."""
    return env.episode_length_buf >= env.cfg.episode_length_steps


def root_height_below(min_height: float):
    def _term(env) -> torch.Tensor:
        return env.orca.state.qpos[:, 2] < min_height

    return _term


def non_finite_state(env) -> torch.Tensor:
    qpos_ok = torch.isfinite(env.orca.state.qpos).all(dim=-1)
    qvel_ok = torch.isfinite(env.orca.state.qvel).all(dim=-1)
    return ~(qpos_ok & qvel_ok)
