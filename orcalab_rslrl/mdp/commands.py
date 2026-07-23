from __future__ import annotations

from dataclasses import dataclass

import torch

from .math_utils import wrap_to_pi, yaw_from_quat


def fixed_velocity_command(lin_vel_x: float, lin_vel_y: float = 0.0, yaw_rate: float = 0.0):
    def _reset(env, env_ids: torch.Tensor) -> torch.Tensor:
        command = torch.zeros(env_ids.numel(), 3, device=env.device)
        command[:, 0] = lin_vel_x
        command[:, 1] = lin_vel_y
        command[:, 2] = yaw_rate
        return command

    return _reset


@dataclass
class VelocityRanges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None


class UniformVelocityCommand:
    """Port of Orca's UniformVelocityCommand (heading + standing envs).

    ``sample`` is the CommandTermCfg.func; ``update`` runs every control step
    and applies heading control / zeroes standing envs. All per-env state
    lives on GPU; the ranges are mutable so a curriculum can widen them.
    """

    def __init__(
        self,
        ranges: VelocityRanges,
        *,
        rel_standing_envs: float = 0.0,
        rel_heading_envs: float = 1.0,
        heading_command: bool = False,
        heading_control_stiffness: float = 1.0,
    ):
        if heading_command and ranges.heading is None:
            raise ValueError("heading_command=True requires ranges.heading")
        self.ranges = ranges
        self.rel_standing_envs = rel_standing_envs
        self.rel_heading_envs = rel_heading_envs
        self.heading_command = heading_command
        self.heading_control_stiffness = heading_control_stiffness
        self.heading_target: torch.Tensor | None = None
        self.is_heading_env: torch.Tensor | None = None
        self.is_standing_env: torch.Tensor | None = None

    def _ensure_state(self, env) -> None:
        if self.heading_target is None:
            self.heading_target = torch.zeros(env.num_envs, device=env.device)
            self.is_heading_env = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def sample(self, env, env_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_state(env)
        n = env_ids.numel()
        command = torch.zeros(n, 3, device=env.device)
        command[:, 0].uniform_(*self.ranges.lin_vel_x)
        command[:, 1].uniform_(*self.ranges.lin_vel_y)
        command[:, 2].uniform_(*self.ranges.ang_vel_z)
        # Orca zeroes near-zero commands so tiny twists don't fight stand-still.
        command *= (torch.norm(command, dim=1) > 0.1).unsqueeze(1).float()
        if self.heading_command:
            self.heading_target[env_ids] = torch.empty(n, device=env.device).uniform_(*self.ranges.heading)
            self.is_heading_env[env_ids] = (
                torch.rand(n, device=env.device) <= self.rel_heading_envs
            )
        self.is_standing_env[env_ids] = torch.rand(n, device=env.device) <= self.rel_standing_envs
        return command

    def sample_masked(self, env, mask: torch.Tensor) -> torch.Tensor:
        """Sample replacements for a GPU mask without materializing indices.

        ``torch.nonzero`` on a CUDA tensor synchronizes the host because the
        result has a dynamic length. Command expiration is checked every
        control step, so using indexed sampling there serializes the rollout.
        Sampling a full batch and retaining only masked values keeps all state
        transitions on the device. Random values generated for unmasked
        environments are deliberately discarded.
        """

        self._ensure_state(env)
        if mask.shape != (env.num_envs,):
            raise ValueError(f"mask shape {tuple(mask.shape)} != {(env.num_envs,)}")
        mask = mask.to(device=env.device, dtype=torch.bool)
        n = env.num_envs
        command = torch.empty(n, 3, device=env.device)
        command[:, 0].uniform_(*self.ranges.lin_vel_x)
        command[:, 1].uniform_(*self.ranges.lin_vel_y)
        command[:, 2].uniform_(*self.ranges.ang_vel_z)
        command *= (torch.norm(command, dim=1) > 0.1).unsqueeze(1).float()
        if self.heading_command:
            heading_target = torch.empty(n, device=env.device).uniform_(*self.ranges.heading)
            is_heading = torch.rand(n, device=env.device) <= self.rel_heading_envs
            self.heading_target.copy_(torch.where(mask, heading_target, self.heading_target))
            self.is_heading_env.copy_(torch.where(mask, is_heading, self.is_heading_env))
        is_standing = torch.rand(n, device=env.device) <= self.rel_standing_envs
        self.is_standing_env.copy_(torch.where(mask, is_standing, self.is_standing_env))
        return command

    def update(self, env, name: str) -> None:
        self._ensure_state(env)
        command = env.commands[name]
        if self.heading_command:
            heading_w = yaw_from_quat(env.orca.state.qpos[:, 3:7])
            heading_error = wrap_to_pi(self.heading_target - heading_w)
            heading_cmd = torch.clip(
                self.heading_control_stiffness * heading_error,
                min=self.ranges.ang_vel_z[0],
                max=self.ranges.ang_vel_z[1],
            )
            command[:, 2] = torch.where(self.is_heading_env, heading_cmd, command[:, 2])
        command.masked_fill_(self.is_standing_env[:, None], 0.0)
