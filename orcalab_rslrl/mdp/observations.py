from __future__ import annotations

import torch

from .math_utils import quat_apply_inverse

##
# Base state.
##


def base_height(env) -> torch.Tensor:
    return env.orca.state.qpos[:, 2:3]


def base_quat(env) -> torch.Tensor:
    return env.orca.state.qpos[:, 3:7]


def base_lin_vel(env) -> torch.Tensor:
    return env.orca.state.qvel[:, 0:3]


def base_ang_vel(env) -> torch.Tensor:
    return env.orca.state.qvel[:, 3:6]


def builtin_sensor(sensor_name: str):
    """Named MuJoCo sensor reading (gyro, velocimeter, ...)."""

    def _term(env) -> torch.Tensor:
        return env.orca.sensor(sensor_name)

    return _term


def projected_gravity(env) -> torch.Tensor:
    """World gravity direction expressed in the base frame."""
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    return quat_apply_inverse(env.orca.state.qpos[:, 3:7], gravity)


##
# Joint state.
##


def joint_pos_rel(env) -> torch.Tensor:
    value = env.orca.actuated_qpos() - env.orca.default_actuated_qpos()
    bias = getattr(env, "encoder_bias", None)
    return value if bias is None else value + bias


def joint_vel_rel(env) -> torch.Tensor:
    return env.orca.actuated_qvel()


# Backwards-compatible aliases over all named joints.
def joint_vel(env) -> torch.Tensor:
    return env.orca.joint_qvel()


##
# Actions and commands.
##


def last_action(env) -> torch.Tensor:
    return env.last_action


def generated_commands(command_name: str):
    def _term(env) -> torch.Tensor:
        return env.get_command(command_name)

    return _term


def velocity_command(env) -> torch.Tensor:
    return env.get_command("base_velocity")


def phase(period: float, command_name: str, command_threshold: float = 0.1):
    """Gait phase (sin, cos); zero for standing environments."""

    def _term(env) -> torch.Tensor:
        global_phase = (env.episode_length_buf.float() * env.step_dt) % period / period
        value = torch.stack(
            (
                torch.sin(global_phase * torch.pi * 2.0),
                torch.cos(global_phase * torch.pi * 2.0),
            ),
            dim=-1,
        )
        stand = torch.linalg.norm(env.get_command(command_name), dim=1) < command_threshold
        return torch.where(stand[:, None], torch.zeros_like(value), value)

    return _term


##
# Feet (critic-only terms).
##


def foot_height(foot_pos_sensors: tuple[str, ...]):
    def _term(env) -> torch.Tensor:
        return torch.cat(
            [env.orca.sensor(name)[:, 2:3] for name in foot_pos_sensors], dim=-1
        )

    return _term


def foot_air_time(tracker):
    def _term(env) -> torch.Tensor:
        tracker._ensure_state(env)
        return tracker.air_time

    return _term


def foot_contact(tracker):
    def _term(env) -> torch.Tensor:
        return (tracker.found(env) > 0).float()

    return _term


def foot_contact_forces(force_sensors: tuple[str, ...]):
    def _term(env) -> torch.Tensor:
        forces = torch.cat([env.orca.sensor(name) for name in force_sensors], dim=-1)
        return torch.sign(forces) * torch.log1p(torch.abs(forces))

    return _term
