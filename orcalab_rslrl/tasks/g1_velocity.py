"""Unitree G1 flat-ground velocity task for Orca.

Every MDP term is written out explicitly below: twist command with heading
control + standing envs + curriculum, noisy actor observations, privileged
critic observations, the full reward stack (tracking, posture, gait, feet,
regularizers), terminations, and the reset/interval/startup events.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from ..managers import (
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from ..mdp import commands, events, observations, rewards, terminations
from ..mdp.commands import UniformVelocityCommand, VelocityRanges
from ..mdp.trackers import FeetContactTracker
from .._internal.runtime import OrcaPhysicsRuntime
from ..robots.g1_flat import TERRAIN_FLAT, G1FlatRobot, build_g1_flat_model, g1_action_scale
from ..rslrl import RslRlVecEnvAdapter
from ..scene_options import ORCA_TRAIN_SCENE_OPTIONS, UNITREE_ORCA_CONTROL_DT
from .registry import register_task

TASK_NAME = "G1-Velocity-Flat"

# Orca curriculum `commands_vel`: widen the sampling ranges after N control steps.
_COMMAND_STAGES = (
    {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
)

# Per-joint posture widths for the G1 velocity task.
_STD_STANDING = {r".*": 0.05}
_STD_WALKING = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.15, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15, r".*waist_roll.*": 0.1, r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15, r".*shoulder_roll.*": 0.1, r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1, r".*wrist.*": 0.1,
}
_STD_RUNNING = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.25, r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25, r".*waist_roll.*": 0.1, r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.25, r".*shoulder_roll.*": 0.1, r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1, r".*wrist.*": 0.1,
}


def _command_curriculum(twist: UniformVelocityCommand):
    def _event(env, _arg=None) -> None:
        for stage in _COMMAND_STAGES:
            if env.common_step_counter > stage["step"]:
                twist.ranges.lin_vel_x = stage["lin_vel_x"]
                twist.ranges.lin_vel_y = stage["lin_vel_y"]
                if "ang_vel_z" in stage:
                    twist.ranges.ang_vel_z = stage["ang_vel_z"]

    return _event


def _root_reset_pose_range(*, play: bool) -> dict[str, tuple[float, float]]:
    if play:
        return {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
    return {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)}


def _decimation_for_timestep(timestep: float) -> int:
    if timestep <= 0.0:
        raise ValueError(f"physics timestep must be positive, got {timestep}")
    decimation = round(UNITREE_ORCA_CONTROL_DT / timestep)
    if decimation < 1 or not math.isclose(
        timestep * decimation, UNITREE_ORCA_CONTROL_DT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"physics timestep {timestep:g} cannot preserve the "
            f"{UNITREE_ORCA_CONTROL_DT:g}s control period with integer decimation"
        )
    return decimation


def build_g1_flat_env_cfg(
    robot: G1FlatRobot,
    *,
    play: bool = False,
    physics_timestep: float = ORCA_TRAIN_SCENE_OPTIONS.timestep,
) -> ManagerBasedRLEnvCfg:
    sensors = robot.sensors
    tracker = FeetContactTracker(sensors.foot_found)

    twist = UniformVelocityCommand(
        VelocityRanges(
            lin_vel_x=(-0.5, 1.0) if play else _COMMAND_STAGES[0]["lin_vel_x"],
            lin_vel_y=(-0.5, 0.5) if play else _COMMAND_STAGES[0]["lin_vel_y"],
            ang_vel_z=(-0.5, 0.5) if play else _COMMAND_STAGES[0]["ang_vel_z"],
            heading=(-math.pi, math.pi),
        ),
        rel_standing_envs=0.05,
        heading_command=True,
        heading_control_stiffness=0.5,
    )

    ##
    # Observations (actor gets uniform noise, critic is privileged and clean).
    ##

    actor_terms: dict[str, ObservationTermCfg] = {
        "base_ang_vel": ObservationTermCfg(
            observations.builtin_sensor(sensors.base_ang_vel), noise=(-0.2, 0.2)
        ),
        "projected_gravity": ObservationTermCfg(observations.projected_gravity, noise=(-0.05, 0.05)),
        "command": ObservationTermCfg(observations.generated_commands("twist")),
        "phase": ObservationTermCfg(observations.phase(period=0.6, command_name="twist")),
        "joint_pos": ObservationTermCfg(observations.joint_pos_rel, noise=(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(observations.joint_vel_rel, noise=(-1.5, 1.5)),
        "actions": ObservationTermCfg(observations.last_action),
    }
    critic_terms: dict[str, ObservationTermCfg] = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
            observations.builtin_sensor(sensors.base_lin_vel), noise=(-0.5, 0.5)
        ),
        "foot_height": ObservationTermCfg(observations.foot_height(sensors.foot_pos)),
        "foot_air_time": ObservationTermCfg(observations.foot_air_time(tracker)),
        "foot_contact": ObservationTermCfg(observations.foot_contact(tracker)),
        "foot_contact_forces": ObservationTermCfg(observations.foot_contact_forces(sensors.foot_force)),
    }
    observation_groups = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    ##
    # Events.
    ##

    event_terms: dict[str, EventTermCfg] = {
        "reset_base": EventTermCfg(
            events.reset_root_state_uniform(
                pose_range=_root_reset_pose_range(play=play),
                velocity_range={},
            ),
            mode="reset",
        ),
        "reset_robot_joints": EventTermCfg(
            events.reset_joints_by_offset(
                position_range=(-0.0, 0.0),
                velocity_range=(-0.0, 0.0),
                soft_limit_factor=robot.soft_joint_pos_limit_factor,
            ),
            mode="reset",
        ),
        "reset_feet_tracker": EventTermCfg(tracker.reset, mode="reset"),
        "feet_tracker": EventTermCfg(tracker, mode="post_step"),
    }
    if not play:
        event_terms["encoder_bias"] = EventTermCfg(events.encoder_bias((-0.015, 0.015)), mode="startup")
        event_terms["push_robot"] = EventTermCfg(
            events.push_by_setting_velocity(
                {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
                }
            ),
            mode="interval",
            interval_range_s=(5.0, 6.0),
        )
        event_terms["command_curriculum"] = EventTermCfg(_command_curriculum(twist), mode="post_step")

    ##
    # Rewards (weights straight from velocity_env_cfg + G1 overrides).
    ##

    reward_terms = {
        "track_linear_velocity": RewardTermCfg(
            rewards.track_linear_velocity("twist", std=math.sqrt(0.25), lin_vel_sensor=sensors.base_lin_vel),
            weight=1.0,
        ),
        "track_angular_velocity": RewardTermCfg(
            rewards.track_angular_velocity("twist", std=math.sqrt(0.5), ang_vel_sensor=sensors.base_ang_vel),
            weight=1.0,
        ),
        "body_orientation_l2": RewardTermCfg(rewards.body_orientation_l2(robot.torso_body), weight=-1.0),
        "pose": RewardTermCfg(
            rewards.variable_posture(
                "twist", robot.joint_names, _STD_STANDING, _STD_WALKING, _STD_RUNNING,
                walking_threshold=0.1, running_threshold=1.5,
            ),
            weight=1.0,
        ),
        "body_ang_vel": RewardTermCfg(rewards.body_angular_velocity_penalty(robot.torso_body), weight=-0.05),
        "angular_momentum": RewardTermCfg(rewards.angular_momentum_penalty(sensors.root_angmom), weight=-0.025),
        "is_terminated": RewardTermCfg(rewards.is_terminated, weight=-200.0),
        "joint_acc_l2": RewardTermCfg(rewards.joint_acc_l2, weight=-2.5e-7),
        "joint_pos_limits": RewardTermCfg(
            rewards.joint_pos_limits(robot.soft_joint_pos_limit_factor), weight=-10.0
        ),
        "action_rate_l2": RewardTermCfg(rewards.action_rate_l2, weight=-0.05),
        "foot_gait": RewardTermCfg(
            rewards.feet_gait(
                tracker, period=0.6, offset=(0.0, 0.5), threshold=0.56,
                command_name="twist", command_threshold=0.1,
            ),
            weight=0.5,
        ),
        "foot_clearance": RewardTermCfg(
            rewards.feet_clearance(
                sensors.foot_pos, sensors.foot_vel, target_height=0.10,
                command_name="twist", command_threshold=0.1,
            ),
            weight=-1.0,
        ),
        "foot_slip": RewardTermCfg(
            rewards.feet_slip(tracker, sensors.foot_vel, command_name="twist", command_threshold=0.1),
            weight=-0.25,
        ),
        "soft_landing": RewardTermCfg(
            rewards.soft_landing(tracker, sensors.foot_force, command_name="twist", command_threshold=0.1),
            weight=-1e-3,
        ),
        "stand_still": RewardTermCfg(rewards.stand_still("twist", command_threshold=0.1), weight=-1.0),
        "self_collisions": RewardTermCfg(rewards.self_collision_cost(sensors.self_collision), weight=-1.0),
    }

    ##
    # Terminations.
    ##

    termination_terms = {
        "time_out": TerminationTermCfg(terminations.time_out, time_out=True),
        "fell_over": TerminationTermCfg(terminations.bad_orientation(math.radians(70.0))),
    }

    # Preserve Orca's 20ms control period and 20s episode.
    return ManagerBasedRLEnvCfg(
        decimation=_decimation_for_timestep(physics_timestep),
        episode_length_steps=int(1e9) if play else 1000,
        action_scale=g1_action_scale(),
        clip_actions=None,
        action_offset=True,
        reward_scale_by_dt=True,
        observations=observation_groups,
        commands={
            "twist": CommandTermCfg(
                twist.sample, dim=3, resampling_time_range=(3.0, 8.0), update=twist.update
            )
        },
        events=event_terms,
        rewards=reward_terms,
        terminations=termination_terms,
    )


def make_env(
    *,
    num_envs: int,
    device: str,
    headless: bool = True,
    mjcf_path: str | Path | None = None,
    play: bool = False,
    physics_timestep: float | None = None,
    terrain_kind: str = TERRAIN_FLAT,
    seed: int | None = None,
):
    """Create the G1 flat-velocity environment on the Orca runtime."""
    if not headless and not play:
        raise ValueError("G1 velocity training is headless; use play for visualization")
    robot = G1FlatRobot()
    timestep = (
        ORCA_TRAIN_SCENE_OPTIONS.timestep
        if physics_timestep is None
        else physics_timestep
    )
    model = build_g1_flat_model(
        mjcf_path,
        foot_friction_range=None if play else (0.3, 1.6),
        base_com_offset_range=None if play else (-0.05, 0.05),
        timestep=timestep,
        terrain_kind=terrain_kind,
        seed=seed,
    )
    orca = OrcaPhysicsRuntime(model, num_envs=num_envs, device=device, nconmax=None, njmax=300)
    cfg = build_g1_flat_env_cfg(robot, play=play, physics_timestep=float(model.opt.timestep))
    return ManagerBasedRLEnv(orca, cfg)


def make_train_env(
    *,
    num_envs: int,
    device: str,
    headless: bool = True,
    mjcf_path: str | Path | None = None,
    play: bool = False,
    physics_timestep: float | None = None,
    terrain_kind: str = TERRAIN_FLAT,
    seed: int | None = None,
):
    return RslRlVecEnvAdapter(
        make_env(
            num_envs=num_envs,
            device=device,
            headless=headless,
            mjcf_path=mjcf_path,
            play=play,
            physics_timestep=physics_timestep,
            terrain_kind=terrain_kind,
            seed=seed,
        )
    )


def register() -> None:
    register_task(TASK_NAME, make_train_env, "Unitree G1 flat-ground velocity walking")


register()
