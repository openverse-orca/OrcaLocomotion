from __future__ import annotations

from orca_rl.tasks.velocity import mdp
from orca_rl.tasks.velocity.config_types import (
    EventTermCfg,
    JointPositionActionCfg,
    LocomotionEnvCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TermCfg,
    UniformVelocityCommandCfg,
)
from orca_rl.sensor import ContactMatchCfg, ContactSensorCfg
from orca_rl.terrains import TerrainCfg


def make_flat_velocity_env_cfg(
    *,
    name: str,
    robot: str,
    seed: int,
    time_step: float,
    frame_skip: int,
    decimation: int,
    render_mode: str,
    action_safety_scale: float,
    action_max_delta: tuple[float, ...],
    command_ranges: UniformVelocityCommandCfg.Ranges,
    base_height: float,
    min_base_height: float,
    max_base_height: float,
    max_tilt_rad: float,
    reward_scales: dict[str, float],
) -> LocomotionEnvCfg:
    """Create the shared flat velocity task config before robot-specific overrides."""

    observation_terms = {
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=(-0.2, 0.2), scale=0.25),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=(-0.05, 0.05)),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=(-1.5, 1.5), scale=0.05),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0),
        **observation_terms,
    }

    return LocomotionEnvCfg(
        name=name,
        robot=robot,
        num_envs=1,
        seed=seed,
        sim={
            "time_step": time_step,
            "frame_skip": frame_skip,
            "decimation": decimation,
            "render_mode": render_mode,
        },
        terrain=TerrainCfg(terrain_type="plane"),
        sensors={
            "feet_ground_contact": ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatchCfg(mode="body", pattern="feet", entity="robot"),
                secondary=ContactMatchCfg(mode="body", pattern="terrain", entity="terrain"),
                fields=("found",),
                reduce="none",
                track_air_time=True,
            ),
        },
        episode={"length_s": 20.0},
        actions={
            "joint_pos": JointPositionActionCfg(
                func=mdp.joint_position,
                clip=1.0,
                safety_scale=action_safety_scale,
                max_delta=action_max_delta,
            )
        },
        commands={
            "twist": UniformVelocityCommandCfg(
                func=mdp.uniform_velocity_command,
                entity_name="robot",
                resampling_time_s=4.0,
                ranges=command_ranges,
            )
        },
        observations={
            "actor": ObservationGroupCfg(
                terms=observation_terms,
                concatenate_terms=True,
                enable_corruption=True,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
            ),
        },
        observation_scales={
            "add_noise": True,
            "noise_level": 1.0,
            "ang_vel_scale": 0.25,
            "lin_vel_scale": 2.0,
            "dof_pos_scale": 1.0,
            "dof_vel_scale": 0.05,
            "command_scale": (2.0, 2.0, 0.25),
            "height_scale": 5.0,
        },
        rewards={
            "track_linear_velocity": TermCfg(
                func=mdp.track_linear_velocity,
                weight=reward_scales["track_linear_velocity"],
                params={"command_name": "twist", "sigma": 0.25},
            ),
            "track_angular_velocity": TermCfg(
                func=mdp.track_angular_velocity,
                weight=reward_scales["track_angular_velocity"],
                params={"command_name": "twist", "sigma": 0.25},
            ),
            "z_velocity_l2": TermCfg(func=mdp.z_velocity_l2, weight=reward_scales["z_velocity_l2"]),
            "orientation_l2": TermCfg(func=mdp.orientation_l2, weight=reward_scales["orientation_l2"]),
            "base_height_l2": TermCfg(
                func=mdp.base_height_l2,
                weight=reward_scales["base_height_l2"],
                params={
                    "target_height": base_height,
                    "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                },
            ),
            "torques_l2": TermCfg(func=mdp.torques_l2, weight=reward_scales["torques_l2"]),
            "action_rate_l2": TermCfg(func=mdp.action_rate_l2, weight=reward_scales["action_rate_l2"]),
            "joint_pos_limits": TermCfg(func=mdp.joint_pos_limits, weight=reward_scales["joint_pos_limits"]),
            "feet_slip": TermCfg(
                func=mdp.feet_slip,
                weight=reward_scales["feet_slip"],
                params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
            ),
            "termination": TermCfg(func=mdp.termination, weight=reward_scales["termination"]),
        },
        terminations={
            "time_out": TermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TermCfg(func=mdp.bad_orientation, params={"max_tilt_rad": max_tilt_rad}),
            "base_height": TermCfg(
                func=mdp.base_height,
                params={"min_base_height": min_base_height, "max_base_height": max_base_height},
            ),
            "base_contact": TermCfg(func=mdp.base_contact, params={"enabled": True}),
        },
        events={
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={"xy_noise": 0.05, "yaw_noise": 0.2, "base_height": base_height},
            ),
            "reset_robot_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={"joint_noise": 0.03, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "randomize_friction": EventTermCfg(
                func=mdp.randomize_friction,
                mode="startup",
                params={"friction_range": (0.7, 1.3)},
            ),
            "randomize_body_mass": EventTermCfg(
                func=mdp.randomize_body_mass,
                mode="startup",
                params={"base_mass_delta_range": (-0.5, 1.5)},
            ),
            "randomize_actuator_properties": EventTermCfg(
                func=mdp.randomize_actuator_properties,
                mode="startup",
                params={
                    "kp_scale_range": (0.9, 1.1),
                    "kd_scale_range": (0.9, 1.1),
                    "torque_scale_range": (0.9, 1.1),
                },
            ),
            "randomize_action_latency": EventTermCfg(
                func=mdp.randomize_action_latency,
                mode="startup",
                params={"max_action_delay_steps": 2},
            ),
            "randomize_solver_params": EventTermCfg(
                func=mdp.randomize_solver_params,
                mode="startup",
                params={"solver_iterations_range": (40, 80), "solver_tolerance_scale_range": (0.5, 2.0)},
            ),
            "randomize_contact_params": EventTermCfg(
                func=mdp.randomize_contact_params,
                mode="startup",
                params={
                    "contact_solref_timeconst_scale_range": (0.8, 1.2),
                    "contact_solref_dampratio_scale_range": (0.8, 1.2),
                    "contact_solimp_scale_range": (0.9, 1.1),
                    "contact_margin_scale_range": (0.8, 1.2),
                },
            ),
            "push_robot": EventTermCfg(
                func=mdp.push_robot,
                mode="interval",
                params={
                    "push_interval_s": 8.0,
                    "push_velocity_range": (-0.4, 0.4),
                    "push_yaw_velocity_range": (-0.3, 0.3),
                },
            ),
        },
        curriculum={},
        metrics={
            "mean_action_acc": TermCfg(func=mdp.mean_action_acc),
        },
        reset={
            "base_height": base_height,
            "xy_noise": 0.05,
            "yaw_noise": 0.2,
            "joint_noise": 0.03,
        },
        contacts={"touch_threshold": 1.0},
        randomization={
            "enabled": True,
            "friction_range": (0.7, 1.3),
            "base_mass_delta_range": (-0.5, 1.5),
            "base_inertia_scale_range": (0.9, 1.1),
            "base_com_offset_range": (-0.02, 0.02),
            "kp_scale_range": (0.9, 1.1),
            "kd_scale_range": (0.9, 1.1),
            "torque_scale_range": (0.9, 1.1),
            "max_action_delay_steps": 2,
            "push_interval_s": 8.0,
            "push_velocity_range": (-0.4, 0.4),
            "push_yaw_velocity_range": (-0.3, 0.3),
            "solver_iterations_range": (40, 80),
            "solver_tolerance_scale_range": (0.5, 2.0),
            "contact_solref_timeconst_scale_range": (0.8, 1.2),
            "contact_solref_dampratio_scale_range": (0.8, 1.2),
            "contact_solimp_scale_range": (0.9, 1.1),
            "contact_margin_scale_range": (0.8, 1.2),
        },
    )
