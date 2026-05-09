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
from orca_rl.sensor import ContactMatchCfg, ContactSensorCfg, GridPatternCfg, RayCasterCfg
from orca_rl.terrains import TerrainCfg, TerrainGeneratorCfg


def make_flat_velocity_env_cfg(
    *,
    name: str,
    robot: str,
    rsl_rl_config: str,
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
        rsl_rl_config=rsl_rl_config,
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
        },
        train={"num_learning_iterations": 1500},
        play={"device": "cpu"},
        eval={"device": "cpu"},
        export={"enabled": True, "jit": True, "onnx": True},
    )


def make_rough_velocity_env_cfg(
    *,
    name: str,
    robot: str,
    rsl_rl_config: str,
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
    terrain_scan: GridPatternCfg | None = None,
) -> LocomotionEnvCfg:
    """Create a rough-terrain velocity config with mjlab-style terrain/sensor metadata."""

    scan_pattern = terrain_scan or GridPatternCfg(resolution=0.10, size=(1.6, 1.0))
    cfg = make_flat_velocity_env_cfg(
        name=name,
        robot=robot,
        rsl_rl_config=rsl_rl_config,
        seed=seed,
        time_step=time_step,
        frame_skip=frame_skip,
        decimation=decimation,
        render_mode=render_mode,
        action_safety_scale=action_safety_scale,
        action_max_delta=action_max_delta,
        command_ranges=command_ranges,
        base_height=base_height,
        min_base_height=min_base_height,
        max_base_height=max_base_height,
        max_tilt_rad=max_tilt_rad,
        reward_scales=reward_scales,
    )
    cfg.terrain = TerrainCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(curriculum=True),
        static_friction=0.9,
        dynamic_friction=0.8,
    )
    cfg.sensors["terrain_scan"] = RayCasterCfg(
        name="terrain_scan",
        frame_name="base",
        pattern=scan_pattern,
        attach_yaw_only=True,
    )
    cfg.sensors["nonfoot_ground_contact"] = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatchCfg(
            mode="geom",
            pattern=r".*_collision\d*$",
            entity="robot",
            exclude=(r".*foot.*",),
        ),
        secondary=ContactMatchCfg(mode="body", pattern="terrain", entity="terrain"),
        fields=("found", "force"),
        reduce="none",
        history_length=4,
    )
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
        func=mdp.height_scan,
        noise=(-0.05, 0.05),
        scale=1.0,
        params={"sensor_name": "terrain_scan"},
    )
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=mdp.height_scan,
        scale=1.0,
        params={"sensor_name": "terrain_scan"},
    )
    cfg.observation_scales.update(
        {
            "height_scan_dim": scan_pattern.num_rays,
            "height_scan_scale": 1.0,
        }
    )
    cfg.events["randomize_terrain"] = EventTermCfg(
        func=mdp.randomize_terrain,
        mode="reset",
        params={"terrain_name": "rough"},
    )
    cfg.curriculum["terrain_levels"] = TermCfg(
        func=mdp.terrain_levels,
        params={"success_lin_vel_threshold": 0.7, "failure_lin_vel_threshold": 0.2},
    )
    cfg.rewards["feet_air_time"] = TermCfg(
        func=mdp.feet_air_time,
        weight=0.25,
        params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["foot_clearance"] = TermCfg(
        func=mdp.foot_clearance,
        weight=-0.05,
        params={"target_height": 0.08, "sensor_name": "terrain_scan"},
    )
    cfg.rewards["body_ang_vel_l2"] = TermCfg(func=mdp.body_ang_vel_l2, weight=-0.05)
    cfg.terminations["illegal_contact"] = TermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 10.0},
    )
    cfg.randomization.update(
        {
            "terrain": "rough",
            "terrain_curriculum": True,
            "height_scan_dim": scan_pattern.num_rays,
        }
    )
    return cfg
