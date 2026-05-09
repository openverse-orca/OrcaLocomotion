from __future__ import annotations

from orca_rl.tasks.velocity.config_types import LocomotionEnvCfg, UniformVelocityCommandCfg
from orca_rl.tasks.velocity.velocity_env_cfg import make_flat_velocity_env_cfg, make_rough_velocity_env_cfg


G1_MAX_DELTA = (
    0.45,
    0.35,
    0.35,
    0.55,
    0.30,
    0.20,
    0.45,
    0.35,
    0.35,
    0.55,
    0.30,
    0.20,
    0.20,
    0.20,
    0.20,
    0.35,
    0.30,
    0.30,
    0.40,
    0.25,
    0.20,
    0.20,
    0.35,
    0.30,
    0.30,
    0.40,
    0.25,
    0.20,
    0.20,
)


def _apply_g1_common_overrides(cfg: LocomotionEnvCfg, play: bool) -> LocomotionEnvCfg:
    cfg.scene_binding = {
        "resolver": "g1",
        "min_count": 1,
        "max_count": 1,
        "spawn_if_missing": True,
        "spawn_agent_name": "g1_000",
        "asset_path": "assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_old_usda",
    }
    cfg.observation_scales["height_scale"] = 2.0
    cfg.reset.update({"xy_noise": 0.03, "yaw_noise": 0.15, "joint_noise": 0.02})
    cfg.events["reset_base"].params.update({"xy_noise": 0.03, "yaw_noise": 0.15})
    cfg.events["reset_robot_joints"].params["joint_noise"] = 0.02
    cfg.events["randomize_body_mass"].params["base_mass_delta_range"] = (-1.0, 2.0)
    cfg.randomization["base_mass_delta_range"] = (-1.0, 2.0)

    cfg.rewards["base_height_l2"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["feet_slip"].params["asset_cfg"].site_names = ("left_foot", "right_foot")
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].params["asset_cfg"] = cfg.rewards["feet_slip"].params["asset_cfg"]

    if play:
        cfg.episode["length_s"] = 1.0e9
        cfg.observations["actor"].enable_corruption = False

    return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> LocomotionEnvCfg:
    """Create Unitree G1 flat velocity configuration for Orca RSL-RL."""

    cfg = make_flat_velocity_env_cfg(
        name="g1_flat_velocity",
        robot="g1",
        rsl_rl_config="orca_rl/tasks/velocity/config/g1/rl_cfg.py",
        seed=11,
        time_step=0.001,
        frame_skip=20,
        decimation=1,
        render_mode="human",
        action_safety_scale=0.75,
        action_max_delta=G1_MAX_DELTA,
        command_ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.8, 0.8),
        ),
        base_height=0.78,
        min_base_height=0.50,
        max_base_height=1.10,
        max_tilt_rad=0.75,
        reward_scales={
            "track_linear_velocity": 1.2,
            "track_angular_velocity": 0.8,
            "z_velocity_l2": -1.0,
            "orientation_l2": -3.0,
            "base_height_l2": -2.0,
            "torques_l2": -0.000005,
            "action_rate_l2": -0.015,
            "joint_pos_limits": -1.0,
            "feet_slip": -0.08,
            "termination": -3.0,
        },
    )

    return _apply_g1_common_overrides(cfg, play)


def unitree_g1_rough_env_cfg(play: bool = False) -> LocomotionEnvCfg:
    """Create Unitree G1 rough-terrain velocity configuration for Orca RSL-RL."""

    cfg = make_rough_velocity_env_cfg(
        name="g1_rough_velocity",
        robot="g1",
        rsl_rl_config="orca_rl/tasks/velocity/config/g1/rl_cfg.py:unitree_g1_rough_ppo_runner_cfg",
        seed=13,
        time_step=0.001,
        frame_skip=20,
        decimation=1,
        render_mode="human",
        action_safety_scale=0.72,
        action_max_delta=G1_MAX_DELTA,
        command_ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.7, 0.7),
        ),
        base_height=0.78,
        min_base_height=0.48,
        max_base_height=1.12,
        max_tilt_rad=0.80,
        reward_scales={
            "track_linear_velocity": 1.1,
            "track_angular_velocity": 0.7,
            "z_velocity_l2": -1.0,
            "orientation_l2": -3.5,
            "base_height_l2": -1.5,
            "torques_l2": -0.000006,
            "action_rate_l2": -0.02,
            "joint_pos_limits": -1.0,
            "feet_slip": -0.10,
            "termination": -3.0,
        },
    )
    cfg.train["num_learning_iterations"] = 2500
    return _apply_g1_common_overrides(cfg, play)


TASK_CONFIG_FACTORY = unitree_g1_flat_env_cfg
