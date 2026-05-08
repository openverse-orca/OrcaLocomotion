from __future__ import annotations

from orca_rl.tasks.velocity.config_types import LocomotionEnvCfg, UniformVelocityCommandCfg
from orca_rl.tasks.velocity.velocity_env_cfg import make_flat_velocity_env_cfg


GO2_MAX_DELTA = (
    0.35,
    0.8,
    0.8,
    0.35,
    0.8,
    0.8,
    0.35,
    0.8,
    0.8,
    0.35,
    0.8,
    0.8,
)


def unitree_go2_flat_env_cfg(play: bool = False) -> LocomotionEnvCfg:
    """Create Unitree GO2 flat velocity configuration for Orca RSL-RL."""

    cfg = make_flat_velocity_env_cfg(
        name="go2_flat_velocity",
        robot="go2",
        rsl_rl_config="orca_rl/tasks/velocity/config/go2/rl_cfg.py",
        seed=7,
        time_step=0.005,
        frame_skip=1,
        decimation=4,
        render_mode="none",
        action_safety_scale=0.85,
        action_max_delta=GO2_MAX_DELTA,
        command_ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.2),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.8, 0.8),
        ),
        base_height=0.34,
        min_base_height=0.18,
        max_base_height=0.75,
        max_tilt_rad=0.95,
        reward_scales={
            "track_linear_velocity": 1.5,
            "track_angular_velocity": 0.75,
            "z_velocity_l2": -1.0,
            "orientation_l2": -2.0,
            "base_height_l2": -1.0,
            "torques_l2": -0.00002,
            "action_rate_l2": -0.02,
            "joint_pos_limits": -1.0,
            "feet_slip": -0.1,
            "termination": -2.0,
        },
    )

    cfg.scene_binding = {
        "resolver": "orca_rl.tasks.velocity.config.go2.env_cfgs.resolve_scene_binding",
        "min_count": 1,
        "max_count": 1,
    }
    cfg.rewards["base_height_l2"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["feet_slip"].params["asset_cfg"].site_names = ("FR", "FL", "RR", "RL")

    if play:
        cfg.episode["length_s"] = 1.0e9
        cfg.observations["actor"].enable_corruption = False

    return cfg


def resolve_scene_binding(**kwargs):
    """Resolve the GO2 scene instance for this task config."""

    from orca_rl.rsl_env.scene_binding import resolve_go2_scene_binding

    return resolve_go2_scene_binding(**kwargs)


TASK_CONFIG_FACTORY = unitree_go2_flat_env_cfg
