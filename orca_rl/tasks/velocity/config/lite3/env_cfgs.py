from __future__ import annotations

from dataclasses import replace

from orca_rl.rsl_env.scene_binding import LITE3_AGENT_ASSET_PATH
from orca_rl.tasks.velocity.config_types import LocomotionEnvCfg, UniformVelocityCommandCfg
from orca_rl.tasks.velocity.velocity_env_cfg import make_flat_velocity_env_cfg


def deeprobotics_lite3_flat_env_cfg(play: bool = False) -> LocomotionEnvCfg:
    cfg = make_flat_velocity_env_cfg(
        name="lite3_flat_velocity",
        robot="lite3",
        rsl_rl_config="orca_rl/tasks/velocity/config/lite3/rl_cfg.py",
        seed=7,
        time_step=0.005,
        frame_skip=1,
        decimation=4,
        render_mode="none",
        action_safety_scale=1.0,
        action_max_delta=(0.125, 0.25, 0.25) * 4,
        command_ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-0.8, 0.8)
        ),
        base_height=0.28,
        min_base_height=0.12,
        max_base_height=0.65,
        max_tilt_rad=1.22173,
        reward_scales={
            "track_linear_velocity": 1.5, "track_angular_velocity": 0.75,
            "z_velocity_l2": -1.0, "orientation_l2": -2.0, "base_height_l2": -1.0,
            "torques_l2": -0.00002, "action_rate_l2": -0.02,
            "joint_pos_limits": -1.0, "feet_slip": -0.1, "termination": -2.0,
        },
    )
    cfg.scene_binding = {
        "resolver": "lite3", "min_count": 1, "max_count": 1,
        "spawn_if_missing": False, "max_auto_spawn_count": 1,
        "spawn_agent_name": "lite3_000", "spawn_height": 0.0,
        "asset_path": LITE3_AGENT_ASSET_PATH,
    }
    if cfg.terrain is not None:
        cfg.terrain = replace(cfg.terrain, physics_enabled=False)
    cfg.rewards["base_height_l2"].params["asset_cfg"].body_names = ("TORSO",)
    cfg.rewards["feet_slip"].params["asset_cfg"].site_names = ()
    if play:
        cfg.episode["length_s"] = 1.0e9
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.events.pop("push_robot", None)
    return cfg


TASK_CONFIG_FACTORY = deeprobotics_lite3_flat_env_cfg
