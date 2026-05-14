from __future__ import annotations

import argparse
from pathlib import Path
import time

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    find_latest_checkpoint,
    load_task_and_train_cfg,
)
from orca_rl.rsl_env.scene_binding import G1_AGENT_ASSET_PATH, GO2_AGENT_ASSET_PATH

ensure_project_root_on_path()


def _apply_play_scene_mode(task_cfg: dict, *, local_mujoco: bool) -> None:
    task_cfg.setdefault("episode", {})["length_s"] = 1.0e9
    task_cfg.setdefault("observations", {})["add_noise"] = False
    task_cfg.setdefault("randomization", {})["enabled"] = False
    task_cfg["curriculum"] = {}
    events = task_cfg.setdefault("events", {})
    for event_name in (
        "push_robot",
        "randomize_friction",
        "randomize_body_mass",
        "randomize_actuator_properties",
        "randomize_action_latency",
        "randomize_solver_params",
        "randomize_contact_params",
    ):
        events.pop(event_name, None)
    randomization_cfg = task_cfg.setdefault("randomization", {})
    randomization_cfg["max_action_delay_steps"] = 0
    scene_cfg = task_cfg.setdefault("scene_binding", {})
    if local_mujoco:
        return
    if scene_cfg.get("resolver") == "g1":
        scene_cfg["local_xml_path"] = None
        scene_cfg["asset_path"] = G1_AGENT_ASSET_PATH
        scene_cfg["spawn_if_missing"] = True
        scene_cfg["max_auto_spawn_count"] = max(1, int(task_cfg.get("num_envs", 1)))
        terrain_cfg = task_cfg.get("terrain")
        if isinstance(terrain_cfg, dict):
            terrain_cfg["physics_enabled"] = False
    elif scene_cfg.get("resolver") == "go2":
        scene_cfg["asset_path"] = GO2_AGENT_ASSET_PATH
        scene_cfg["spawn_if_missing"] = True
        scene_cfg["max_auto_spawn_count"] = max(1, int(task_cfg.get("num_envs", 1)))
        terrain_cfg = task_cfg.get("terrain")
        if isinstance(terrain_cfg, dict):
            terrain_cfg["physics_enabled"] = False


def _apply_fixed_play_command(
    task_cfg: dict,
    *,
    lin_vel_x: float | None,
    lin_vel_y: float | None,
    ang_vel_z: float | None,
) -> None:
    if lin_vel_x is None and lin_vel_y is None and ang_vel_z is None:
        return
    commands = task_cfg.setdefault("commands", {})
    if lin_vel_x is not None:
        commands["lin_vel_x"] = (float(lin_vel_x), float(lin_vel_x))
    if lin_vel_y is not None:
        commands["lin_vel_y"] = (float(lin_vel_y), float(lin_vel_y))
    if ang_vel_z is not None:
        commands["yaw_vel"] = (float(ang_vel_z), float(ang_vel_z))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Orca locomotion RSL-RL policy.")
    parser.add_argument(
        "--config",
        default="Unitree-GO2-Flat",
        help="Registered task name, or Python cfg file. Use file.py:factory_name for a non-default factory.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List registered task names and exit.")
    parser.add_argument(
        "--mjlab",
        action="store_true",
        help="Shortcut for playing the latest Unitree/mjlab velocity checkpoint for the selected config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="RSL-RL checkpoint path, e.g. model_1000.pt.",
    )
    parser.add_argument(
        "--policy-backend",
        choices=("orca", "mjlab"),
        default="orca",
        help="`orca` loads checkpoints trained by orca_rl; `mjlab` loads Unitree/mjlab velocity checkpoints.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Playback duration in seconds. Overrides --steps when set.",
    )
    parser.add_argument("--lin-vel-x", type=float, default=None, help="Fixed x velocity command for play.")
    parser.add_argument("--lin-vel-y", type=float, default=None, help="Fixed y velocity command for play.")
    parser.add_argument("--ang-vel-z", type=float, default=None, help="Fixed yaw velocity command for play.")
    parser.add_argument(
        "--local-mujoco",
        action="store_true",
        help="Play through the generated local MuJoCo MJCF instead of the OrcaLab scene.",
    )
    parser.add_argument(
        "--remote",
        default=None,
        help="Override OrcaGym address, e.g. localhost:50051.",
    )
    args = parser.parse_args()

    if args.list_tasks:
        from orca_rl.registry import list_tasks

        for spec in list_tasks():
            suffix = f" - {spec.description}" if spec.description else ""
            print(f"{spec.name}{suffix}")
        return

    if args.mjlab:
        args.policy_backend = "mjlab"
        if args.lin_vel_x is None:
            args.lin_vel_x = 0.5
        if args.lin_vel_y is None:
            args.lin_vel_y = 0.0
        if args.ang_vel_z is None:
            args.ang_vel_z = 0.0

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    _apply_fixed_play_command(
        task_cfg,
        lin_vel_x=args.lin_vel_x,
        lin_vel_y=args.lin_vel_y,
        ang_vel_z=args.ang_vel_z,
    )
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    check_orcagym_addresses(task_cfg)

    try:
        import torch

        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.rsl_env.runtime_policy import load_inference_runner
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    robot_name = str(task_cfg.get("robot", "")).strip().lower()
    if args.policy_backend == "mjlab":
        from orca_rl.rsl_env.mjlab_policy import find_latest_unitree_mjlab_checkpoint

        project_root = Path(__file__).resolve().parents[1]
        checkpoint = str(args.checkpoint or find_latest_unitree_mjlab_checkpoint(project_root, robot=robot_name or "g1"))
    else:
        checkpoint = args.checkpoint or str(
            find_latest_checkpoint(task_name=str(train_cfg.get("experiment_name") or task_cfg.get("name", "")) or None)
        )
    try:
        env = make_locomotion_vec_env(task_cfg, device=device, render_mode="human", headless=False)
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc
    try:
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=checkpoint,
        )
        if args.policy_backend == "mjlab":
            from orca_rl.rsl_env.mjlab_policy import MjlabRslRlActorPolicy, make_mjlab_orca_play_bridge

            policy = MjlabRslRlActorPolicy.from_checkpoint(checkpoint, device=device)
            if policy.output_dim != env.num_actions:
                raise ValueError(
                    "Unitree/mjlab checkpoint action dimension does not match the OrcaLab scene robot. "
                    f"checkpoint_action_dim={policy.output_dim}, env_num_actions={env.num_actions}, "
                    f"config_robot={robot_name or 'unknown'}"
                )
            bridge = make_mjlab_orca_play_bridge(env, expected_obs_dim=policy.input_dim, robot=robot_name)
            obs = bridge.get_observations()
            alignment = bridge.alignment_report
            print(
                "[orca_rl.play] Loaded Unitree/mjlab policy bridge: "
                f"obs_dim={policy.input_dim}, action_dim={policy.output_dim}, checkpoint={checkpoint}"
            )
            print(
                "[orca_rl.play] Mjlab runtime alignment: "
                f"tasks={alignment['tasks']}, agents={alignment['agents']}, "
                f"joints={alignment['joints']}, actuators={alignment['actuators']}, "
                f"position_actuator_tasks={alignment['position_actuator_tasks']}"
            )
            if "foot_contact_geoms" in alignment:
                print(
                    "[orca_rl.play] GO2 mjlab contact alignment: "
                    f"contact_geoms={alignment.get('contact_geoms', 0)}, "
                    f"foot_contact_geoms={alignment.get('foot_contact_geoms', 0)}, "
                    f"nonfoot_contact_geoms={alignment.get('nonfoot_contact_geoms', 0)}, "
                    f"base_height_resets={alignment.get('base_height_resets', 0)}, "
                    f"imu_gyro_sensors={alignment.get('imu_gyro_sensors', 0)}"
                )
            if args.lin_vel_x is not None or args.lin_vel_y is not None or args.ang_vel_z is not None:
                print(
                    "[orca_rl.play] Fixed command: "
                    f"vx={args.lin_vel_x if args.lin_vel_x is not None else 'sampled'}, "
                    f"vy={args.lin_vel_y if args.lin_vel_y is not None else 'sampled'}, "
                    f"wz={args.ang_vel_z if args.ang_vel_z is not None else 'sampled'}"
                )
        else:
            _runner, policy = load_inference_runner(env, train_cfg, checkpoint, log_dir=None, device=device)
            obs = env.get_observations().to(device)
        step = 0
        dt = float(task_cfg["sim"]["time_step"]) * int(task_cfg["sim"]["frame_skip"]) * int(task_cfg["sim"]["decimation"])
        max_steps = int(round(float(args.seconds) / dt)) if args.seconds is not None else int(args.steps)
        while max_steps <= 0 or step < max_steps:
            start = time.perf_counter()
            if args.policy_backend == "mjlab":
                actions_np = policy.act_numpy(obs, device=device)
                obs = bridge.step(actions_np)
            else:
                with torch.inference_mode():
                    actions = policy(obs, stochastic_output=False)
                obs, _rewards, _dones, _extras = env.step(actions.to(env.device))
                obs = obs.to(device)
            step += 1
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Interrupted RSL-RL playback.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
