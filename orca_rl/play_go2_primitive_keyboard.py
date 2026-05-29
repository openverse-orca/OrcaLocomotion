from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from orca_rl.play_g1_grabbox_keyboard import _clip_xy_command, _make_keyboard_backend, _print_status
from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    load_task_and_train_cfg,
)

ensure_project_root_on_path()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/test_model_Go2_mjlab_Rough.pt"
DEFAULT_ROUGH_TERRAIN_ASSET_PATH = "assets/e071469a36d3c8aa/test_terrain/prefabs/terrain_usda"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleop for Go2 rough play with the mjlab rough terrain visual.")
    parser.add_argument("--config", default="Unitree-Go2-Rough", help="Registered Go2 rough task or cfg path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Unitree/mjlab Go2 rough checkpoint path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--remote", default=None, help="Override OrcaGym address, e.g. localhost:50051.")
    parser.add_argument("--local-mujoco", action="store_true", help="Run against local MuJoCo XML.")
    parser.add_argument(
        "--keyboard-backend",
        choices=("auto", "global", "terminal", "none"),
        default="auto",
        help="Keyboard source. global listens while OrcaLab has focus; terminal reads stdin.",
    )
    parser.add_argument("--max-speed", type=float, default=1.0, help="Maximum xy command norm.")
    parser.add_argument("--command-speed", type=float, default=0.5, help="XY speed sent by one arrow/keypad direction.")
    parser.add_argument("--yaw-speed", type=float, default=0.8, help="Yaw rate sent by Z/C or keypad 7/9.")
    parser.add_argument("--lin-vel-x", type=float, default=0.5, help="Initial commanded forward velocity.")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial commanded lateral velocity.")
    parser.add_argument(
        "--robot-spawn-height",
        type=float,
        default=None,
        help="Initial Go2 actor spawn z for OrcaLab auto-publish. Defaults to --terrain-z.",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Optional run duration.")
    parser.add_argument(
        "--terrain-asset",
        default=DEFAULT_ROUGH_TERRAIN_ASSET_PATH,
        help="OrcaLab spawnable path for the mjlab rough terrain visual asset.",
    )
    parser.add_argument("--terrain-actor", default="mjlab_rough_5x5_terrain", help="Terrain actor name in OrcaLab.")
    parser.add_argument("--terrain-z", type=float, default=0.05, help="Terrain actor spawn z in OrcaLab.")
    parser.add_argument("--terrain-scale", type=float, default=1.0, help="Terrain actor scale.")
    parser.add_argument("--no-terrain", action="store_true", help="Do not publish the rough terrain visual.")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401

        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.run_play import _apply_play_scene_mode, _set_env_manual_command
        from orca_rl.rsl_env.mjlab_policy import (
            MjlabRslRlActorPolicy,
            find_latest_unitree_mjlab_checkpoint,
            make_mjlab_orca_play_bridge,
        )
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    robot_spawn_height = float(args.terrain_z if args.robot_spawn_height is None else args.robot_spawn_height)
    task_cfg.setdefault("scene_binding", {})["spawn_height"] = robot_spawn_height
    _configure_rough_terrain_actor(task_cfg, args, local_mujoco=bool(args.local_mujoco))
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    check_orcagym_addresses(task_cfg)
    _maybe_publish_rough_terrain(task_cfg, args, local_mujoco=bool(args.local_mujoco))

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    checkpoint = args.checkpoint or str(
        find_latest_unitree_mjlab_checkpoint(PROJECT_ROOT, robot="go2", terrain="rough")
    )
    env = make_locomotion_vec_env(task_cfg, device=device, render_mode="human", headless=False)
    try:
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=str(checkpoint),
        )
        policy = MjlabRslRlActorPolicy.from_checkpoint(checkpoint, device=device)
        bridge = make_mjlab_orca_play_bridge(env, expected_obs_dim=policy.input_dim, robot="go2")

        command = _clip_xy_command(
            np.array([float(args.lin_vel_x), float(args.lin_vel_y), 0.0], dtype=np.float64),
            max_speed=float(args.max_speed),
        )
        keyboard = _make_keyboard_backend(
            args.keyboard_backend,
            initial_command=command,
            command_speed=float(args.command_speed),
            yaw_speed=float(args.yaw_speed),
            max_speed=float(args.max_speed),
        )
        _set_env_manual_command(env, command)
        obs = bridge.get_observations()

        print(
            "\nGo2 rough terrain teleop ready: hold keypad/arrows Up/Down for vx, Left/Right for vy, "
            "Z/C or keypad 7/9 for yaw, release to stop, Space/5 zeros command, Q quits. "
            f"Keyboard backend={keyboard.name}. "
            f"Terrain actor={args.terrain_actor}, asset={args.terrain_asset}.\n"
        )
        start_time = time.perf_counter()
        last_status = 0.0
        dt = float(task_cfg["sim"]["time_step"]) * int(task_cfg["sim"]["frame_skip"]) * int(task_cfg["sim"]["decimation"])
        with keyboard:
            while args.seconds is None or time.perf_counter() - start_time < float(args.seconds):
                step_start = time.perf_counter()
                command, quit_requested = keyboard.poll()
                if quit_requested:
                    break
                _set_env_manual_command(env, command)
                obs = bridge.get_observations()
                obs = bridge.step(policy.act_numpy(obs, device=device))

                now = time.perf_counter()
                if now - last_status > 0.25:
                    _print_status(command)
                    last_status = now
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _configure_rough_terrain_actor(task_cfg: dict, args: argparse.Namespace, *, local_mujoco: bool) -> None:
    if local_mujoco or bool(args.no_terrain):
        return
    task_cfg.setdefault("scene_binding", {}).setdefault("extra_actors", []).append(
        {
            "name": str(args.terrain_actor),
            "asset_path": str(args.terrain_asset),
            "position": [0.0, 0.0, float(args.terrain_z)],
            "rotation_euler": [0.0, 0.0, 0.0],
            "scale": float(args.terrain_scale),
        }
    )


def _maybe_publish_rough_terrain(task_cfg: dict, args: argparse.Namespace, *, local_mujoco: bool) -> None:
    if local_mujoco or bool(args.no_terrain):
        return
    try:
        from orca_rl.rsl_env.debug_visualizer import ensure_scene_actor
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    for address in task_cfg.get("orcagym_addresses") or ["localhost:50051"]:
        try:
            published = ensure_scene_actor(
                orcagym_addr=str(address),
                actor_name=str(args.terrain_actor),
                asset_path=str(args.terrain_asset),
                position=[0.0, 0.0, float(args.terrain_z)],
                rotation_euler=[0.0, 0.0, 0.0],
                scale=float(args.terrain_scale),
            )
        except Exception as exc:
            print(f"[orca_rl.go2_primitive_keyboard] Rough terrain auto-publish skipped for {address}: {exc}")
            continue
        if published:
            print(
                "[orca_rl.go2_primitive_keyboard] Rough terrain auto-published: "
                f"address={address}, actor={args.terrain_actor}, asset={args.terrain_asset}"
            )


if __name__ == "__main__":
    main()
