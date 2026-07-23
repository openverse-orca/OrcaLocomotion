from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

from orca_rl.heft_keyboard import clip_xy_command, make_keyboard_backend
from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
)


ensure_project_root_on_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "checkpoints/heft/G1_PMG/policy.onnx"
DEFAULT_MOTION_DIR = PROJECT_ROOT / "assets/heft/recorded_commands"
DEFAULT_WALK_MOTION_DIR = PROJECT_ROOT / "assets/heft/motions"

HELP = """
G1 + Dex3 HEFT velocity command

Move:
  W / Up / 8       forward            S / Down         backward
  A / Left / 4     left               D / Right / 6    right
  Z / 7            yaw left           C / 9            yaw right

Original HEFT motions:
  F1                walk1              F2               walk2
  F3                walk3

Control:
  Space / 5        stand              R                 reset robot and velocity
  Q / Esc          quit
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Velocity-command HEFT G1 play for OrcaLab.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--motion-dir", default=str(DEFAULT_MOTION_DIR))
    parser.add_argument("--walk-motion-dir", default=str(DEFAULT_WALK_MOTION_DIR))
    parser.add_argument("--remote", default=None, help="OrcaGym address override, e.g. localhost:50051.")
    parser.add_argument("--keyboard-backend", choices=("auto", "global", "terminal", "none"), default="auto")
    parser.add_argument("--command-speed", type=float, default=0.5, help="Forward/lateral speed in m/s.")
    parser.add_argument("--yaw-speed", type=float, default=0.8, help="Yaw speed in rad/s.")
    parser.add_argument("--max-speed", type=float, default=0.8, help="Maximum xy command norm.")
    parser.add_argument("--lin-vel-x", type=float, default=0.0)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--transition-s", type=float, default=0.4)
    parser.add_argument("--onnx-threads", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()

    try:
        from orca_rl.heft_env import make_heft_env
        from orca_rl.rsl_env.heft_policy import HeftG1RecordedCommandBridge
        from orca_rl.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    task_cfg = unitree_g1_flat_env_cfg(play=True).to_dict()
    apply_remote_override(task_cfg, args.remote)
    _configure_heft_scene(task_cfg)
    check_orcagym_addresses(task_cfg)

    command = clip_xy_command(
        np.asarray([args.lin_vel_x, args.lin_vel_y, args.ang_vel_z], dtype=np.float64),
        max_speed=float(args.max_speed),
    )
    keyboard = make_keyboard_backend(
        str(args.keyboard_backend),
        initial_command=command,
        command_speed=float(args.command_speed),
        yaw_speed=float(args.yaw_speed),
        max_speed=float(args.max_speed),
    )
    env = make_heft_env(task_cfg)
    try:
        print(
            "[orca_rl.heft] "
            f"robot=G1+Dex3-1 action_dim={env.num_actions} envs={env.num_envs} "
            f"remote={task_cfg['orcagym_addresses'][0]}"
        )
        dt = float(env.tasks[0].control_dt)
        bridge = HeftG1RecordedCommandBridge(
            env,
            policy_path=args.policy,
            motion_dir=args.motion_dir,
            walk_motion_dir=args.walk_motion_dir,
            transition_steps=max(0, int(round(float(args.transition_s) / dt))),
            onnx_threads=args.onnx_threads,
        )
        bridge.reset(reset_env=True)
        bridge.set_commands(np.repeat(command.reshape(1, 3), env.num_envs, axis=0))
        print(HELP)
        print(
            f"Policy={Path(args.policy).expanduser().resolve()}  keyboard={keyboard.name}  "
            f"recorded_motions={Path(args.motion_dir).expanduser().resolve()}"
        )

        start_time = time.perf_counter()
        last_status = 0.0
        with keyboard:
            while args.seconds is None or time.perf_counter() - start_time < float(args.seconds):
                step_start = time.perf_counter()
                state = keyboard.poll()
                if state.quit_requested:
                    break
                command = state.command
                if state.reset_requested:
                    bridge.reset(reset_env=True)
                    command[:] = 0.0
                    print("\n[orca_rl.heft] reset; velocity command cleared.")
                if state.motion_selection is not None:
                    bridge.select_motion_key(state.motion_selection)
                else:
                    bridge.set_commands(np.repeat(command.reshape(1, 3), env.num_envs, axis=0))
                actions = bridge.act()
                bridge.step(actions)

                now = time.perf_counter()
                if now - last_status >= 0.25:
                    sys.stdout.write(
                        f"\rvx={command[0]: .2f} vy={command[1]: .2f} wz={command[2]: .2f} "
                        f"clip={bridge.current_motion:<24} max|action|={float(np.abs(actions).max()):5.2f}   "
                    )
                    sys.stdout.flush()
                    last_status = now
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _configure_heft_scene(task_cfg: dict) -> None:
    task_cfg["num_envs"] = 1
    task_cfg.setdefault("sim", {}).update(
        render_mode="human",
        headless=False,
        unitree_play_global_settings=True,
    )
    task_cfg.setdefault("episode", {})["length_s"] = 1.0e9
    task_cfg.setdefault("observations", {})["add_noise"] = False
    task_cfg["curriculum"] = {}
    randomization = task_cfg.setdefault("randomization", {})
    randomization.update(enabled=False, max_action_delay_steps=0)
    events = task_cfg.setdefault("events", {})
    for name in tuple(events):
        if name.startswith("randomize_") or name == "push_robot":
            events.pop(name, None)
    terrain = task_cfg.get("terrain")
    if isinstance(terrain, dict) and terrain.get("terrain_type") == "plane":
        terrain["physics_enabled"] = False


if __name__ == "__main__":
    main()
