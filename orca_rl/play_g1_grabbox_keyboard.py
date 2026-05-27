from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Protocol

import numpy as np

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    load_task_and_train_cfg,
)

ensure_project_root_on_path()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRABBOX_CSV = PROJECT_ROOT / "GrabBox_put_it_DOWN_v2.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/test_model_G1_mjlab_Flat.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard teleop for G1 flat play with a fixed grab-box upper-body pose."
    )
    parser.add_argument("--config", default="Unitree-G1-Flat", help="Registered G1 task or cfg path.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Unitree/mjlab G1 checkpoint path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--remote", default=None, help="Override OrcaGym address, e.g. localhost:50051.")
    parser.add_argument("--local-mujoco", action="store_true", help="Run against local MuJoCo XML.")
    parser.add_argument(
        "--keyboard-backend",
        choices=("auto", "global", "terminal", "none"),
        default="auto",
        help="Keyboard source. global listens while OrcaLab has focus; terminal reads stdin.",
    )
    parser.add_argument("--csv", default=str(DEFAULT_GRABBOX_CSV), help="Grab-box motion CSV.")
    parser.add_argument("--row", type=int, default=270, help="CSV row used for the initial and fixed arm pose.")
    parser.add_argument(
        "--apply-csv-initial-state",
        action="store_true",
        help="Also reset the whole robot to the CSV row before play. Off by default because it can topple the policy.",
    )
    parser.add_argument("--max-speed", type=float, default=1.0, help="Maximum xy command norm.")
    parser.add_argument("--command-speed", type=float, default=0.5, help="XY speed sent by one arrow/keypad direction.")
    parser.add_argument("--yaw-speed", type=float, default=0.8, help="Yaw rate sent by Z/C or keypad 7/9.")
    parser.add_argument("--speed-step", type=float, default=None, help="Deprecated alias for --command-speed.")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Initial commanded forward velocity.")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial commanded lateral velocity.")
    parser.add_argument(
        "--arm-pose-weight",
        type=float,
        default=0.5,
        help="Blend weight for the grab-box arm pose. 0 uses policy arms, 1 fully overrides arms.",
    )
    parser.add_argument(
        "--fixed-joints",
        choices=("arms", "upper-body"),
        default="arms",
        help="Which joints are held at the CSV pose during play.",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Optional run duration.")
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
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    check_orcagym_addresses(task_cfg)

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    checkpoint = args.checkpoint or str(
        find_latest_unitree_mjlab_checkpoint(PROJECT_ROOT, robot="g1", terrain="flat")
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
        bridge = make_mjlab_orca_play_bridge(env, expected_obs_dim=policy.input_dim, robot="g1")

        frame = _load_csv_frame(args.csv, args.row)
        if args.apply_csv_initial_state:
            _apply_initial_frame(env, frame)
        fixed_action = _fixed_pose_action(env, bridge.action_spec.scale, frame[7:], args.fixed_joints)

        command_speed = float(args.command_speed if args.speed_step is None else args.speed_step)
        arm_pose_weight = float(np.clip(float(args.arm_pose_weight), 0.0, 1.0))
        command = _clip_xy_command(
            np.array([float(args.lin_vel_x), float(args.lin_vel_y), 0.0], dtype=np.float64),
            max_speed=float(args.max_speed),
        )
        keyboard = _make_keyboard_backend(
            args.keyboard_backend,
            initial_command=command,
            command_speed=command_speed,
            yaw_speed=float(args.yaw_speed),
            max_speed=float(args.max_speed),
        )
        _set_env_manual_command(env, command)
        obs = bridge.get_observations()

        print(
            "\nOrcaLab teleop ready: hold keypad/arrows Up/Down for vx, Left/Right for vy, "
            "Z/C or keypad 7/9 for yaw, release to stop, Space/5 zeros command, Q quits. "
            f"direction speed={command_speed:.2f}, xy speed is clipped to norm <= {float(args.max_speed):.2f}. "
            f"yaw speed={float(args.yaw_speed):.2f} rad/s. "
            f"Initial command: vx={command[0]:.2f}, vy={command[1]:.2f}. "
            f"Keyboard backend={keyboard.name}. "
            f"Arm pose weight={arm_pose_weight:.2f}. Fixed joints: {args.fixed_joints}. CSV row: {args.row}.\n"
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

                actions = policy.act_numpy(obs, device=device)
                if arm_pose_weight > 0.0:
                    policy_arm_actions = actions[:, fixed_action.mask]
                    fixed_arm_actions = fixed_action.values.reshape(1, -1)
                    actions[:, fixed_action.mask] = (
                        (1.0 - arm_pose_weight) * policy_arm_actions + arm_pose_weight * fixed_arm_actions
                    )
                obs = bridge.step(actions)

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


class FixedAction:
    def __init__(self, mask: np.ndarray, values: np.ndarray) -> None:
        self.mask = mask
        self.values = values


class KeyboardBackend(Protocol):
    name: str

    def __enter__(self) -> "KeyboardBackend": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...

    def poll(self) -> tuple[np.ndarray, bool]: ...


class NoKeyboardBackend:
    name = "none"

    def __init__(self, initial_command: np.ndarray) -> None:
        self.command = np.asarray(initial_command, dtype=np.float64).reshape(3).copy()

    def __enter__(self) -> "NoKeyboardBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def poll(self) -> tuple[np.ndarray, bool]:
        return self.command.copy(), False


class TerminalKeyboardBackend:
    name = "terminal"

    def __init__(self, initial_command: np.ndarray, *, command_speed: float, yaw_speed: float, max_speed: float) -> None:
        self.command = np.asarray(initial_command, dtype=np.float64).reshape(3).copy()
        self.command_speed = float(command_speed)
        self.yaw_speed = float(yaw_speed)
        self.max_speed = float(max_speed)
        self.quit_requested = False
        self._old_settings = None

    def __enter__(self) -> "TerminalKeyboardBackend":
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def poll(self) -> tuple[np.ndarray, bool]:
        key = _read_terminal_key()
        if key in {"q", "Q", "\x03", "\x04"}:
            self.quit_requested = True
        elif key:
            self.command = _command_from_direction_key(
                key,
                command_speed=self.command_speed,
                yaw_speed=self.yaw_speed,
                max_speed=self.max_speed,
                current=self.command,
            )
        return self.command.copy(), self.quit_requested


class GlobalKeyboardBackend:
    name = "global"

    def __init__(self, initial_command: np.ndarray, *, command_speed: float, yaw_speed: float, max_speed: float) -> None:
        self.command = np.asarray(initial_command, dtype=np.float64).reshape(3).copy()
        self.command_speed = float(command_speed)
        self.yaw_speed = float(yaw_speed)
        self.max_speed = float(max_speed)
        self.quit_requested = False
        self._lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._listener = None

    def __enter__(self) -> "GlobalKeyboardBackend":
        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError as exc:
            raise RuntimeError(
                "Global keyboard backend needs pynput. Install it in the runtime env with: "
                "python -m pip install pynput"
            ) from exc

        self._pynput_keyboard = pynput_keyboard
        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def poll(self) -> tuple[np.ndarray, bool]:
        with self._lock:
            return self.command.copy(), self.quit_requested

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            if name in {"q", "Q"}:
                self.quit_requested = True
                return
            if name in {"up", "down", "left", "right", "8", "2", "4", "6", "7", "9", "z", "Z", "c", "C"}:
                self._active_keys.add(name)
                self.command = _command_from_active_keys(
                    self._active_keys,
                    command_speed=self.command_speed,
                    yaw_speed=self.yaw_speed,
                    max_speed=self.max_speed,
                )
            elif name in {"space", "5"}:
                self._active_keys.clear()
                self.command[:2] = 0.0

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            self._active_keys.discard(name)
            if name in {"up", "down", "left", "right", "8", "2", "4", "6", "7", "9", "z", "Z", "c", "C"}:
                self.command = _command_from_active_keys(
                    self._active_keys,
                    command_speed=self.command_speed,
                    yaw_speed=self.yaw_speed,
                    max_speed=self.max_speed,
                )

    def _key_name(self, key) -> str:
        keyboard_module = self._pynput_keyboard
        if key == keyboard_module.Key.up:
            return "up"
        if key == keyboard_module.Key.down:
            return "down"
        if key == keyboard_module.Key.left:
            return "left"
        if key == keyboard_module.Key.right:
            return "right"
        if key == keyboard_module.Key.space:
            return "space"
        char = getattr(key, "char", None)
        return str(char) if char is not None else str(key)


def _make_keyboard_backend(
    backend: str,
    *,
    initial_command: np.ndarray,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
) -> KeyboardBackend:
    if backend == "none":
        return NoKeyboardBackend(initial_command)
    if backend == "terminal":
        return TerminalKeyboardBackend(
            initial_command,
            command_speed=command_speed,
            yaw_speed=yaw_speed,
            max_speed=max_speed,
        )
    if backend == "global":
        return GlobalKeyboardBackend(
            initial_command,
            command_speed=command_speed,
            yaw_speed=yaw_speed,
            max_speed=max_speed,
        )
    try:
        import pynput  # noqa: F401
    except ImportError:
        print(
            "[orca_rl.teleop] pynput is not installed; falling back to terminal keyboard input. "
            "For OrcaLab-window control, install: python -m pip install pynput"
        )
        return TerminalKeyboardBackend(
            initial_command,
            command_speed=command_speed,
            yaw_speed=yaw_speed,
            max_speed=max_speed,
        )
    return GlobalKeyboardBackend(
        initial_command,
        command_speed=command_speed,
        yaw_speed=yaw_speed,
        max_speed=max_speed,
    )


def _load_csv_frame(path: str | Path, row: int) -> np.ndarray:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Grab-box CSV does not exist: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 36:
        raise ValueError(f"Expected grab-box CSV shape (N, 36), got {data.shape} from {csv_path}")
    row_index = int(row)
    if row_index < 0:
        row_index += data.shape[0]
    if row_index < 0 or row_index >= data.shape[0]:
        raise IndexError(f"CSV row {row} is outside valid range [0, {data.shape[0] - 1}]")
    frame = data[row_index].copy()
    quat_norm = np.linalg.norm(frame[3:7])
    if quat_norm <= 1.0e-9:
        raise ValueError(f"CSV row {row} has an invalid base quaternion.")
    frame[3:7] /= quat_norm
    return frame


def _apply_initial_frame(env, frame: np.ndarray) -> None:
    for task in getattr(env, "tasks", []):
        joint_qpos: dict[str, np.ndarray] = {}
        joint_qvel: dict[str, np.ndarray] = {}
        for agent in task.agents:
            base_qpos = frame[:7].copy()
            joint_qpos[agent.base_joint_name] = base_qpos
            joint_qvel[agent.base_joint_name] = np.zeros(6, dtype=np.float64)
            agent.initial_base_qpos = base_qpos.copy()
            for joint_name, joint_value in zip(agent.leg_joint_names, frame[7:]):
                joint_qpos[joint_name] = np.array([joint_value], dtype=np.float64)
                joint_qvel[joint_name] = np.zeros(1, dtype=np.float64)
        task.set_joint_qpos(joint_qpos)
        task.set_joint_qvel(joint_qvel)
        task.ctrl[:] = 0.0
        task.set_ctrl(task.ctrl)
        task.mj_forward()
        task.update_data()


def _fixed_pose_action(env, action_scale: np.ndarray, target_qpos: np.ndarray, fixed_joints: str) -> FixedAction:
    if not getattr(env, "tasks", None):
        raise ValueError("The Orca env has no tasks.")
    task = env.tasks[0]
    joint_names = list(task.robot_config.get("leg_joint_names") or [])
    mask = np.array([_is_fixed_joint(name, fixed_joints) for name in joint_names], dtype=bool)
    if not np.any(mask):
        raise ValueError(f"No joints matched fixed-joints mode: {fixed_joints}")
    nominal_qpos = np.asarray(task._nominal_qpos[0], dtype=np.float64)
    scale = np.asarray(action_scale, dtype=np.float64)
    values = (np.asarray(target_qpos, dtype=np.float64) - nominal_qpos) / np.where(np.abs(scale) > 1e-9, scale, 1.0)
    values = np.clip(values, -1.0, 1.0)
    return FixedAction(mask=mask, values=values[mask].astype(np.float64))


def _is_fixed_joint(name: str, mode: str) -> bool:
    suffix = name.rsplit("/", 1)[-1]
    is_arm = any(part in suffix for part in ("shoulder", "elbow", "wrist"))
    if mode == "arms":
        return is_arm
    return is_arm or suffix.startswith("waist_")


def _command_from_direction_key(
    key: str,
    *,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
    current: np.ndarray,
) -> np.ndarray:
    updated = np.asarray(current, dtype=np.float64).reshape(3).copy()
    speed = min(abs(float(command_speed)), max(0.0, float(max_speed)))
    if key in {"up", "8"}:
        updated[:2] = (speed, 0.0)
    elif key in {"down", "2"}:
        updated[:2] = (-speed, 0.0)
    elif key in {"left", "4"}:
        updated[:2] = (0.0, speed)
    elif key in {"right", "6"}:
        updated[:2] = (0.0, -speed)
    elif key in {"7", "z", "Z"}:
        updated[:] = (0.0, 0.0, float(yaw_speed))
    elif key in {"9", "c", "C"}:
        updated[:] = (0.0, 0.0, -float(yaw_speed))
    elif key in {" ", "5"}:
        updated[:] = 0.0
    return _clip_xy_command(updated, max_speed=max_speed)


def _command_from_active_keys(
    active_keys: set[str],
    *,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
) -> np.ndarray:
    command = np.zeros(3, dtype=np.float64)
    speed = min(abs(float(command_speed)), max(0.0, float(max_speed)))
    if "up" in active_keys or "8" in active_keys:
        command[0] += speed
    if "down" in active_keys or "2" in active_keys:
        command[0] -= speed
    if "left" in active_keys or "4" in active_keys:
        command[1] += speed
    if "right" in active_keys or "6" in active_keys:
        command[1] -= speed
    if "7" in active_keys or "z" in active_keys or "Z" in active_keys:
        command[2] += float(yaw_speed)
    if "9" in active_keys or "c" in active_keys or "C" in active_keys:
        command[2] -= float(yaw_speed)
    return _clip_xy_command(command, max_speed=max_speed)


def _clip_xy_command(command: np.ndarray, *, max_speed: float) -> np.ndarray:
    updated = np.asarray(command, dtype=np.float64).reshape(3).copy()
    norm = float(np.linalg.norm(updated[:2]))
    limit = max(0.0, float(max_speed))
    if norm > limit > 0.0:
        updated[:2] *= limit / norm
    return updated


def _read_terminal_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    char = sys.stdin.read(1)
    if char != "\x1b":
        return char
    readable, _, _ = select.select([sys.stdin], [], [], 0.001)
    if not readable:
        return char
    second = sys.stdin.read(1)
    readable, _, _ = select.select([sys.stdin], [], [], 0.001)
    if not readable:
        return char + second
    third = sys.stdin.read(1)
    if second == "[":
        return {
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
        }.get(third, char + second + third)
    if second == "O":
        return {
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
            "x": "up",
            "r": "down",
            "t": "left",
            "v": "right",
            "u": "5",
        }.get(third, char + second + third)
    return char + second + third


def _print_status(command: np.ndarray) -> None:
    sys.stdout.write(f"\rcommand vx={command[0]: .2f} vy={command[1]: .2f} |norm|={np.linalg.norm(command[:2]):.2f}   ")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
