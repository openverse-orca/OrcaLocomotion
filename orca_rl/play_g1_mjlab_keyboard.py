from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/test_model_G1_mjlab_Flat.pt"

HELP = """
G1 mjlab keyboard play

Move (hold with global backend; tap-to-set with terminal backend):
  W / Up / 8       forward            S / Down / 2     backward
  A / Left / 4     left               D / Right / 6    right
  Z / 7            yaw left           C / 9            yaw right

Control:
  Space / 5        zero command       R                 reset robot and command
  Q / Esc          quit               ?                 print this help
"""


@dataclass(frozen=True)
class KeyboardState:
    command: np.ndarray
    reset_requested: bool
    quit_requested: bool
    motion_selection: str | None = None


class KeyboardBackend(Protocol):
    name: str

    def __enter__(self) -> "KeyboardBackend": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...

    def poll(self) -> KeyboardState: ...


class NoKeyboardBackend:
    name = "none"

    def __init__(self, initial_command: np.ndarray) -> None:
        self.command = _as_command(initial_command)

    def __enter__(self) -> "NoKeyboardBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def poll(self) -> KeyboardState:
        return KeyboardState(self.command.copy(), False, False)


class TerminalKeyboardBackend:
    name = "terminal"

    def __init__(
        self,
        initial_command: np.ndarray,
        *,
        command_speed: float,
        yaw_speed: float,
        max_speed: float,
        enable_motion_selection: bool = False,
    ) -> None:
        self.command = _as_command(initial_command)
        self.command_speed = float(command_speed)
        self.yaw_speed = float(yaw_speed)
        self.max_speed = float(max_speed)
        self.enable_motion_selection = bool(enable_motion_selection)
        self.quit_requested = False
        self.reset_requested = False
        self.motion_selection: str | None = None
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

    def poll(self) -> KeyboardState:
        key = _read_terminal_key()
        if key in {"q", "Q", "esc", "\x03", "\x04"}:
            self.quit_requested = True
        elif key in {"r", "R"}:
            self.command[:] = 0.0
            self.reset_requested = True
            self.motion_selection = None
        elif self.enable_motion_selection and key in {"f1", "f2", "f3"}:
            self.command[:] = 0.0
            self.motion_selection = key
        elif key in {" ", "space", "5"}:
            self.command[:] = 0.0
            self.motion_selection = "stand"
        elif key == "?":
            print(HELP)
        elif key:
            self.command = _command_from_direction_key(
                key,
                command_speed=self.command_speed,
                yaw_speed=self.yaw_speed,
                max_speed=self.max_speed,
                current=self.command,
            )
        state = KeyboardState(
            self.command.copy(),
            self.reset_requested,
            self.quit_requested,
            self.motion_selection,
        )
        self.reset_requested = False
        self.motion_selection = None
        return state


class GlobalKeyboardBackend:
    name = "global"

    def __init__(
        self,
        initial_command: np.ndarray,
        *,
        command_speed: float,
        yaw_speed: float,
        max_speed: float,
        enable_motion_selection: bool = False,
    ) -> None:
        self.command = _as_command(initial_command)
        self.command_speed = float(command_speed)
        self.yaw_speed = float(yaw_speed)
        self.max_speed = float(max_speed)
        self.enable_motion_selection = bool(enable_motion_selection)
        self.quit_requested = False
        self.reset_requested = False
        self.motion_selection: str | None = None
        self._active_keys: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None

    def __enter__(self) -> "GlobalKeyboardBackend":
        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError as exc:
            raise RuntimeError("Global keyboard control requires pynput: python -m pip install pynput") from exc
        self._pynput_keyboard = pynput_keyboard
        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def poll(self) -> KeyboardState:
        with self._lock:
            state = KeyboardState(
                self.command.copy(),
                self.reset_requested,
                self.quit_requested,
                self.motion_selection,
            )
            self.reset_requested = False
            self.motion_selection = None
            return state

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            if name in {"q", "Q", "esc"}:
                self.quit_requested = True
            elif name in {"r", "R"}:
                self._active_keys.clear()
                self.command[:] = 0.0
                self.reset_requested = True
                self.motion_selection = None
            elif name in {"space", "5"}:
                self._active_keys.clear()
                self.command[:] = 0.0
                self.motion_selection = "stand"
            elif self.enable_motion_selection and name in {"f1", "f2", "f3"}:
                self._active_keys.clear()
                self.command[:] = 0.0
                self.motion_selection = name
            elif name == "?":
                print(HELP)
            elif name in _DIRECTION_KEYS:
                self._active_keys.add(name)
                self._update_command()

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            if name in self._active_keys:
                self._active_keys.discard(name)
                self._update_command()

    def _update_command(self) -> None:
        self.command = _command_from_active_keys(
            self._active_keys,
            command_speed=self.command_speed,
            yaw_speed=self.yaw_speed,
            max_speed=self.max_speed,
        )

    def _key_name(self, key) -> str:
        keyboard = self._pynput_keyboard
        special = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space",
            keyboard.Key.esc: "esc",
            keyboard.Key.f1: "f1",
            keyboard.Key.f2: "f2",
            keyboard.Key.f3: "f3",
        }
        if key in special:
            return special[key]
        char = getattr(key, "char", None)
        if char is not None:
            return str(char)
        digit = _numpad_digit(getattr(key, "vk", None))
        return digit if digit is not None else str(key)


_DIRECTION_KEYS = frozenset(
    {"w", "W", "s", "S", "a", "A", "d", "D", "up", "down", "left", "right", "8", "2", "4", "6", "7", "9", "z", "Z", "c", "C"}
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard command/reset play for the bundled mjlab G1 flat policy.")
    parser.add_argument("--config", default="Unitree-G1-Flat", help="Registered G1 task or cfg path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Unitree/mjlab G1 checkpoint.")
    parser.add_argument("--device", default=None, help="Torch inference device; defaults to the play config.")
    parser.add_argument("--remote", default=None, help="OrcaGym address override, e.g. localhost:50051.")
    parser.add_argument("--local-mujoco", action="store_true", help="Use the bundled local G1 MuJoCo model.")
    parser.add_argument("--keyboard-backend", choices=("auto", "global", "terminal", "none"), default="auto")
    parser.add_argument("--command-speed", type=float, default=0.5, help="Forward/lateral speed in m/s.")
    parser.add_argument("--yaw-speed", type=float, default=0.8, help="Yaw speed in rad/s.")
    parser.add_argument("--max-speed", type=float, default=1.0, help="Maximum xy command norm.")
    parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Initial forward command.")
    parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Initial lateral command.")
    parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Initial yaw command.")
    parser.add_argument("--arm-mode", choices=("neutral", "policy"), default="neutral")
    parser.add_argument("--reset-on-start", action="store_true", help="Reset once after loading the policy bridge.")
    parser.add_argument("--seconds", type=float, default=None, help="Optional run duration.")
    args = parser.parse_args()

    try:
        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.run_play import _apply_play_scene_mode, _set_env_manual_command
        from orca_rl.rsl_env.mjlab_policy import MjlabRslRlActorPolicy, make_mjlab_orca_play_bridge
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    if not args.local_mujoco:
        check_orcagym_addresses(task_cfg)

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    command = _clip_xy_command(
        np.array([args.lin_vel_x, args.lin_vel_y, args.ang_vel_z], dtype=np.float64),
        max_speed=float(args.max_speed),
    )
    keyboard = _make_keyboard_backend(
        str(args.keyboard_backend),
        initial_command=command,
        command_speed=float(args.command_speed),
        yaw_speed=float(args.yaw_speed),
        max_speed=float(args.max_speed),
    )

    env = make_locomotion_vec_env(task_cfg, device=device, render_mode="human", headless=False)
    try:
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=str(Path(args.checkpoint).expanduser()),
        )
        policy = MjlabRslRlActorPolicy.from_checkpoint(args.checkpoint, device=device)
        bridge = make_mjlab_orca_play_bridge(
            env,
            expected_obs_dim=policy.input_dim,
            robot="g1",
            g1_arm_mode=str(args.arm_mode),
        )
        if args.reset_on_start:
            _reset_env(env)
            command[:] = 0.0
        _set_env_manual_command(env, command)

        dt = float(task_cfg["sim"]["time_step"]) * int(task_cfg["sim"]["frame_skip"]) * int(task_cfg["sim"]["decimation"])
        print(HELP)
        print(
            f"Checkpoint={Path(args.checkpoint).expanduser().resolve()}  "
            f"keyboard={keyboard.name}  dt={dt:.3f}s  device={device}"
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
                    _reset_env(env)
                    command[:] = 0.0
                    print("\n[orca_rl.play] G1 reset; command cleared.")
                _set_env_manual_command(env, command)
                observations = bridge.get_observations()
                actions = policy.act_numpy(observations, device=device)
                bridge.step(actions)

                now = time.perf_counter()
                if now - last_status >= 0.25:
                    _print_status(command)
                    last_status = now
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _reset_env(env) -> None:
    env.reset()
    for task in getattr(env, "tasks", []):
        task.set_manual_commands(np.zeros((task.num_envs, 3), dtype=np.float64))


def _make_keyboard_backend(
    backend: str,
    *,
    initial_command: np.ndarray,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
    enable_motion_selection: bool = False,
) -> KeyboardBackend:
    kwargs = {
        "command_speed": command_speed,
        "yaw_speed": yaw_speed,
        "max_speed": max_speed,
        "enable_motion_selection": enable_motion_selection,
    }
    if backend == "none":
        return NoKeyboardBackend(initial_command)
    if backend == "terminal":
        return TerminalKeyboardBackend(initial_command, **kwargs)
    if backend == "global":
        return GlobalKeyboardBackend(initial_command, **kwargs)
    try:
        import pynput  # noqa: F401
    except ImportError:
        print("[orca_rl.play] pynput unavailable; using terminal keyboard backend.")
        return TerminalKeyboardBackend(initial_command, **kwargs)
    return GlobalKeyboardBackend(initial_command, **kwargs)


def _command_from_direction_key(
    key: str,
    *,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
    current: np.ndarray,
) -> np.ndarray:
    updated = _as_command(current)
    speed = min(abs(float(command_speed)), max(0.0, float(max_speed)))
    if key in {"w", "W", "up", "8"}:
        updated[:] = (speed, 0.0, 0.0)
    elif key in {"s", "S", "down", "2"}:
        updated[:] = (-speed, 0.0, 0.0)
    elif key in {"a", "A", "left", "4"}:
        updated[:] = (0.0, speed, 0.0)
    elif key in {"d", "D", "right", "6"}:
        updated[:] = (0.0, -speed, 0.0)
    elif key in {"z", "Z", "7"}:
        updated[:] = (0.0, 0.0, float(yaw_speed))
    elif key in {"c", "C", "9"}:
        updated[:] = (0.0, 0.0, -float(yaw_speed))
    elif key in {" ", "space", "5"}:
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
    if active_keys & {"w", "W", "up", "8"}:
        command[0] += speed
    if active_keys & {"s", "S", "down", "2"}:
        command[0] -= speed
    if active_keys & {"a", "A", "left", "4"}:
        command[1] += speed
    if active_keys & {"d", "D", "right", "6"}:
        command[1] -= speed
    if active_keys & {"z", "Z", "7"}:
        command[2] += float(yaw_speed)
    if active_keys & {"c", "C", "9"}:
        command[2] -= float(yaw_speed)
    return _clip_xy_command(command, max_speed=max_speed)


def _clip_xy_command(command: np.ndarray, *, max_speed: float) -> np.ndarray:
    updated = _as_command(command)
    norm = float(np.linalg.norm(updated[:2]))
    limit = max(0.0, float(max_speed))
    if norm > limit > 0.0:
        updated[:2] *= limit / norm
    elif limit == 0.0:
        updated[:2] = 0.0
    return updated


def _as_command(command: np.ndarray) -> np.ndarray:
    return np.asarray(command, dtype=np.float64).reshape(3).copy()


def _read_terminal_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first
    readable, _, _ = select.select([sys.stdin], [], [], 0.001)
    if not readable:
        return "esc"
    second = sys.stdin.read(1)
    readable, _, _ = select.select([sys.stdin], [], [], 0.001)
    if not readable:
        return "esc"
    third = sys.stdin.read(1)
    if second == "O":
        return {"P": "f1", "Q": "f2", "R": "f3"}.get(third, "esc")
    if second == "[":
        arrow = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(third)
        if arrow is not None:
            return arrow
        suffix = third
        while len(suffix) < 3:
            readable, _, _ = select.select([sys.stdin], [], [], 0.001)
            if not readable:
                break
            suffix += sys.stdin.read(1)
            if suffix.endswith("~"):
                break
        return {"11~": "f1", "12~": "f2", "13~": "f3"}.get(suffix, "esc")
    return "esc"


def _numpad_digit(vk) -> str | None:
    if vk is None:
        return None
    try:
        value = int(vk)
    except (TypeError, ValueError):
        return None
    if 96 <= value <= 105:
        return str(value - 96)
    if 65456 <= value <= 65465:
        return str(value - 65456)
    return None


def _print_status(command: np.ndarray) -> None:
    sys.stdout.write(
        f"\rcommand vx={command[0]: .2f} vy={command[1]: .2f} wz={command[2]: .2f} "
        f"|xy|={np.linalg.norm(command[:2]):.2f}   "
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
