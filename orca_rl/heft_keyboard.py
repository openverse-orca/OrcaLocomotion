from __future__ import annotations

from dataclasses import dataclass
import select
import sys
import termios
import threading
import tty
from typing import Protocol

import numpy as np


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
        enable_motion_selection: bool,
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
        enable_motion_selection: bool,
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
        from pynput import keyboard

        self._keyboard = keyboard
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
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
        special = {
            self._keyboard.Key.up: "up",
            self._keyboard.Key.down: "down",
            self._keyboard.Key.left: "left",
            self._keyboard.Key.right: "right",
            self._keyboard.Key.space: "space",
            self._keyboard.Key.esc: "esc",
            self._keyboard.Key.f1: "f1",
            self._keyboard.Key.f2: "f2",
            self._keyboard.Key.f3: "f3",
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


def make_keyboard_backend(
    backend: str,
    *,
    initial_command: np.ndarray,
    command_speed: float,
    yaw_speed: float,
    max_speed: float,
) -> KeyboardBackend:
    kwargs = {
        "command_speed": command_speed,
        "yaw_speed": yaw_speed,
        "max_speed": max_speed,
        "enable_motion_selection": True,
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
        print("[orca_rl.heft] Global keyboard unavailable; using terminal input.")
        return TerminalKeyboardBackend(initial_command, **kwargs)
    return GlobalKeyboardBackend(initial_command, **kwargs)


def clip_xy_command(command: np.ndarray, *, max_speed: float) -> np.ndarray:
    updated = _as_command(command)
    norm = float(np.linalg.norm(updated[:2]))
    limit = max(0.0, float(max_speed))
    if norm > limit > 0.0:
        updated[:2] *= limit / norm
    elif limit == 0.0:
        updated[:2] = 0.0
    return updated


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
    return clip_xy_command(updated, max_speed=max_speed)


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
    return clip_xy_command(command, max_speed=max_speed)


def _as_command(command: np.ndarray) -> np.ndarray:
    return np.asarray(command, dtype=np.float64).reshape(3).copy()


def _numpad_digit(vk) -> str | None:
    try:
        value = int(vk)
    except (TypeError, ValueError):
        return None
    if 96 <= value <= 105:
        return str(value - 96)
    if 65456 <= value <= 65465:
        return str(value - 65456)
    return None


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
