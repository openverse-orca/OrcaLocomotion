from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import inspect
from typing import Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    factory: Callable
    description: str = ""


_TASKS: dict[str, TaskSpec] = {}
_BUILTINS_LOADED = False


def _load_builtin_tasks() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from . import g1_velocity as _g1_velocity  # noqa: F401


def register_task(name: str, factory: Callable, description: str = "") -> None:
    _load_builtin_tasks()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Task name must be a non-empty string")
    if not callable(factory):
        raise TypeError("Task factory must be callable")
    if name in _TASKS:
        raise ValueError(f"Task {name!r} is already registered")
    _TASKS[name] = TaskSpec(name, factory, description)


def make_task(name: str, **kwargs):
    _load_builtin_tasks()
    try:
        factory = _TASKS[name].factory
    except KeyError as exc:
        suggestions = get_close_matches(name, _TASKS, n=3)
        hint = f"; did you mean: {', '.join(suggestions)}" if suggestions else ""
        raise KeyError(
            f"Unknown Orca task {name!r}{hint}; available: {', '.join(_TASKS)}"
        ) from exc
    signature = inspect.signature(factory)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return factory(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return factory(**supported)


def list_tasks() -> tuple[TaskSpec, ...]:
    _load_builtin_tasks()
    return tuple(_TASKS.values())
