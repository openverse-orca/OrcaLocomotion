from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_factory: str
    runner_factory: str
    description: str = ""


_TASKS: dict[str, TaskSpec] = {}


def register_task(name: str, env_factory: str, runner_factory: str, description: str = "") -> None:
    key = _normalize_task_name(name)
    _TASKS[key] = TaskSpec(
        name=name,
        env_factory=env_factory,
        runner_factory=runner_factory,
        description=description,
    )


def get_task_spec(name: str) -> TaskSpec:
    _ensure_builtin_tasks()
    key = _normalize_task_name(name)
    if key not in _TASKS:
        known = ", ".join(spec.name for spec in list_tasks())
        raise KeyError(f"Unknown registered task {name!r}. Known tasks: {known}")
    return _TASKS[key]


def list_tasks() -> list[TaskSpec]:
    _ensure_builtin_tasks()
    return sorted(_TASKS.values(), key=lambda spec: spec.name.lower())


def is_registered_task(name: str | None) -> bool:
    if not name:
        return False
    _ensure_builtin_tasks()
    return _normalize_task_name(name) in _TASKS


def load_registered_task(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = get_task_spec(name)
    task_cfg = _config_to_dict(_load_factory(spec.env_factory)(), spec.env_factory)
    raw_runner_cfg = _config_to_dict(_load_factory(spec.runner_factory)(), spec.runner_factory)
    runner_cfg = raw_runner_cfg.get("runner", raw_runner_cfg)
    return task_cfg, runner_cfg


def _ensure_builtin_tasks() -> None:
    if _TASKS:
        return
    register_task(
        "Unitree-G1-Flat",
        "orca_rl.tasks.velocity.config.g1.env_cfgs:unitree_g1_flat_env_cfg",
        "orca_rl.tasks.velocity.config.g1.rl_cfg:unitree_g1_ppo_runner_cfg",
        "Unitree G1 velocity tracking on flat terrain.",
    )
    register_task(
        "Unitree-G1-Rough",
        "orca_rl.tasks.velocity.config.g1.env_cfgs:unitree_g1_rough_env_cfg",
        "orca_rl.tasks.velocity.config.g1.rl_cfg:unitree_g1_rough_ppo_runner_cfg",
        "Unitree G1 velocity tracking with generated physical rough terrain.",
    )
    register_task(
        "Unitree-GO2-Flat",
        "orca_rl.tasks.velocity.config.go2.env_cfgs:unitree_go2_flat_env_cfg",
        "orca_rl.tasks.velocity.config.go2.rl_cfg:unitree_go2_ppo_runner_cfg",
        "Unitree Go2 velocity tracking on flat terrain.",
    )
    register_task(
        "Unitree-GO2-Rough",
        "orca_rl.tasks.velocity.config.go2.env_cfgs:unitree_go2_rough_env_cfg",
        "orca_rl.tasks.velocity.config.go2.rl_cfg:unitree_go2_rough_ppo_runner_cfg",
        "Unitree Go2 velocity tracking with generated rough-terrain metadata.",
    )


def _load_factory(path: str) -> Callable[[], Any]:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Expected import path in module:function form, got: {path}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ValueError(f"Registered config factory is not callable: {path}")
    return factory


def _config_to_dict(config: Any, source: str) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        raise ValueError(f"Registered config factory must return a dict-like object: {source}")
    return config


def _normalize_task_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
