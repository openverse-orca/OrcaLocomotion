"""Standalone RSL-RL bridge for OrcaLab / OrcaGym."""

from __future__ import annotations

__all__ = [
    "OrcaRslRlVecEnv",
    "get_task_spec",
    "list_tasks",
    "load_task_and_train_cfg",
    "make_locomotion_vec_env",
    "print_runtime_summary",
]


def __getattr__(name: str):
    if name == "OrcaRslRlVecEnv":
        from .rsl_env import OrcaRslRlVecEnv

        return OrcaRslRlVecEnv
    if name == "make_locomotion_vec_env":
        from .rsl_env import make_locomotion_vec_env

        return make_locomotion_vec_env
    if name == "load_task_and_train_cfg":
        from .utils import load_task_and_train_cfg

        return load_task_and_train_cfg
    if name == "list_tasks":
        from .registry import list_tasks

        return list_tasks
    if name == "get_task_spec":
        from .registry import get_task_spec

        return get_task_spec
    if name == "print_runtime_summary":
        from .diagnostics import print_runtime_summary

        return print_runtime_summary
    raise AttributeError(name)
