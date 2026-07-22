"""Task registry with lazy built-in task discovery."""

from .registry import list_tasks, make_task, register_task

__all__ = ["list_tasks", "make_task", "register_task"]
