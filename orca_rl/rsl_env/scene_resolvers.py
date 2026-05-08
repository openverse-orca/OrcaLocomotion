from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_scene_binding(resolver: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(resolver):
        return resolver
    if resolver == "g1":
        from .scene_binding import resolve_g1_scene_binding

        return resolve_g1_scene_binding
    if resolver == "go2":
        from .scene_binding import resolve_go2_scene_binding

        return resolve_go2_scene_binding
    if "." in resolver:
        import importlib

        module_name, _, attr_name = resolver.rpartition(".")
        resolved = getattr(importlib.import_module(module_name), attr_name)
        if callable(resolved):
            return resolved
    raise ValueError(
        f"Unknown scene binding resolver: {resolver!r}. "
        "Expected 'g1', 'go2', or import path."
    )
