from __future__ import annotations

from collections.abc import Callable
from typing import Any


def resolve_scene_binding(resolver: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(resolver):
        return resolver
    if resolver == "g1":
        from .scene_binding import resolve_g1_scene_binding

        return resolve_g1_scene_binding
    raise ValueError(f"HEFT supports only the 'g1' scene resolver, got {resolver!r}.")
