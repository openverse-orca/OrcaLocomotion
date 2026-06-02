from __future__ import annotations

from typing import Any


NO_RENDER_MODES = {"", "none", "headless", "no-render", "no_render", "off", "disabled"}


def is_headless_render_mode(render_mode: Any) -> bool:
    if render_mode is None:
        return True
    return str(render_mode).strip().lower() in NO_RENDER_MODES


def normalize_headless(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on", "headless", "none", "no-render", "no_render"}:
            return True
        if text in {"0", "false", "no", "n", "off", "render", "human"}:
            return False
    return bool(value)


def resolve_rendering(
    task_cfg: dict[str, Any],
    *,
    render_mode: str | None = None,
    headless: bool | None = None,
) -> tuple[bool, str]:
    sim_cfg = task_cfg.get("sim", {})
    resolved_headless = normalize_headless(headless)
    if resolved_headless is None:
        resolved_headless = normalize_headless(sim_cfg.get("headless"))

    mode = render_mode if render_mode is not None else sim_cfg.get("render_mode")
    if resolved_headless is None:
        resolved_headless = is_headless_render_mode(mode)

    if resolved_headless:
        return True, "none"
    if mode is None or is_headless_render_mode(mode):
        return False, "human"
    return False, str(mode)
