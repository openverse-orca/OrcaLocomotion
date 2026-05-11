from __future__ import annotations

from typing import Any


def make_locomotion_vec_env(
    task_cfg: dict[str, Any],
    *,
    device: str = "cpu",
    render_mode: str | None = None,
    headless: bool | None = None,
):
    from .vecenv import OrcaRslRlVecEnv

    return OrcaRslRlVecEnv(task_cfg, device=device, render_mode=render_mode, headless=headless)
