from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_inference_runner(env, train_cfg: dict[str, Any], checkpoint: str, log_dir: str | None, device: str):
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    runner.load(checkpoint, map_location=device)
    return runner, runner.get_inference_policy(device=device)


def export_policy(runner, export_dir: str | Path, *, onnx: bool = True, jit: bool = True) -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    if jit:
        runner.export_policy_to_jit(str(export_dir), filename="policy.pt")
    if onnx:
        runner.export_policy_to_onnx(str(export_dir), filename="policy.onnx")

