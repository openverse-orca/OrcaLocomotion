from __future__ import annotations

import torch


def configure_torch_backends(*, allow_tf32: bool = True, deterministic: bool = False) -> None:
    """Configure PyTorch for Orca training throughput."""

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
