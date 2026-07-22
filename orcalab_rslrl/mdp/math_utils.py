from __future__ import annotations

import math

import torch


def quat_apply_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame ``vec`` into the frame described by ``quat_wxyz``."""
    w, xyz = quat_wxyz[..., :1], quat_wxyz[..., 1:]
    if vec.dim() < quat_wxyz.dim():
        vec = vec.expand_as(xyz)
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion for a pure z rotation."""
    half = 0.5 * yaw
    quat = torch.zeros(*yaw.shape, 4, device=yaw.device, dtype=yaw.dtype)
    quat[..., 0] = torch.cos(half)
    quat[..., 3] = torch.sin(half)
    return quat


def yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
