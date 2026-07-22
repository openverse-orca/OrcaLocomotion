from __future__ import annotations

from collections.abc import Callable

import numpy as np


class WandbVideoRecorder:
    """Periodically logs a short evaluation rollout as W&B video.

    ``frame_source`` isolates rendering from the headless training hot path. It
    may use Orca offscreen rendering or a single OrcaLab play environment.
    """
    def __init__(self, frame_source: Callable[[], np.ndarray], *, fps: int = 30, key: str = "eval/video"):
        self.frame_source, self.fps, self.key = frame_source, fps, key
        self.frames: list[np.ndarray] = []

    def capture(self) -> None:
        frame = np.asarray(self.frame_source(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("video frame must be HxWx3 or HxWx4 uint8")
        self.frames.append(frame[..., :3])

    def log(self, step: int) -> None:
        if not self.frames:
            return
        import wandb
        video = np.stack(self.frames).transpose(0, 3, 1, 2)
        wandb.log({self.key: wandb.Video(video, fps=self.fps, format="mp4")}, step=step)
        self.frames.clear()
