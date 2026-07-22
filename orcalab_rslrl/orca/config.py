"""Validated public runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OrcaRuntimeConfig:
    """Configuration shared by programmatic training and playback."""

    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    play: bool = False
    asset: str | Path | None = None
    physics_timestep: float | None = None
    terrain: str = "flat"
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.physics_timestep is not None and self.physics_timestep <= 0:
            raise ValueError("physics_timestep must be positive")
        if not self.terrain:
            raise ValueError("terrain must be non-empty")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer")

    def task_kwargs(self) -> dict[str, object]:
        """Translate the public config into the task-factory contract."""

        return {
            "num_envs": self.num_envs,
            "device": self.device,
            "headless": self.headless,
            "play": self.play,
            "mjcf_path": self.asset,
            "physics_timestep": self.physics_timestep,
            "terrain_kind": self.terrain,
            "seed": self.seed,
        }
