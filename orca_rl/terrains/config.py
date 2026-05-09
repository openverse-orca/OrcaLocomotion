from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SubTerrainCfg:
    name: str
    proportion: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "proportion": self.proportion,
            "params": self.params,
        }


@dataclass(frozen=True)
class TerrainGeneratorCfg:
    """Lightweight rough-terrain generator metadata inspired by mjlab/IsaacLab."""

    num_rows: int = 8
    num_cols: int = 8
    size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.10
    border_width: float = 2.0
    curriculum: bool = True
    sub_terrains: tuple[SubTerrainCfg, ...] = (
        SubTerrainCfg("random_uniform", 0.35, {"height_range": (-0.04, 0.04), "step": 0.01}),
        SubTerrainCfg("pyramid_stairs", 0.25, {"step_height_range": (0.02, 0.10), "step_width": 0.30}),
        SubTerrainCfg("discrete_obstacles", 0.20, {"height_range": (0.02, 0.10), "num_obstacles": 16}),
        SubTerrainCfg("wave", 0.20, {"amplitude_range": (0.02, 0.08), "num_waves": 3}),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "size": self.size,
            "horizontal_scale": self.horizontal_scale,
            "border_width": self.border_width,
            "curriculum": self.curriculum,
            "sub_terrains": tuple(item.to_dict() for item in self.sub_terrains),
        }


@dataclass(frozen=True)
class TerrainCfg:
    terrain_type: str = "plane"
    terrain_generator: TerrainGeneratorCfg | None = None
    static_friction: float = 0.8
    dynamic_friction: float = 0.8
    restitution: float = 0.0
    physics_enabled: bool = False
    export_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "terrain_type": self.terrain_type,
            "terrain_generator": self.terrain_generator.to_dict() if self.terrain_generator else None,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "physics_enabled": self.physics_enabled,
            "export_path": self.export_path,
        }
