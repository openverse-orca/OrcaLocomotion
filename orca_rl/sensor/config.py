from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContactMatchCfg:
    """Mjlab-style contact selector metadata."""

    mode: str
    pattern: str | tuple[str, ...]
    entity: str = "robot"
    exclude: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pattern": self.pattern,
            "entity": self.entity,
            "exclude": self.exclude,
        }


@dataclass(frozen=True)
class ContactSensorCfg:
    """Declarative contact sensor config used by velocity task metadata."""

    name: str
    primary: ContactMatchCfg
    secondary: ContactMatchCfg | None = None
    fields: tuple[str, ...] = ("found",)
    reduce: str = "none"
    num_slots: int = 1
    history_length: int = 1
    track_air_time: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict() if self.secondary is not None else None,
            "fields": self.fields,
            "reduce": self.reduce,
            "num_slots": self.num_slots,
            "history_length": self.history_length,
            "track_air_time": self.track_air_time,
        }


@dataclass(frozen=True)
class GridPatternCfg:
    """2-D ray grid pattern for terrain height scans."""

    resolution: float = 0.1
    size: tuple[float, float] = (1.6, 1.0)

    @property
    def num_rays(self) -> int:
        x_count = int(round(self.size[0] / self.resolution)) + 1
        y_count = int(round(self.size[1] / self.resolution)) + 1
        return x_count * y_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "grid",
            "resolution": self.resolution,
            "size": self.size,
            "num_rays": self.num_rays,
        }


@dataclass(frozen=True)
class RayCasterCfg:
    """Terrain ray-caster metadata.

    The current Orca runtime exposes this as config/diagnostic metadata. Real
    ray queries can be wired here once the OrcaLab terrain service is selected.
    """

    name: str = "terrain_scan"
    frame_name: str = "base"
    pattern: GridPatternCfg = field(default_factory=GridPatternCfg)
    attach_yaw_only: bool = True
    max_distance: float = 3.0
    debug_vis: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_name": self.frame_name,
            "pattern": self.pattern.to_dict(),
            "attach_yaw_only": self.attach_yaw_only,
            "max_distance": self.max_distance,
            "debug_vis": self.debug_vis,
        }
