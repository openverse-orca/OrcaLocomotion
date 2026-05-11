from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from orca_rl.terrains import HeightField, generate_height_field

from .math_utils import quat_wxyz_to_rotmat


@dataclass(frozen=True)
class TerrainScanConfig:
    enabled: bool = False
    size: tuple[float, float] = (1.6, 1.0)
    resolution: float = 0.10
    scale: float = 1.0

    @property
    def num_rays(self) -> int:
        x_count = int(round(self.size[0] / self.resolution)) + 1
        y_count = int(round(self.size[1] / self.resolution)) + 1
        return x_count * y_count


class TerrainRuntime:
    """Generated terrain backing for height scans and future OrcaLab mesh import."""

    def __init__(
        self,
        *,
        height_field: HeightField,
        scan_cfg: TerrainScanConfig,
        physics_enabled: bool,
        export_path: str | None,
        terrain_cfg: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.height_field = height_field
        self.scan_cfg = scan_cfg
        self.physics_enabled = physics_enabled
        self.export_path = export_path
        self.terrain_cfg = terrain_cfg or {"terrain_type": "plane"}
        self.rng = rng
        self.exported_mesh_path: Path | None = None

    @classmethod
    def from_task_cfg(
        cls,
        cfg: dict[str, Any],
        rng: np.random.Generator,
        *,
        terrain_cfg: dict[str, Any] | None = None,
    ) -> "TerrainRuntime":
        terrain_cfg = terrain_cfg or cfg.get("terrain") or {"terrain_type": "plane"}
        sensors = cfg.get("sensors", {})
        height_field = generate_height_field(terrain_cfg, rng)
        scan_cfg = _scan_config(sensors.get("terrain_scan"), cfg.get("observations", {}))
        runtime = cls(
            height_field=height_field,
            scan_cfg=scan_cfg,
            physics_enabled=bool(terrain_cfg.get("physics_enabled", False)),
            export_path=terrain_cfg.get("export_path"),
            terrain_cfg=terrain_cfg,
            rng=rng,
        )
        if runtime.export_path:
            runtime.exported_mesh_path = runtime.export_mesh(runtime.export_path)
        return runtime

    def height_at(self, x: float, y: float) -> float:
        return self.height_field.height_at(x, y)

    def resample(self) -> None:
        if self.rng is None:
            return
        if self.physics_enabled:
            return
        if self.terrain_cfg.get("terrain_type", "plane") == "plane":
            return
        self.height_field = generate_height_field(self.terrain_cfg, self.rng)
        if self.export_path:
            self.exported_mesh_path = self.export_mesh(self.export_path)

    def scan(self, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        if not self.scan_cfg.enabled:
            return np.zeros(0, dtype=np.float64)
        rot = quat_wxyz_to_rotmat(base_quat)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        terrain_heights = self.height_field.scan(
            np.asarray(base_pos[:2], dtype=np.float64),
            yaw,
            size=self.scan_cfg.size,
            resolution=self.scan_cfg.resolution,
        )
        relative_heights = terrain_heights - float(base_pos[2])
        return (relative_heights * self.scan_cfg.scale).astype(np.float64)

    def export_mesh(self, path: str | Path) -> Path:
        return self.height_field.to_mesh().write_obj(path)


def _scan_config(sensor_cfg: Any, obs_cfg: dict[str, Any]) -> TerrainScanConfig:
    if not isinstance(sensor_cfg, dict):
        return TerrainScanConfig(enabled=False)
    pattern = sensor_cfg.get("pattern") or {}
    size = tuple(float(value) for value in pattern.get("size", (1.6, 1.0)))
    resolution = float(pattern.get("resolution", 0.10))
    return TerrainScanConfig(
        enabled=True,
        size=(size[0], size[1]),
        resolution=resolution,
        scale=float(obs_cfg.get("height_scan_scale", 1.0)),
    )
