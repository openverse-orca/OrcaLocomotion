from __future__ import annotations

import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parent
TERRAIN_ROOT = ROOT.parent
TILE_SIZE = 2.0
TILE_SPACING = 2.4
HALF_TILE = TILE_SIZE * 0.5
GRID_ROWS = 5
GRID_COLS = 5
RESOLUTION = 0.05
FRICTION = "0.9 0.005 0.0001"


def fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def tile_center(row: int, col: int) -> tuple[float, float]:
    x = row * TILE_SPACING
    y = (col - (GRID_COLS - 1) * 0.5) * TILE_SPACING
    return x, y


def add_geom(
    worldbody: ET.Element,
    *,
    name: str,
    geom_type: str = "box",
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    material: str = "tile_mat",
    rgba: str | None = None,
    quat: tuple[float, float, float, float] | None = None,
    contype: str = "1",
    conaffinity: str = "1",
) -> ET.Element:
    attrs = {
        "name": name,
        "type": geom_type,
        "pos": fmt(pos),
        "size": fmt(size),
        "friction": FRICTION,
        "contype": contype,
        "conaffinity": conaffinity,
    }
    if material:
        attrs["material"] = material
    if rgba is not None:
        attrs["rgba"] = rgba
    if quat is not None:
        attrs["quat"] = fmt(quat)
    return ET.SubElement(worldbody, "geom", attrs)


def add_tile_base(worldbody: ET.Element, row: int, col: int, *, rgba: str) -> None:
    x, y = tile_center(row, col)
    add_geom(
        worldbody,
        name=f"terrain_r{row}_c{col}_base",
        pos=(x, y, 0.01),
        size=(HALF_TILE, HALF_TILE, 0.01),
        rgba=rgba,
    )


def add_block(
    worldbody: ET.Element,
    name: str,
    x: float,
    y: float,
    sx: float,
    sy: float,
    height: float,
    *,
    rgba: str,
) -> None:
    add_geom(
        worldbody,
        name=name,
        pos=(x, y, 0.02 + height * 0.5),
        size=(sx, sy, height * 0.5),
        rgba=rgba,
    )


def add_low_blocks(worldbody: ET.Element, row: int, col: int, *, difficulty: float) -> None:
    x0, y0 = tile_center(row, col)
    heights = np.array(
        [
            [0.03, 0.00, 0.05],
            [0.00, 0.04, 0.02],
            [0.05, 0.02, 0.00],
        ]
    ) * difficulty
    for i in range(3):
        for j in range(3):
            height = float(heights[i, j])
            if height <= 0.005:
                continue
            x = x0 + (i - 1) * 0.55
            y = y0 + (j - 1) * 0.55
            add_block(
                worldbody,
                f"terrain_r{row}_c{col}_low_block_{i}_{j}",
                x,
                y,
                0.22,
                0.22,
                height,
                rgba="0.46 0.48 0.40 1",
            )


def add_rough_blocks(worldbody: ET.Element, row: int, col: int, *, difficulty: float) -> None:
    x0, y0 = tile_center(row, col)
    pattern = np.array(
        [
            [0.02, 0.09, 0.00, 0.06],
            [0.07, 0.00, 0.11, 0.03],
            [0.00, 0.12, 0.04, 0.08],
            [0.06, 0.03, 0.10, 0.00],
        ]
    ) * difficulty
    for i in range(4):
        for j in range(4):
            height = float(pattern[i, j])
            if height <= 0.005:
                continue
            x = x0 - 0.72 + i * 0.48
            y = y0 - 0.72 + j * 0.48
            add_block(
                worldbody,
                f"terrain_r{row}_c{col}_rough_block_{i}_{j}",
                x,
                y,
                0.16,
                0.16,
                height,
                rgba="0.39 0.45 0.36 1",
            )


def add_stairs(worldbody: ET.Element, row: int, col: int, *, step_height: float) -> None:
    x0, y0 = tile_center(row, col)
    for i in range(5):
        height = step_height * (i + 1)
        x = x0 - 0.8 + i * 0.4
        add_block(
            worldbody,
            f"terrain_r{row}_c{col}_stair_{i}",
            x,
            y0,
            0.19,
            0.9,
            height,
            rgba="0.42 0.42 0.36 1",
        )


def add_ramp(worldbody: ET.Element, row: int, col: int, *, angle_deg: float) -> None:
    x0, y0 = tile_center(row, col)
    angle = math.radians(angle_deg)
    quat = (math.cos(angle * 0.5), 0.0, math.sin(angle * 0.5), 0.0)
    add_geom(
        worldbody,
        name=f"terrain_r{row}_c{col}_ramp",
        pos=(x0, y0, 0.08),
        size=(0.92, 0.88, 0.05),
        quat=quat,
        rgba="0.44 0.43 0.35 1",
    )


def add_stepping_stones(worldbody: ET.Element, row: int, col: int, *, height: float) -> None:
    x0, y0 = tile_center(row, col)
    stones = [(-0.55, -0.45), (-0.15, 0.35), (0.35, -0.20), (0.72, 0.52)]
    for i, (dx, dy) in enumerate(stones):
        add_block(
            worldbody,
            f"terrain_r{row}_c{col}_stone_{i}",
            x0 + dx,
            y0 + dy,
            0.26,
            0.22,
            height + 0.02 * (i % 2),
            rgba="0.37 0.40 0.34 1",
        )


def add_wave_bars(worldbody: ET.Element, row: int, col: int, *, height: float) -> None:
    x0, y0 = tile_center(row, col)
    for i in range(5):
        bar_height = height * (0.55 + 0.45 * ((i % 2) == 0))
        x = x0 - 0.8 + i * 0.4
        add_block(
            worldbody,
            f"terrain_r{row}_c{col}_wave_{i}",
            x,
            y0,
            0.05,
            0.9,
            bar_height,
            rgba="0.40 0.46 0.38 1",
        )


def add_gap_edges(worldbody: ET.Element, row: int, col: int, *, height: float) -> None:
    x0, y0 = tile_center(row, col)
    add_block(worldbody, f"terrain_r{row}_c{col}_gap_left", x0 - 0.55, y0, 0.28, 0.9, height, rgba="0.40 0.39 0.34 1")
    add_block(worldbody, f"terrain_r{row}_c{col}_gap_right", x0 + 0.55, y0, 0.28, 0.9, height, rgba="0.40 0.39 0.34 1")


def build_xml() -> ET.ElementTree:
    root = ET.Element("mujoco", {"model": "mjlab_rough_5x5_terrain"})
    option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")
    option.set("timestep", "0.005")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", {"name": "tile_mat", "rgba": "0.45 0.47 0.42 1"})
    ET.SubElement(asset, "material", {"name": "floor_mat", "rgba": "0.30 0.32 0.31 1"})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "terrain_light",
            "pos": "3 -4 8",
            "dir": "-0.4 0.4 -1",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.2 0.2 0.2",
        },
    )
    add_geom(
        worldbody,
        name="terrain_floor",
        geom_type="plane",
        pos=(4.8, 0.0, 0.0),
        size=(9.0, 8.0, 0.02),
        material="floor_mat",
        rgba=None,
    )

    base_colors = [
        "0.48 0.52 0.46 1",
        "0.46 0.50 0.44 1",
        "0.44 0.48 0.42 1",
        "0.42 0.46 0.40 1",
        "0.40 0.44 0.38 1",
    ]
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            add_tile_base(worldbody, row, col, rgba=base_colors[row])

    for col in range(GRID_COLS):
        add_low_blocks(worldbody, 1, col, difficulty=0.75 + 0.12 * col)

    add_low_blocks(worldbody, 2, 0, difficulty=1.15)
    add_rough_blocks(worldbody, 2, 1, difficulty=0.85)
    add_wave_bars(worldbody, 2, 2, height=0.08)
    add_ramp(worldbody, 2, 3, angle_deg=8.0)
    add_gap_edges(worldbody, 2, 4, height=0.10)

    add_rough_blocks(worldbody, 3, 0, difficulty=1.15)
    add_stairs(worldbody, 3, 1, step_height=0.035)
    add_ramp(worldbody, 3, 2, angle_deg=-10.0)
    add_wave_bars(worldbody, 3, 3, height=0.13)
    add_stepping_stones(worldbody, 3, 4, height=0.12)

    add_stairs(worldbody, 4, 0, step_height=0.05)
    add_rough_blocks(worldbody, 4, 1, difficulty=1.45)
    add_stepping_stones(worldbody, 4, 2, height=0.16)
    add_gap_edges(worldbody, 4, 3, height=0.16)
    add_wave_bars(worldbody, 4, 4, height=0.18)

    return ET.ElementTree(root)


def add_rect_height(height: np.ndarray, xs: np.ndarray, ys: np.ndarray, x: float, y: float, sx: float, sy: float, z: float) -> None:
    mask_x = np.abs(xs - x) <= sx
    mask_y = np.abs(ys - y) <= sy
    height[np.ix_(mask_y, mask_x)] = np.maximum(height[np.ix_(mask_y, mask_x)], z)


def build_height_field() -> dict[str, np.ndarray]:
    min_x = -HALF_TILE - 0.4
    max_x = (GRID_ROWS - 1) * TILE_SPACING + HALF_TILE + 0.4
    min_y = -((GRID_COLS - 1) * 0.5) * TILE_SPACING - HALF_TILE - 0.4
    max_y = ((GRID_COLS - 1) * 0.5) * TILE_SPACING + HALF_TILE + 0.4
    xs = np.arange(min_x, max_x + RESOLUTION * 0.5, RESOLUTION, dtype=np.float64)
    ys = np.arange(min_y, max_y + RESOLUTION * 0.5, RESOLUTION, dtype=np.float64)
    height = np.zeros((ys.size, xs.size), dtype=np.float64)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x0, y0 = tile_center(row, col)
            add_rect_height(height, xs, ys, x0, y0, HALF_TILE, HALF_TILE, 0.02)

    # Conservative approximation for height scans. Tilted ramps are represented
    # by their peak height so the policy sees an obstacle instead of empty air.
    for row in range(1, GRID_ROWS):
        for col in range(GRID_COLS):
            x0, y0 = tile_center(row, col)
            difficulty = 0.04 + 0.035 * row + 0.01 * col
            add_rect_height(height, xs, ys, x0, y0, 0.9, 0.9, difficulty)

    np.clip(height, 0.0, 0.24, out=height)
    return {
        "x": xs,
        "y": ys,
        "height": height,
        "resolution": np.asarray(RESOLUTION, dtype=np.float64),
        "min_height": np.asarray(float(height.min()), dtype=np.float64),
        "max_height": np.asarray(float(height.max()), dtype=np.float64),
    }


def write_readme() -> None:
    (ROOT / "README_ORCA_RL.md").write_text(
        """# Mjlab Rough 5x5 Terrain XML

A standalone terrain-only MuJoCo XML package for OrcaLab XML import and
`orca_rl.run_play --local-terrain-map` testing.

Layout:

- 5 rows x 5 columns, matching the fixed mjlab rough play layout.
- Rows increase in difficulty along +X.
- Columns mix low blocks, rough blocks, wave bars, ramps, gaps, stairs, and stepping stones.
- The first row is intentionally mild so a single Go2 can spawn and walk into harder tiles.

Files:

- `terrain.xml`: primitive-only MuJoCo terrain. It uses only `plane`, `box`, and `light`.
- `terrain_height_field.npz`: approximate height scan alignment for local MuJoCo play.
- `generate_terrain.py`: deterministic generator used to rebuild both files.

Upload package:

`assets/terrain/MjlabRough5x5Xml.zip`
""",
        encoding="utf-8",
    )


def zip_stored(output_path: Path, entries: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname)


def write_xml_package() -> Path:
    package_path = TERRAIN_ROOT / "MjlabRough5x5Xml.zip"
    zip_stored(
        package_path,
        [
            (ROOT / "README_ORCA_RL.md", "mjlab_rough_5x5_xml/README_ORCA_RL.md"),
            (ROOT / "terrain.xml", "mjlab_rough_5x5_xml/terrain.xml"),
            (ROOT / "terrain_height_field.npz", "mjlab_rough_5x5_xml/terrain_height_field.npz"),
        ],
    )
    return package_path


def main() -> None:
    tree = build_xml()
    ET.indent(tree, space="  ")
    xml_path = ROOT / "terrain.xml"
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    np.savez(ROOT / "terrain_height_field.npz", **build_height_field())
    write_readme()
    write_xml_package()


if __name__ == "__main__":
    main()
