from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from orca_rl.terrains import generate_height_field
from orca_rl.utils import load_task_and_train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a generated Orca RL terrain mesh.")
    parser.add_argument("--config", required=True, help="Task cfg path, including optional file.py:factory selector.")
    parser.add_argument("--out", required=True, help="Output OBJ path.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    task_cfg, _train_cfg = load_task_and_train_cfg(args.config)
    seed = int(args.seed if args.seed is not None else task_cfg.get("seed", 1))
    height_field = generate_height_field(task_cfg.get("terrain"), np.random.default_rng(seed))
    output_path = height_field.to_mesh().write_obj(Path(args.out))
    print(f"exported terrain mesh: {output_path}")
    print(f"height_field_shape: {height_field.heights.shape}")
    print(f"resolution: {height_field.resolution}")
    print(f"origin_xy: {height_field.origin_xy}")


if __name__ == "__main__":
    main()
