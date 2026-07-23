from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "checkpoints/heft/G1_PMG/policy.onnx"
MOTION_DIRS = (ROOT / "assets/heft/recorded_commands", ROOT / "assets/heft/motions")
EXPECTED_RECORDED = {
    "heft_stand",
    "heft_forward",
    "heft_backward",
    "heft_left",
    "heft_right",
    "heft_yaw_left",
    "heft_yaw_right",
}


def main() -> None:
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 1729]:
        raise RuntimeError(f"Unexpected HEFT input ABI: {[(item.name, item.shape) for item in inputs]}")
    action_outputs = [item for item in session.get_outputs() if item.shape == [1, 29]]
    if not action_outputs:
        raise RuntimeError("HEFT ONNX has no 29-D action output.")
    output = session.run(
        [action_outputs[-1].name],
        {inputs[0].name: np.zeros((1, 1729), dtype=np.float32)},
    )[0]
    if output.shape != (1, 29) or not np.isfinite(output).all():
        raise RuntimeError("HEFT ONNX smoke inference returned an invalid action.")

    recorded = {path.stem for path in MOTION_DIRS[0].glob("*.npz")}
    if recorded != EXPECTED_RECORDED:
        raise RuntimeError(f"Recorded motion set mismatch: {sorted(recorded)}")
    for directory in MOTION_DIRS:
        for path in sorted(directory.glob("*.npz")):
            _check_motion(path)
    print("[HEFT smoke] ONNX 1729 -> 29 and all 50 Hz motions are valid.")


def _check_motion(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        fps = float(np.asarray(data["fps"]).reshape(()))
        joint_names = np.asarray(data["joint_names"]).reshape(-1)
        joint_pos = np.asarray(data["dof_pos"])
        root_pos = np.asarray(data["root_pos"])
        root_rot = np.asarray(data["root_rot"])
    if not np.isclose(fps, 50.0):
        raise RuntimeError(f"{path.name}: expected 50 Hz, got {fps:g}")
    if len(joint_names) < 29 or joint_pos.ndim != 2 or joint_pos.shape[1] < 29:
        raise RuntimeError(f"{path.name}: invalid joint layout {joint_pos.shape}")
    if root_pos.shape != (joint_pos.shape[0], 3) or root_rot.shape != (joint_pos.shape[0], 4):
        raise RuntimeError(f"{path.name}: root/joint frame count mismatch")
    if not all(np.isfinite(value).all() for value in (joint_pos, root_pos, root_rot)):
        raise RuntimeError(f"{path.name}: non-finite motion data")


if __name__ == "__main__":
    main()
