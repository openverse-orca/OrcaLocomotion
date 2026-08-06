"""Full-skeleton PICO retargeting with standing-only lower-body output.

This module consumes the same ``xrobotoolkit_sdk`` callback snapshots as the
reference teleop stack.  It validates a full PICO skeleton and uses its
shoulder/elbow/wrist chains for IK, while the emitted HEFT reference keeps
every non-arm joint at its startup standing value and replaces only the two
7-DoF G1 arms.  Dex3-1 remains a separate passive end effector; it is
deliberately not part of this IK.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np


# This is the XRoboToolkit body-array layout used by the reference HEFT branch.
XR_BODY_JOINT_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)
UPPER_BODY_NAMES = (
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
)
UPPER_BODY_INDICES = np.asarray([XR_BODY_JOINT_NAMES.index(name) for name in UPPER_BODY_NAMES], dtype=np.int64)
_PICO_TO_MUJOCO = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), dtype=np.float64)
# The XRobot-to-G1 orientation offsets from GMR's ``xrobot_to_g1.json``.
# XRoboToolkit reports link frames, not visual bone frames; drawing those raw
# frames is particularly misleading at the wrists.  The offsets below are
# display-only and match the target frames used by the TWIST2/GMR pipeline.
_GMR_XROBOT_VISUAL_OFFSETS_WXYZ = {
    "Left_Shoulder": (0.7071067811865475, 0.0, 0.7071067811865475, 0.0),
    "Left_Elbow": (0.0, 0.0, 1.0, 0.0),
    "Left_Wrist": (0.0, 0.0, 1.0, 0.0),
    "Right_Shoulder": (0.0, 0.7071067811865475, 0.0, -0.7071067811865475),
    "Right_Elbow": (0.0, 1.0, 0.0, 0.0),
    "Right_Wrist": (0.0, 1.0, 0.0, 0.0),
}

# Keep this lightweight module importable for raw PICO-stream diagnostics even
# when the full MuJoCo/ONNX HEFT runtime is not installed.
HEFT_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
_ARM_JOINT_NAMES = tuple(name for name in HEFT_JOINT_NAMES if "shoulder" in name or "elbow" in name or "wrist" in name)
_ARM_JOINT_INDICES = np.asarray([HEFT_JOINT_NAMES.index(name) for name in _ARM_JOINT_NAMES], dtype=np.int64)
_FIXED_STANDING_JOINT_INDICES = np.asarray(
    [index for index, name in enumerate(HEFT_JOINT_NAMES) if name not in _ARM_JOINT_NAMES],
    dtype=np.int64,
)
_ARM_TARGETS = (
    ("left_elbow_link", "Left_Elbow", "Left_Shoulder", 0.45, False),
    ("left_wrist_yaw_link", "Left_Wrist", "Left_Shoulder", 1.00, True),
    ("right_elbow_link", "Right_Elbow", "Right_Shoulder", 0.45, False),
    ("right_wrist_yaw_link", "Right_Wrist", "Right_Shoulder", 1.00, True),
)
_ROBOT_SHOULDER_LINKS = {
    "Left": "left_shoulder_yaw_link",
    "Right": "right_shoulder_yaw_link",
}


@dataclass(frozen=True)
class PicoUpperBodyFrame:
    """A validated full PICO skeleton frame in the XRoboToolkit pose layout."""

    sequence: int
    timestamp_ns: int
    poses: np.ndarray  # 24 rows in XR_BODY_JOINT_NAMES order: xyz + xyzw.


def load_xrobotoolkit_sdk() -> ModuleType:
    """Load the same callback-capable SDK used by ``unitree_sdk_for_demo``."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise ImportError(
            "Missing xrobotoolkit_sdk. Install the XRoboToolkit PC-Service-Pybind SDK used by "
            "BenHuHuan/unitree_sdk_for_demo, then rerun this player."
        ) from exc

    required = ("init", "register_frame_callback", "clear_frame_callback", "has_frame_callback")
    missing = tuple(name for name in required if not hasattr(xrt, name))
    if missing:
        raise ImportError(
            "xrobotoolkit_sdk is not the callback-enabled XRoboToolkit build; missing "
            + ", ".join(missing)
        )
    return xrt


def parse_pico_upper_body_snapshot(snapshot: Any, *, sequence: int) -> PicoUpperBodyFrame | None:
    """Parse a PICO snapshot without requiring a pelvis or lower-body tracker.

    All 24 XRoboToolkit skeleton rows are required.  The player uses the
    shoulder/elbow/wrist chain for arm-only retargeting and keeps every other
    G1 reference joint at the startup standing pose.
    """
    frame, _reason = _parse_pico_upper_body_snapshot(snapshot, sequence=sequence)
    return frame


def _parse_pico_upper_body_snapshot(snapshot: Any, *, sequence: int) -> tuple[PicoUpperBodyFrame | None, str]:
    if not isinstance(snapshot, dict):
        return None, "snapshot is not a dict"

    body = snapshot.get("body", snapshot.get("upper_body", {}))
    if not isinstance(body, dict):
        return None, f"body is not a dict (top-level keys={sorted(snapshot)})"
    try:
        all_poses = np.asarray(body.get("poses"), dtype=np.float32)
    except (TypeError, ValueError):
        return None, f"body.poses is not numeric (body keys={sorted(body)})"
    if all_poses.ndim != 2 or all_poses.shape[1] < 7 or all_poses.shape[0] < len(XR_BODY_JOINT_NAMES):
        return None, f"body.poses shape {all_poses.shape}; need at least (24, 7)"

    poses = np.ascontiguousarray(all_poses[: len(XR_BODY_JOINT_NAMES), :7], dtype=np.float32)
    if not np.all(np.isfinite(poses)):
        return None, "full skeleton poses contain non-finite values"
    quat_norm = np.linalg.norm(poses[:, 3:7], axis=1)
    if np.any(quat_norm < 1e-5):
        return None, (
            "full skeleton contains a zero quaternion; "
            f"body.available={body.get('available')}, quat_norms={np.array2string(quat_norm, precision=3)}"
        )

    raw_timestamp = body.get("timestamp_ns", snapshot.get("timestamp_ns", time.monotonic_ns()))
    try:
        timestamp_ns = int(raw_timestamp)
    except (TypeError, ValueError):
        timestamp_ns = time.monotonic_ns()
    if timestamp_ns <= 0:
        timestamp_ns = time.monotonic_ns()
    return PicoUpperBodyFrame(sequence=int(sequence), timestamp_ns=timestamp_ns, poses=poses), ""


class PicoUpperBodyStream:
    """Thread-safe latest-frame slot for the XRoboToolkit callback thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: PicoUpperBodyFrame | None = None
        self._sequence = 0
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.last_rejection_reason = ""
        self.rejection_reasons: dict[str, int] = {}

    def on_frame(self, snapshot: Any) -> None:
        with self._lock:
            self._sequence += 1
            frame, reason = _parse_pico_upper_body_snapshot(snapshot, sequence=self._sequence)
            if frame is None:
                self.rejected_frames += 1
                self.last_rejection_reason = reason
                self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
                return
            self._frame = frame
            self.accepted_frames += 1

    def latest_after(self, sequence: int) -> PicoUpperBodyFrame | None:
        with self._lock:
            if self._frame is None or self._frame.sequence <= int(sequence):
                return None
            return PicoUpperBodyFrame(
                sequence=self._frame.sequence,
                timestamp_ns=self._frame.timestamp_ns,
                poses=self._frame.poses.copy(),
            )


def _xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    x, y, z, w = quat
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return _xyzw_to_matrix(quat[[1, 2, 3, 0]])


def _pico_rotation_to_mujoco(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an XRoboToolkit world orientation into the MuJoCo world frame.

    XRoboToolkit sends position in its world frame and quaternion in ``xyzw``
    order.  The world conversion is a *left multiplication*, not a basis
    conjugation.  The latter leaves an identity PICO hand orientation as
    identity in Viser, which is why the hand axes appeared incorrect.
    """
    return _PICO_TO_MUJOCO @ _xyzw_to_matrix(quat_xyzw)


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector for a proper 3x3 rotation."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]),
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        # The exactly-180-degree case is rare in normal controller motion;
        # this stable diagonal fallback keeps the retargeter finite.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0))
        axis_norm = float(np.linalg.norm(axis))
    return axis * (angle / max(axis_norm, 1e-12))


def _relative_rotation(
    rotations: dict[str, np.ndarray],
    initial_rotations: dict[str, np.ndarray],
    *,
    child: str,
    anchor: str,
) -> np.ndarray:
    """Return the change in a PICO link frame relative to its shoulder.

    PICO reports world-space link orientations.  Applying their world-space
    change directly to a robot wrist makes a small headset/root-frame change
    look like a wrist rotation.  The arm solver is rooted at each shoulder, so
    use the corresponding shoulder-relative frame for the wrist objective.
    """
    current_relative = _task_frame_rotation(rotations[anchor], anchor).T @ _task_frame_rotation(rotations[child], child)
    initial_relative = _task_frame_rotation(initial_rotations[anchor], anchor).T @ _task_frame_rotation(
        initial_rotations[child], child
    )
    return current_relative @ initial_relative.T


def _task_frame_rotation(rotation: np.ndarray, body_name: str) -> np.ndarray:
    """Map an XRobot link frame into the task frame used by GMR's G1 config."""
    offset_wxyz = _GMR_XROBOT_VISUAL_OFFSETS_WXYZ.get(body_name)
    if offset_wxyz is None:
        return np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ _wxyz_to_matrix(offset_wxyz)


def _shoulder_local_position_delta(
    positions: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    initial_positions: dict[str, np.ndarray],
    initial_rotations: dict[str, np.ndarray],
    *,
    child: str,
    shoulder: str,
) -> np.ndarray:
    """Return a human-link displacement expressed in its calibrated shoulder frame.

    The raw PICO world frame is not the G1 body frame.  A direct subtraction
    in that world frame makes a reach rotate with the operator's heading.  The
    GMR reference pipeline converts upper-body segment offsets to their parent
    frame before mapping them to G1; this is the same calculation.
    """
    current_shoulder_rotation = _task_frame_rotation(rotations[shoulder], shoulder)
    initial_shoulder_rotation = _task_frame_rotation(initial_rotations[shoulder], shoulder)
    current_local = current_shoulder_rotation.T @ (positions[child] - positions[shoulder])
    initial_local = initial_shoulder_rotation.T @ (initial_positions[child] - initial_positions[shoulder])
    return current_local - initial_local


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Return a right-handed 3x3 rotation for the lightweight G1 diagnostic."""
    x, y, z = np.asarray(axis, dtype=np.float64)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        (
            (cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine),
            (y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine),
            (z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine),
        ),
        dtype=np.float64,
    )


def _quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo ``wxyz`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm((w, x, y, z)))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.asarray((w, x, y, z), dtype=np.float64) / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _model_name_id(model: Any, mujoco: Any, object_type: Any, suffix: str) -> int:
    """Resolve a G1 name even when OrcaLab has added a model prefix."""
    exact_id = int(mujoco.mj_name2id(model, object_type, suffix))
    if exact_id >= 0:
        return exact_id
    count = int(model.nbody) if object_type == mujoco.mjtObj.mjOBJ_BODY else int(model.njnt)
    matches = [
        object_id
        for object_id in range(count)
        if (name := mujoco.mj_id2name(model, object_type, object_id)) is not None
        and (name == suffix or name.endswith(suffix))
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely resolve {suffix!r}; matches={matches}")
    return int(matches[0])


def _model_reference_stick_figure(
    model: Any,
    mujoco: Any,
    joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build an arms-only reference skeleton from MuJoCo local body frames.

    ``MjData.xpos`` from an OrcaLab mirror can occasionally collapse all
    dynamic bodies to one coordinate.  The raw model tree still contains the
    correct G1 body offsets, rest orientations and hinge axes.  Forwarding
    that tree locally preserves the real shoulder mounting frames, so a G1
    forward reach is not drawn as a sideways or foot-like segment.
    """
    values = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if values.shape != (len(HEFT_JOINT_NAMES),):
        raise ValueError(f"HEFT reference must have shape (29,), got {values.shape}")
    try:
        qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
        if qpos.shape != (int(model.nq),):
            return None
        for name, value in zip(HEFT_JOINT_NAMES, values, strict=True):
            joint_id = _model_name_id(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos[int(model.jnt_qposadr[joint_id])] = value

        positions = np.zeros((int(model.nbody), 3), dtype=np.float64)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float64), (int(model.nbody), 3, 3)).copy()
        for body_id in range(1, int(model.nbody)):
            parent_id = int(model.body_parentid[body_id])
            parent_position = positions[parent_id]
            parent_rotation = rotations[parent_id]
            position = parent_position + parent_rotation @ np.asarray(model.body_pos[body_id], dtype=np.float64)
            rotation = parent_rotation @ _quaternion_to_matrix_wxyz(model.body_quat[body_id])
            joint_start = int(model.body_jntadr[body_id])
            joint_count = int(model.body_jntnum[body_id])
            for joint_id in range(joint_start, joint_start + joint_count):
                joint_type = int(model.jnt_type[joint_id])
                qpos_address = int(model.jnt_qposadr[joint_id])
                joint_position = np.asarray(model.jnt_pos[joint_id], dtype=np.float64)
                if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                    position = qpos[qpos_address : qpos_address + 3].copy()
                    rotation = _quaternion_to_matrix_wxyz(qpos[qpos_address + 3 : qpos_address + 7])
                elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                    joint_rotation = _quaternion_to_matrix_wxyz(qpos[qpos_address : qpos_address + 4])
                    pivot = position + rotation @ joint_position
                    position = pivot - rotation @ joint_rotation @ joint_position
                    rotation = rotation @ joint_rotation
                elif joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    position = position + rotation @ np.asarray(model.jnt_axis[joint_id], dtype=np.float64) * qpos[qpos_address]
                elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    joint_rotation = _axis_rotation(model.jnt_axis[joint_id], qpos[qpos_address])
                    pivot = position + rotation @ joint_position
                    position = pivot - rotation @ joint_rotation @ joint_position
                    rotation = rotation @ joint_rotation
            positions[body_id] = position
            rotations[body_id] = rotation

        def body_position(name: str) -> np.ndarray:
            return positions[_model_name_id(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, name)]

        # This is deliberately arms-only: the lower body is fixed for this
        # player and drawing its red points made the arm diagnostic look like
        # it was attached to a foot in tight/cropped browser camera views.
        ordered_points = (
            body_position("pelvis"),
            body_position("torso_link"),
            body_position("head_mimic"),
            body_position("left_shoulder_pitch_link"),
            body_position("left_elbow_link"),
            body_position("left_wrist_yaw_link"),
            body_position("left_hand_mimic"),
            body_position("right_shoulder_pitch_link"),
            body_position("right_elbow_link"),
            body_position("right_wrist_yaw_link"),
            body_position("right_hand_mimic"),
        )
        points = np.asarray(ordered_points, dtype=np.float32)
        segments = np.asarray(
            (
                (0, 1), (1, 2),
                (1, 3), (3, 4), (4, 5), (5, 6),
                (1, 7), (7, 8), (8, 9), (9, 10),
            ),
            dtype=np.int32,
        )
        if not np.all(np.isfinite(points)) or float(np.ptp(points, axis=0).max()) < 0.10:
            return None
        return points, segments
    except (IndexError, TypeError, ValueError):
        # A non-standard imported model can still use the simple diagnostic
        # below; the actual online reference and retargeting are unaffected.
        return None


def _g1_reference_stick_figure(joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a visible kinematic diagnostic directly from HEFT's 29 qpos.

    OrcaLab's local MuJoCo mirror can contain correct joints but degenerate
    visual body transforms (all link origins at one point).  This intentionally
    simple G1-shaped stick figure has fixed anthropometric dimensions and uses
    the exact 14 arm qpos supplied to HEFT.  It is an arms-only fallback when
    an imported model lacks the normal G1 body tree, not a replacement physics
    model.
    """
    joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if joints.shape != (len(HEFT_JOINT_NAMES),):
        raise ValueError(f"HEFT reference must have shape (29,), got {joints.shape}")
    values = dict(zip(HEFT_JOINT_NAMES, joints, strict=True))
    points: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []

    def add(point: np.ndarray) -> int:
        points.append(np.asarray(point, dtype=np.float64))
        return len(points) - 1

    pelvis = add(np.asarray((0.0, 0.0, 0.88)))
    waist = add(np.asarray((0.0, 0.0, 1.08)))
    chest = add(np.asarray((0.0, 0.0, 1.27)))
    neck = add(np.asarray((0.0, 0.0, 1.43)))
    head = add(np.asarray((0.0, 0.0, 1.62)))
    segments.extend(((pelvis, waist), (waist, chest), (chest, neck), (neck, head)))

    def arm(side: str, sign: float) -> None:
        shoulder = np.asarray((0.0, sign * 0.21, 1.34))
        shoulder_id = add(shoulder)
        segments.append((chest, shoulder_id))
        rotation = (
            _axis_rotation((0.0, 1.0, 0.0), values[f"{side}_shoulder_pitch_joint"])
            @ _axis_rotation((1.0, 0.0, 0.0), values[f"{side}_shoulder_roll_joint"])
            @ _axis_rotation((0.0, 0.0, 1.0), values[f"{side}_shoulder_yaw_joint"])
        )
        elbow = shoulder + rotation @ np.asarray((0.0, 0.0, -0.29))
        elbow_id = add(elbow)
        segments.append((shoulder_id, elbow_id))
        rotation = rotation @ _axis_rotation((0.0, 1.0, 0.0), values[f"{side}_elbow_joint"])
        wrist = elbow + rotation @ np.asarray((0.0, 0.0, -0.25))
        wrist_id = add(wrist)
        segments.append((elbow_id, wrist_id))
        rotation = (
            rotation
            @ _axis_rotation((1.0, 0.0, 0.0), values[f"{side}_wrist_roll_joint"])
            @ _axis_rotation((0.0, 1.0, 0.0), values[f"{side}_wrist_pitch_joint"])
            @ _axis_rotation((0.0, 0.0, 1.0), values[f"{side}_wrist_yaw_joint"])
        )
        hand = wrist + rotation @ np.asarray((0.0, 0.0, -0.11))
        hand_id = add(hand)
        segments.append((wrist_id, hand_id))

    arm("left", 1.0)
    arm("right", -1.0)
    return np.asarray(points, dtype=np.float32), np.asarray(segments, dtype=np.int32)


class G1SkeletonArmsRetargeter:
    """Read a full PICO frame and solve only the G1 shoulder/elbow/wrist chain."""

    def __init__(
        self,
        task: Any,
        *,
        position_scale: float = 0.8,
        iterations: int = 6,
        damping: float = 0.08,
        max_joint_step: float = 0.08,
        max_reference_step: float = 0.06,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("mujoco is required for PICO upper-body retargeting") from exc

        self.mujoco = mujoco
        self.model = getattr(getattr(task, "gym", None), "_mjModel", None)
        if self.model is None:
            raise ValueError("PICO retargeting requires the local MuJoCo model exposed by OrcaGym.")
        self.task = task
        self.position_scale = float(position_scale)
        self.iterations = max(1, int(iterations))
        self.damping = max(float(damping), 1e-6)
        self.max_joint_step = max(float(max_joint_step), 1e-4)
        self.max_reference_step = max(float(max_reference_step), 1e-4)
        self.data = mujoco.MjData(self.model)

        self._joint_ids = {name: self._resolve_name(mujoco.mjtObj.mjOBJ_JOINT, name) for name in HEFT_JOINT_NAMES}
        self._joint_qpos = {
            name: int(self.model.jnt_qposadr[joint_id]) for name, joint_id in self._joint_ids.items()
        }
        self._joint_dof = {
            name: int(self.model.jnt_dofadr[joint_id]) for name, joint_id in self._joint_ids.items()
        }
        # TWIST2/GMR solves with explicit limits.  This player has a stricter
        # contract: HEFT receives only arm references, so only those 14 joints
        # are variables in IK.  Letting legs or waist solve internally was an
        # unnecessary source of arm configurations outside HEFT's training
        # distribution.
        self._active_joint_names = _ARM_JOINT_NAMES
        self._active_qpos_ids = np.asarray([self._joint_qpos[name] for name in self._active_joint_names], dtype=np.int64)
        self._active_dof_ids = np.asarray([self._joint_dof[name] for name in self._active_joint_names], dtype=np.int64)
        self._body_ids = {
            suffix: self._resolve_name(mujoco.mjtObj.mjOBJ_BODY, suffix)
            for suffix in {item[0] for item in _ARM_TARGETS}
        }
        self._shoulder_body_ids = {
            side: self._resolve_name(mujoco.mjtObj.mjOBJ_BODY, body_name)
            for side, body_name in _ROBOT_SHOULDER_LINKS.items()
        }

        self._base_qpos: np.ndarray | None = None
        self._qpos: np.ndarray | None = None
        self._standing_joint_pos: np.ndarray | None = None
        self._last_reference: np.ndarray | None = None
        self._initial_human_pos: dict[str, np.ndarray] | None = None
        self._initial_human_rot: dict[str, np.ndarray] | None = None
        self._initial_robot_pos: dict[str, np.ndarray] = {}
        self._initial_robot_rot: dict[str, np.ndarray] = {}
        self._initial_robot_shoulder_pos: dict[str, np.ndarray] = {}
        self._initial_robot_shoulder_rot: dict[str, np.ndarray] = {}

    def _resolve_name(self, object_type: Any, suffix: str) -> int:
        direct = self.mujoco.mj_name2id(self.model, object_type, suffix)
        if direct >= 0:
            return int(direct)
        matches = []
        count = self.model.njnt if object_type == self.mujoco.mjtObj.mjOBJ_JOINT else self.model.nbody
        for idx in range(count):
            name = self.mujoco.mj_id2name(self.model, object_type, idx)
            # OrcaGym's G1 scene appends the USD revision to a small subset
            # of bodies, for example ``..._torso_link_rev_1_0``.  Accept that
            # suffix form as well as the ordinary instance-prefixed name.
            if name and (
                name == suffix
                or name.endswith("_" + suffix)
                or ("_" + suffix + "_") in name
            ):
                matches.append(idx)
        if len(matches) != 1:
            kind = "joint" if object_type == self.mujoco.mjtObj.mjOBJ_JOINT else "body"
            raise ValueError(f"Cannot uniquely resolve G1 {kind} {suffix!r}; matches={matches}")
        return int(matches[0])

    @property
    def calibrated(self) -> bool:
        return self._initial_human_pos is not None

    @property
    def qpos(self) -> np.ndarray:
        if self._qpos is None:
            raise RuntimeError("Retargeter is not initialized; call reset() first.")
        return self._qpos.copy()

    @property
    def standing_reference(self) -> np.ndarray:
        """The immutable non-PICO standing reference used by this session."""
        if self._standing_joint_pos is None:
            raise RuntimeError("Retargeter is not initialized; call reset() first.")
        return self._standing_joint_pos.copy()

    def qpos_for_reference(self, joint_positions: np.ndarray) -> np.ndarray:
        """Embed the *published* 29-D HEFT reference in this MuJoCo model.

        The internal IK configuration may be ahead of the slew-limited target.
        Rendering that configuration made the browser view disagree with the
        pose actually supplied to the HEFT policy.
        """
        if self._base_qpos is None:
            raise RuntimeError("Retargeter is not initialized; call reset() first.")
        joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if joints.shape != (len(HEFT_JOINT_NAMES),):
            raise ValueError(f"joint_positions must have shape (29,), got {joints.shape}")
        qpos = self._base_qpos.copy()
        for name, value in zip(HEFT_JOINT_NAMES, joints, strict=True):
            qpos[self._joint_qpos[name]] = value
        return qpos

    def clamp_arm_reference(self, joint_positions: np.ndarray) -> np.ndarray:
        """Clamp a 29-D online reference to the physical arm joint ranges."""
        reference = np.asarray(joint_positions, dtype=np.float32).reshape(-1).copy()
        if reference.shape != (len(HEFT_JOINT_NAMES),):
            raise ValueError(f"joint_positions must have shape (29,), got {reference.shape}")
        for name in self._active_joint_names:
            joint_id = self._joint_ids[name]
            if bool(self.model.jnt_limited[joint_id]):
                index = HEFT_JOINT_NAMES.index(name)
                lower, upper = self.model.jnt_range[joint_id]
                reference[index] = np.clip(reference[index], lower, upper)
        return reference

    def reset(self, initial_joint_pos: np.ndarray) -> None:
        """Capture the standing output pose and wait for a skeleton calibration frame."""
        joints = np.asarray(initial_joint_pos, dtype=np.float64).reshape(-1)
        if joints.shape != (len(HEFT_JOINT_NAMES),):
            raise ValueError(f"initial_joint_pos must have shape (29,), got {joints.shape}")
        qpos = np.asarray(self.task.data.qpos, dtype=np.float64).copy()
        for name, value in zip(HEFT_JOINT_NAMES, joints, strict=True):
            qpos[self._joint_qpos[name]] = value
        self._base_qpos = qpos.copy()
        self._qpos = qpos.copy()
        self._standing_joint_pos = joints.astype(np.float32, copy=True)
        self._last_reference = self._standing_joint_pos.copy()
        self._initial_human_pos = None
        self._initial_human_rot = None
        self.data.qpos[:] = self._qpos
        self.mujoco.mj_forward(self.model, self.data)
        self._initial_robot_pos = {
            name: self.data.xpos[body_id].copy() for name, body_id in self._body_ids.items()
        }
        self._initial_robot_rot = {
            name: self.data.xmat[body_id].reshape(3, 3).copy() for name, body_id in self._body_ids.items()
        }
        self._initial_robot_shoulder_pos = {
            side: self.data.xpos[body_id].copy() for side, body_id in self._shoulder_body_ids.items()
        }
        self._initial_robot_shoulder_rot = {
            side: self.data.xmat[body_id].reshape(3, 3).copy()
            for side, body_id in self._shoulder_body_ids.items()
        }

    def _human_arrays(self, frame: PicoUpperBodyFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        positions = np.asarray(frame.poses[:, :3], dtype=np.float64) @ _PICO_TO_MUJOCO.T
        rotations = [_pico_rotation_to_mujoco(quat) for quat in frame.poses[:, 3:7]]
        return (
            {name: positions[idx].copy() for idx, name in enumerate(XR_BODY_JOINT_NAMES)},
            {name: rotations[idx].copy() for idx, name in enumerate(XR_BODY_JOINT_NAMES)},
        )

    def retarget(self, frame: PicoUpperBodyFrame) -> np.ndarray:
        """Retarget one PICO frame and return a safe 29-D HEFT joint reference."""
        if (
            self._base_qpos is None
            or self._qpos is None
            or self._standing_joint_pos is None
            or self._last_reference is None
        ):
            raise RuntimeError("Retargeter is not initialized; call reset() first.")
        human_pos, human_rot = self._human_arrays(frame)
        if self._initial_human_pos is None or self._initial_human_rot is None:
            self._initial_human_pos = human_pos
            self._initial_human_rot = human_rot
            return self._last_reference.copy()

        # Use the complete PICO skeleton as the input format, but solve only
        # the shoulder-elbow-wrist chain.  Legs, pelvis and waist cannot leak
        # into the reference sent to HEFT.
        qpos = self._base_qpos.copy()
        qpos[self._active_qpos_ids] = self._qpos[self._active_qpos_ids]
        for _ in range(self.iterations):
            self.data.qpos[:] = qpos
            self.mujoco.mj_forward(self.model, self.data)
            jacobian_blocks: list[np.ndarray] = []
            error_blocks: list[np.ndarray] = []
            for robot_body, human_body, human_anchor, position_weight, use_orientation in _ARM_TARGETS:
                body_id = self._body_ids[robot_body]
                side = human_anchor.split("_", maxsplit=1)[0]
                human_delta_local = _shoulder_local_position_delta(
                    human_pos,
                    human_rot,
                    self._initial_human_pos,
                    self._initial_human_rot,
                    child=human_body,
                    shoulder=human_anchor,
                )
                target_pos = self._initial_robot_pos[robot_body] + self.position_scale * (
                    self._initial_robot_shoulder_rot[side] @ human_delta_local
                )
                jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
                jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
                self.mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
                weight = float(np.sqrt(position_weight))
                jacobian_blocks.append(weight * jac_pos[:, self._active_dof_ids])
                error_blocks.append(weight * (target_pos - self.data.xpos[body_id]))
                if use_orientation:
                    human_delta_rot = _relative_rotation(
                        human_rot,
                        self._initial_human_rot,
                        child=human_body,
                        anchor=human_anchor,
                    )
                    target_rot = human_delta_rot @ self._initial_robot_rot[robot_body]
                    current_rot = self.data.xmat[body_id].reshape(3, 3)
                    orientation_weight = 0.18
                    jacobian_blocks.append(np.sqrt(orientation_weight) * jac_rot[:, self._active_dof_ids])
                    error_blocks.append(np.sqrt(orientation_weight) * _rotation_log(target_rot @ current_rot.T))

            jacobian = np.concatenate(jacobian_blocks, axis=0)
            error = np.concatenate(error_blocks, axis=0)
            regularizer = np.sqrt(self.damping) * np.eye(len(self._active_dof_ids), dtype=np.float64)
            delta = np.linalg.lstsq(
                np.concatenate((jacobian, regularizer), axis=0),
                np.concatenate((error, np.zeros(len(self._active_dof_ids), dtype=np.float64)), axis=0),
                rcond=None,
            )[0]
            qvel = np.zeros(self.model.nv, dtype=np.float64)
            qvel[self._active_dof_ids] = np.clip(delta, -self.max_joint_step, self.max_joint_step)
            self.mujoco.mj_integratePos(self.model, qpos, qvel, 1.0)
            self._clamp_active_joint_limits(qpos)

        self._qpos = qpos
        solved = self._heft_joint_positions(qpos)
        reference = self._standing_joint_pos.copy()
        # TWIST2/GMR applies motor-velocity limits before publishing a target.
        # Bound our 50-Hz reference change equivalently; this prevents a noisy
        # PICO frame or an IK local minimum from feeding HEFT an abrupt OOD
        # target even though the pose is still within joint limits.
        arm_delta = np.clip(
            solved[_ARM_JOINT_INDICES] - self._last_reference[_ARM_JOINT_INDICES],
            -self.max_reference_step,
            self.max_reference_step,
        )
        reference[_ARM_JOINT_INDICES] = self._last_reference[_ARM_JOINT_INDICES] + arm_delta
        # Keep the HEFT reference invariant explicit: full-body PICO input
        # informs the arm IK only; pelvis, waist and legs remain the standing
        # reference captured at player startup.
        if not np.array_equal(
            reference[_FIXED_STANDING_JOINT_INDICES],
            self._standing_joint_pos[_FIXED_STANDING_JOINT_INDICES],
        ):
            raise AssertionError("Non-arm PICO retarget output must remain at the startup standing pose.")
        self._last_reference = reference.copy()
        return reference

    def _clamp_active_joint_limits(self, qpos: np.ndarray) -> None:
        for name in self._active_joint_names:
            joint_id = self._joint_ids[name]
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            qpos_id = self._joint_qpos[name]
            lower, upper = self.model.jnt_range[joint_id]
            qpos[qpos_id] = np.clip(qpos[qpos_id], lower, upper)

    def _heft_joint_positions(self, qpos: np.ndarray) -> np.ndarray:
        return np.asarray([qpos[self._joint_qpos[name]] for name in HEFT_JOINT_NAMES], dtype=np.float32)

    def human_visualization(self, frame: PicoUpperBodyFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return the calibration-aligned arm targets for the browser view.

        Raw PICO positions are expressed in the headset's world frame, while
        the MuJoCo scene is expressed in the robot's world frame.  Rendering
        the raw values was therefore often completely off screen and looked
        like an empty viewer.  These points are the exact elbow/wrist targets
        used by IK, plus shoulder anchors derived from that same calibration.
        """
        positions, rotations = self._human_arrays(frame)
        if self._initial_human_pos is None or self._initial_human_rot is None:
            # This can only happen before the first valid frame.  Keep a
            # finite diagnostic view until calibration has taken place.
            display_rotations = [
                _task_frame_rotation(rotations[name], name)
                for name in UPPER_BODY_NAMES
            ]
            return (
                np.asarray([positions[name] for name in UPPER_BODY_NAMES], dtype=np.float32),
                np.asarray(display_rotations, dtype=np.float64),
            )

        visual_positions: dict[str, np.ndarray] = {}
        visual_rotations: dict[str, np.ndarray] = {}
        for side, elbow_body, wrist_body in (
            ("Left", "left_elbow_link", "left_wrist_yaw_link"),
            ("Right", "right_elbow_link", "right_wrist_yaw_link"),
        ):
            shoulder = f"{side}_Shoulder"
            elbow = f"{side}_Elbow"
            wrist = f"{side}_Wrist"
            elbow_delta_local = _shoulder_local_position_delta(
                positions,
                rotations,
                self._initial_human_pos,
                self._initial_human_rot,
                child=elbow,
                shoulder=shoulder,
            )
            wrist_delta_local = _shoulder_local_position_delta(
                positions,
                rotations,
                self._initial_human_pos,
                self._initial_human_rot,
                child=wrist,
                shoulder=shoulder,
            )
            robot_shoulder_rotation = self._initial_robot_shoulder_rot[side]
            elbow_target = self._initial_robot_pos[elbow_body] + self.position_scale * (
                robot_shoulder_rotation @ elbow_delta_local
            )
            wrist_target = self._initial_robot_pos[wrist_body] + self.position_scale * (
                robot_shoulder_rotation @ wrist_delta_local
            )
            # This is the calibrated robot shoulder root, not a raw PICO
            # world point, so every visual target remains next to the robot.
            shoulder_target = self._initial_robot_shoulder_pos[side]
            elbow_rotation = _relative_rotation(
                rotations,
                self._initial_human_rot,
                child=elbow,
                anchor=shoulder,
            ) @ self._initial_robot_rot[elbow_body]
            wrist_rotation = _relative_rotation(
                rotations,
                self._initial_human_rot,
                child=wrist,
                anchor=shoulder,
            ) @ self._initial_robot_rot[wrist_body]
            visual_positions[shoulder] = shoulder_target
            visual_positions[elbow] = elbow_target
            visual_positions[wrist] = wrist_target
            visual_rotations[shoulder] = elbow_rotation
            visual_rotations[elbow] = elbow_rotation
            visual_rotations[wrist] = wrist_rotation
        return (
            np.asarray([visual_positions[name] for name in UPPER_BODY_NAMES], dtype=np.float32),
            np.asarray([visual_rotations[name] for name in UPPER_BODY_NAMES], dtype=np.float64),
        )


class PicoUpperBodyViser:
    """The same Viser/MJViser browser-viewer stack as the reference teleop server."""

    def __init__(
        self,
        model: Any,
        *,
        reference_joint_positions: np.ndarray,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        try:
            import mujoco
            from mjviser.scene import ViserMujocoScene
            from viser import ViserServer
        except ImportError as exc:
            raise ImportError(
                "PICO visualization needs mjviser and viser. Install requirements.txt or run with --no-visualize."
            ) from exc
        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.server = ViserServer(host=host, port=int(port), label="heft-pico-skeleton", verbose=True)
        # mjviser only renders MuJoCo geom groups 0--5.  OrcaLab's converted
        # USD assets may put *all* robot visual geoms in a higher group; the
        # resulting scene tree exists but the browser canvas is empty.  Geom
        # groups are render categories only, so temporarily map those hidden
        # groups to the normal visible group while this viewer is alive.
        geom_groups = np.asarray(model.geom_group[: model.ngeom], dtype=np.int32)
        self._remapped_geom_ids = np.flatnonzero((geom_groups < 0) | (geom_groups >= 6)).astype(np.int32)
        self._remapped_geom_groups = geom_groups[self._remapped_geom_ids].copy()
        if self._remapped_geom_ids.size:
            model.geom_group[self._remapped_geom_ids] = 0
        self.scene = ViserMujocoScene(self.server, model, num_envs=1)
        # Populate the dynamic-body scene immediately.  Previously it was
        # updated only after a valid PICO frame, so opening the URL while
        # waiting for tracking showed a blank canvas instead of the G1 scene.
        mujoco.mj_forward(model, self.data)
        self.scene.update_from_mjdata(self.data)
        self._robot_body_ids, self._robot_parent_indices = self._select_robot_skeleton()
        self._robot_points = None
        self._robot_bones = None
        self._create_robot_skeleton_fallback()
        self._reference_points = None
        self._reference_bones = None
        self._set_reference_stick_figure(reference_joint_positions)
        self.scene.create_visualization_gui(camera_distance=3.0)
        self._points = None
        self._axes = None

    def update(
        self,
        qpos: np.ndarray,
        human_positions: np.ndarray | None,
        human_rotations: np.ndarray | None,
        *,
        reference_joint_positions: np.ndarray,
    ) -> None:
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self._mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        self._update_robot_skeleton_fallback()
        self._set_reference_stick_figure(reference_joint_positions)
        if human_positions is None or human_rotations is None:
            return
        # MJViser centers the MuJoCo scene around ``_scene_offset``.  Apply
        # the same offset to the human skeleton, as the upstream sim2real
        # viewer does, so the yellow axes share the robot's visual frame.
        scene_offset = np.asarray(getattr(self.scene, "_scene_offset", np.zeros(3)), dtype=np.float32).reshape(3)
        positions = np.asarray(human_positions, dtype=np.float32) + scene_offset
        colors = np.broadcast_to(np.asarray((255, 209, 26), dtype=np.uint8), (positions.shape[0], 3))
        if self._points is None:
            self._points = self.server.scene.add_point_cloud(
                "/teleop/human/points", points=positions, colors=colors, point_size=0.012, point_shape="circle",
                precision="float32",
            )
            self._axes = self.server.scene.add_batched_axes(
                "/teleop/human/axes", batched_positions=positions, batched_wxyzs=_matrices_to_wxyz(human_rotations),
                axes_length=0.12, axes_radius=0.003,
            )
            return
        self._points.points = positions
        self._points.colors = colors
        self._axes.batched_positions = positions
        self._axes.batched_wxyzs = _matrices_to_wxyz(human_rotations)

    def _select_robot_skeleton(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dynamic body ids and parent indices for a mesh-free fallback.

        This deliberately uses only MuJoCo body transforms.  It therefore
        remains visible for OrcaLab models whose USD meshes are unavailable to
        MJViser or whose original geom groups are not browser-renderable.
        """
        try:
            from mjviser.conversions import is_fixed_body
        except ImportError:  # pragma: no cover - bundled with mjviser
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        body_ids = np.asarray(
            [
                body_id
                for body_id in range(1, int(self.model.nbody))
                if not is_fixed_body(self.model, body_id)
            ],
            dtype=np.int32,
        )
        if not body_ids.size:
            return body_ids, np.zeros(0, dtype=np.int32)
        index_by_id = {int(body_id): index for index, body_id in enumerate(body_ids)}
        parent_indices: list[tuple[int, int]] = []
        for child_index, child_id in enumerate(body_ids):
            parent_id = int(self.model.body_parentid[int(child_id)])
            parent_index = index_by_id.get(parent_id)
            if parent_index is not None:
                parent_indices.append((parent_index, child_index))
        return body_ids, np.asarray(parent_indices, dtype=np.int32).reshape(-1, 2)

    def _create_robot_skeleton_fallback(self) -> None:
        if not self._robot_body_ids.size:
            return
        positions = self._robot_positions_for_view()
        # A collapsed local mirror is not a robot visualization.  Do not draw
        # its misleading one-pixel dot; the qpos-based diagnostic below is
        # guaranteed to remain useful in that situation.
        if not np.all(np.isfinite(positions)) or float(np.ptp(positions, axis=0).max()) < 0.10:
            return
        # Cyan points and blue bones are intentionally high contrast.  They
        # guarantee that the browser communicates the live retarget pose even
        # if the original USD visual mesh cannot be converted by MJViser.
        self._robot_points = self.server.scene.add_point_cloud(
            "/teleop/robot_fallback/joints",
            points=positions,
            colors=np.asarray((0, 214, 255), dtype=np.uint8),
            point_size=0.035,
            point_shape="circle",
            precision="float32",
        )
        self._robot_bones = self.server.scene.add_line_segments(
            "/teleop/robot_fallback/bones",
            points=self._robot_bone_points(positions),
            colors=np.asarray((20, 115, 230), dtype=np.uint8),
            line_width=2.5,
        )

    def _robot_positions_for_view(self) -> np.ndarray:
        scene_offset = np.asarray(getattr(self.scene, "_scene_offset", np.zeros(3)), dtype=np.float32).reshape(3)
        return np.asarray(self.data.xpos[self._robot_body_ids], dtype=np.float32) + scene_offset

    def _robot_bone_points(self, positions: np.ndarray) -> np.ndarray:
        if not self._robot_parent_indices.size:
            return np.zeros((0, 2, 3), dtype=np.float32)
        return np.asarray(positions[self._robot_parent_indices], dtype=np.float32)

    def _update_robot_skeleton_fallback(self) -> None:
        if self._robot_points is None or self._robot_bones is None:
            return
        positions = self._robot_positions_for_view()
        self._robot_points.points = positions
        self._robot_bones.points = self._robot_bone_points(positions)

    def _set_reference_stick_figure(self, reference_joint_positions: np.ndarray) -> None:
        """Update the always-visible diagnostic from the exact HEFT reference."""
        reference_figure = _model_reference_stick_figure(
            self.model,
            self._mujoco,
            reference_joint_positions,
        )
        if reference_figure is None:
            points, segments = _g1_reference_stick_figure(reference_joint_positions)
        else:
            points, segments = reference_figure
        if self._reference_points is None:
            self._reference_points = self.server.scene.add_point_cloud(
                "/teleop/reference_g1/joints",
                points=points,
                colors=np.asarray((255, 82, 82), dtype=np.uint8),
                point_size=0.055,
                point_shape="circle",
                precision="float32",
            )
            self._reference_bones = self.server.scene.add_line_segments(
                "/teleop/reference_g1/bones",
                points=np.asarray(points[segments], dtype=np.float32),
                colors=np.asarray((225, 35, 35), dtype=np.uint8),
                line_width=4.0,
            )
            return
        self._reference_points.points = points
        self._reference_bones.points = np.asarray(points[segments], dtype=np.float32)

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass
        if self._remapped_geom_ids.size:
            self.model.geom_group[self._remapped_geom_ids] = self._remapped_geom_groups


def _matrices_to_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Use MuJoCo's robust matrix-to-quaternion implementation for Viser."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - guarded by PicoUpperBodyViser
        raise RuntimeError("mujoco is required") from exc
    matrices = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    quats = np.empty((len(matrices), 4), dtype=np.float32)
    for idx, matrix in enumerate(matrices):
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
        quats[idx] = quat
    return quats


__all__ = [
    "G1SkeletonArmsRetargeter",
    "PicoUpperBodyFrame",
    "PicoUpperBodyStream",
    "PicoUpperBodyViser",
    "UPPER_BODY_NAMES",
    "load_xrobotoolkit_sdk",
    "parse_pico_upper_body_snapshot",
]
