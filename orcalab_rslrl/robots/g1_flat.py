"""Unitree G1 flat-terrain model builder for Orca.

Builds a MuJoCo model from the actuator-free ``g1.xml`` (with ``left_foot`` /
``right_foot`` sites) and adds the runtime configuration in code:

- ground plane and collision setup (``FULL_COLLISION``: self collisions on,
  feet condim=3 priority=1 friction=0.6, everything else condim=1),
- built-in PD <position> actuators with per-motor stiffness/damping
  and reflected-inertia armature,
- the HOME keyframe as default state,
- native contact sensors (feet vs floor: found + net force; whole-body self
  collision), foot framepos/framelinvel sensors, IMU gyro/velocimeter and the
  root subtree angular-momentum sensor already present in the XML.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import mujoco
import numpy as np

from ..scene_options import (
    ORCA_TRAIN_SCENE_OPTIONS,
    UNITREE_ORCA_GROUND_FRICTION,
    UNITREE_ORCA_GROUND_SOLIMP,
    UNITREE_ORCA_GROUND_SOLREF,
    apply_orca_train_scene_options,
)


G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _reflected_inertia(rotor_inertia: tuple[float, float, float], gear: tuple[float, float, float]) -> float:
    assert gear[0] == 1
    return (
        rotor_inertia[0] * (gear[1] * gear[2]) ** 2
        + rotor_inertia[1] * gear[2] ** 2
        + rotor_inertia[2]
    )


ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1, 5, 5))

_NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
_DAMPING_RATIO = 2.0


def _pd_gains(armature: float) -> tuple[float, float]:
    return armature * _NATURAL_FREQ**2, 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


# (armature, effort_limit, joint name patterns) per motor group, mirroring g1_constants.py.
_ACTUATOR_GROUPS: tuple[tuple[float, float, tuple[str, ...]], ...] = (
    (ARMATURE_5020, 25.0, (r".*_elbow_joint", r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_wrist_roll_joint")),
    (ARMATURE_7520_14, 88.0, (r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint")),
    (ARMATURE_7520_22, 139.0, (r".*_hip_roll_joint", r".*_knee_joint")),
    (ARMATURE_4010, 5.0, (r".*_wrist_pitch_joint", r".*_wrist_yaw_joint")),
    (2 * ARMATURE_5020, 50.0, (r"waist_pitch_joint", r"waist_roll_joint")),
    (2 * ARMATURE_5020, 50.0, (r".*_ankle_pitch_joint", r".*_ankle_roll_joint")),
)

HOME_JOINT_POS: dict[str, float] = {
    r".*_hip_pitch_joint": -0.1,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": -0.2,
    r".*_shoulder_pitch_joint": 0.35,
    r".*_elbow_joint": 0.87,
    r"left_shoulder_roll_joint": 0.18,
    r"right_shoulder_roll_joint": -0.18,
}
HOME_BASE_POS = (0.0, 0.0, 0.8)
TERRAIN_FLAT = "flat"
TERRAIN_STAIR_MID_FLAT = "stair-mid-flat"
TERRAIN_KINDS = (TERRAIN_FLAT, TERRAIN_STAIR_MID_FLAT)


def _joint_group(joint_name: str) -> tuple[float, float]:
    """(armature, effort_limit) for a joint."""
    for armature, effort, patterns in _ACTUATOR_GROUPS:
        if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
            return armature, effort
    raise KeyError(f"No actuator group for joint {joint_name!r}")


def g1_action_scale() -> tuple[float, ...]:
    """Orca G1_ACTION_SCALE = 0.25 * effort_limit / stiffness, in ctrl order."""
    scales = []
    for joint_name in G1_JOINT_NAMES:
        armature, effort = _joint_group(joint_name)
        stiffness, _ = _pd_gains(armature)
        scales.append(0.25 * effort / stiffness)
    return tuple(scales)


def resolve_g1_flat_mjcf(explicit_path: str | Path | None = None) -> Path:
    """Find the bundled Orca-style actuator-free ``g1.xml`` with foot sites.

    The package-local XML is the default, so training and playback do not
    depend on an external asset checkout. ``explicit_path`` and
    ``ORCALAB_G1_FLAT_MJCF`` remain opt-in overrides for experiments.
    """
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if env_path := os.environ.get("ORCALAB_G1_FLAT_MJCF"):
        candidates.append(Path(env_path).expanduser())
    candidates.append(
        resources.files("orcalab_rslrl")
        .joinpath("assets/robots/unitree_g1/xmls/g1.xml")
    )
    for path in candidates:
        local_path = Path(path)
        if local_path.exists():
            return local_path
    raise FileNotFoundError(
        "No bundled Orca-style g1.xml found; reinstall package assets or set ORCALAB_G1_FLAT_MJCF"
    )


def _add_flat_floor(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01],
        contype=1, conaffinity=1, condim=3, priority=0,
        friction=list(UNITREE_ORCA_GROUND_FRICTION),
        solref=list(UNITREE_ORCA_GROUND_SOLREF),
        solimp=list(UNITREE_ORCA_GROUND_SOLIMP),
    )


def _add_stair_mid_flat_terrain(spec: mujoco.MjSpec) -> None:
    """Simple collision proxy for OrcaLab's terrain_stair_mid_flat_usda prefab."""
    _add_flat_floor(spec)
    step_count = 6
    step_height = 0.08
    step_depth = 0.30
    stair_width = 2.40
    start_x = 0.45

    for index in range(step_count):
        height = (index + 1) * step_height
        spec.worldbody.add_geom(
            name=f"terrain_stair_{index:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[start_x + index * step_depth, 0.0, 0.5 * height],
            size=[0.5 * step_depth, 0.5 * stair_width, 0.5 * height],
            contype=1,
            conaffinity=1,
            condim=3,
            friction=[0.8, 0.005, 0.0001],
        )

    platform_height = step_count * step_height
    spec.worldbody.add_geom(
        name="terrain_stair_top_flat",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[start_x + step_count * step_depth + 1.0, 0.0, 0.5 * platform_height],
        size=[1.0, 0.5 * stair_width, 0.5 * platform_height],
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.8, 0.005, 0.0001],
    )


def _add_terrain(spec: mujoco.MjSpec, terrain_kind: str) -> None:
    if terrain_kind == TERRAIN_FLAT:
        _add_flat_floor(spec)
        return
    if terrain_kind == TERRAIN_STAIR_MID_FLAT:
        _add_stair_mid_flat_terrain(spec)
        return
    raise ValueError(f"unknown terrain_kind {terrain_kind!r}; expected one of {TERRAIN_KINDS}")


_FOUND_BIT = 1 << 0
_FORCE_BIT = 1 << 1
_REDUCE_NONE = 0
_REDUCE_NETFORCE = 3


@dataclass(frozen=True)
class G1FlatSensors:
    """Names of the sensors the MDP terms read from the backend."""

    base_ang_vel: str = "imu_ang_vel"
    base_lin_vel: str = "imu_lin_vel"
    root_angmom: str = "root_angmom"
    foot_found: tuple[str, str] = ("left_foot_ground_found", "right_foot_ground_found")
    foot_force: tuple[str, str] = ("left_foot_ground_force", "right_foot_ground_force")
    foot_pos: tuple[str, str] = ("left_foot_pos", "right_foot_pos")
    foot_vel: tuple[str, str] = ("left_foot_vel", "right_foot_vel")
    self_collision: str = "self_collision_found"


@dataclass(frozen=True)
class G1FlatRobot:
    joint_names: tuple[str, ...] = G1_JOINT_NAMES
    action_scale: tuple[float, ...] = field(default_factory=g1_action_scale)
    torso_body: str = "torso_link"
    pelvis_body: str = "pelvis"
    sensors: G1FlatSensors = field(default_factory=G1FlatSensors)
    soft_joint_pos_limit_factor: float = 0.9


def build_g1_flat_spec(
    mjcf_path: str | Path | None = None,
    *,
    foot_friction_range: tuple[float, float] | None = (0.3, 1.6),
    base_com_offset_range: tuple[float, float] | None = (-0.05, 0.05),
    timestep: float = ORCA_TRAIN_SCENE_OPTIONS.timestep,
    terrain_kind: str = TERRAIN_FLAT,
    seed: int | None = None,
) -> mujoco.MjSpec:
    """Build the G1 flat velocity specification.

    ``foot_friction_range`` ports the Orca ``foot_friction`` startup event
    (shared_random=True: one friction sample shared by every foot geom).
    ``base_com_offset_range`` ports the ``base_com`` startup event on
    ``torso_link``. Both are sampled once at build time because MJWarp shares
    a single model across all parallel worlds. Pass None to disable.
    """
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec.from_file(str(resolve_g1_flat_mjcf(mjcf_path)))
    _add_terrain(spec, terrain_kind)

    # FULL_COLLISION: feet condim=3 priority=1 friction=0.6, other collisions condim=1.
    foot_friction = 0.6
    if foot_friction_range is not None:
        foot_friction = float(rng.uniform(*foot_friction_range))
    for geom in spec.geoms:
        if geom.name and geom.name.endswith("_collision"):
            if re.fullmatch(r"^(left|right)_foot[1-7]_collision$", geom.name):
                geom.condim = 3
                geom.priority = 1
                geom.friction = [foot_friction, 0.005, 0.0001]
            else:
                geom.condim = 1

    # base_com startup randomization (shared across worlds).
    if base_com_offset_range is not None:
        for body in spec.bodies:
            if body.name == "torso_link":
                body.ipos = np.asarray(body.ipos) + rng.uniform(*base_com_offset_range, size=3)

    # Built-in PD <position> actuators, one per joint, Orca gains and armature.
    for joint_name in G1_JOINT_NAMES:
        armature, effort = _joint_group(joint_name)
        stiffness, damping = _pd_gains(armature)
        joint = spec.joint(joint_name)
        joint.armature = armature
        actuator = spec.add_actuator(
            name=joint_name.removesuffix("_joint"),
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        actuator.gainprm[0] = stiffness
        actuator.biasprm[1] = -stiffness
        actuator.biasprm[2] = -damping
        actuator.forcerange = [-effort, effort]
        actuator.ctrlrange = list(joint.range)
        actuator.ctrllimited = True

    # Contact sensors (native mjSENS_CONTACT, supported by MJWarp).
    def add_contact_sensor(name, objname, refname, bits, reduce_mode, reftype):
        sensor = spec.add_sensor(
            name=name, type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY, objname=objname,
        )
        sensor.intprm[0] = bits
        sensor.intprm[1] = reduce_mode
        sensor.intprm[2] = 1  # num_slots
        if refname is not None:
            sensor.reftype = reftype
            sensor.refname = refname

    for side in ("left", "right"):
        ground_ref = "floor" if terrain_kind == TERRAIN_FLAT else None
        add_contact_sensor(
            f"{side}_foot_ground_found", f"{side}_ankle_roll_link", ground_ref,
            _FOUND_BIT, _REDUCE_NETFORCE, mujoco.mjtObj.mjOBJ_GEOM,
        )
        add_contact_sensor(
            f"{side}_foot_ground_force", f"{side}_ankle_roll_link", ground_ref,
            _FORCE_BIT, _REDUCE_NETFORCE, mujoco.mjtObj.mjOBJ_GEOM,
        )
        spec.add_sensor(
            name=f"{side}_foot_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_foot",
        )
        spec.add_sensor(
            name=f"{side}_foot_vel", type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_foot",
        )
    add_contact_sensor(
        "self_collision_found", "pelvis", "pelvis",
        _FOUND_BIT, _REDUCE_NONE, mujoco.mjtObj.mjOBJ_XBODY,
    )

    # HOME keyframe becomes the default reset state.
    qpos = list(HOME_BASE_POS) + [1.0, 0.0, 0.0, 0.0] + [0.0] * len(G1_JOINT_NAMES)
    for index, joint_name in enumerate(G1_JOINT_NAMES):
        for pattern, value in HOME_JOINT_POS.items():
            if re.fullmatch(pattern, joint_name):
                qpos[7 + index] = value
    spec.add_key(name="home", qpos=qpos)

    # Apply the canonical Orca G1 training profile.
    apply_orca_train_scene_options(spec, timestep=timestep)

    return spec


def build_g1_flat_model(
    mjcf_path: str | Path | None = None,
    *,
    foot_friction_range: tuple[float, float] | None = (0.3, 1.6),
    base_com_offset_range: tuple[float, float] | None = (-0.05, 0.05),
    timestep: float = ORCA_TRAIN_SCENE_OPTIONS.timestep,
    terrain_kind: str = TERRAIN_FLAT,
    seed: int | None = None,
) -> mujoco.MjModel:
    """Build and compile the G1 flat velocity model."""

    return build_g1_flat_spec(
        mjcf_path,
        foot_friction_range=foot_friction_range,
        base_com_offset_range=base_com_offset_range,
        timestep=timestep,
        terrain_kind=terrain_kind,
        seed=seed,
    ).compile()
