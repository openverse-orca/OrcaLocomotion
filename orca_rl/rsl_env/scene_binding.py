from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .model_scanner import build_suffix_template, match_robot_instances, probe_scene_model, require_complete_matches


@dataclass(frozen=True)
class SceneBinding:
    agent_names: list[str]
    robot_config: dict
    model_xml_path: str | None = None
    source: str = "orcalab_scene"


G1_AGENT_ASSET_PATH = "assets/13951baeb514b4b9/default_project/prefabs/g1_pick_usda"

G1_JOINT_SUFFIXES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

G1_ACTUATOR_SUFFIXES = [name.removesuffix("_joint") for name in G1_JOINT_SUFFIXES]
G1_ACTUATOR_SUFFIX_VARIANTS = [
    G1_ACTUATOR_SUFFIXES,
    list(G1_JOINT_SUFFIXES),
]

G1_DEFAULT_DOF_ANGLES = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

G1_MOTOR_KP = [
    100.0, 100.0, 100.0, 200.0, 20.0, 20.0,
    100.0, 100.0, 100.0, 200.0, 20.0, 20.0,
    400.0, 400.0, 400.0,
    90.0, 60.0, 20.0, 60.0, 4.0, 4.0, 0.0,
    90.0, 60.0, 20.0, 60.0, 4.0, 4.0, 4.0,
]

G1_MOTOR_KD = [
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
    5.0, 5.0, 5.0,
    5.0, 2.0, 1.0, 0.4, 1.0, 0.2, 0.2,
    0.2, 2.0, 1.0, 0.4, 1.0, 0.2, 0.2,
]

G1_MOTOR_EFFORT_LIMITS = [
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
]

G1_JOINT_LIMITS = [
    [-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576],
    [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
    [-2.5307, 2.8798], [-2.9671, 0.5236], [-2.7576, 2.7576],
    [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
    [-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52],
    [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.618, 2.618],
    [-1.0472, 2.0944], [-1.972222054, 1.972222054], [-1.61443, 1.61443],
    [-1.61443, 1.61443], [-3.0892, 2.6704], [-2.2515, 1.5882],
    [-2.618, 2.618], [-1.0472, 2.0944], [-1.972222054, 1.972222054],
    [-1.61443, 1.61443], [-1.61443, 1.61443],
]


def resolve_g1_scene_binding(
    orcagym_addr: str,
    time_step: float,
    min_count: int = 1,
    max_count: int | None = 1,
    num_envs: int | None = None,
    *,
    asset_path: str = G1_AGENT_ASSET_PATH,
) -> SceneBinding:
    desired_count = int(num_envs or min_count)
    if desired_count != 1:
        raise ValueError("HEFT G1 + Dex3 playback supports exactly one robot.")
    if num_envs is not None:
        min_count = desired_count
        max_count = desired_count

    report, actuator_suffixes = _scan_g1_scene_binding(orcagym_addr, time_step)
    _raise_if_g1_scene_has_no_actuators(report, asset_path=asset_path)
    matches = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )
    agent_names = [matches[0].agent_name]
    robot_config = _build_g1_robot_config()
    robot_config.update(
        actuator_names=list(actuator_suffixes),
        model_name="G1",
        log_agent_names=agent_names,
        visualize_command_agent_names=agent_names,
        playable_agent_name=agent_names[0],
    )
    return SceneBinding(agent_names=agent_names, robot_config=robot_config)


def _scan_g1_scene_binding(orcagym_addr: str, time_step: float):
    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=time_step)
    candidates = []
    for actuator_suffixes in G1_ACTUATOR_SUFFIX_VARIANTS:
        template = build_suffix_template(
            model_name="G1",
            joints=["floating_base_joint", *G1_JOINT_SUFFIXES],
            actuators=actuator_suffixes,
        )
        report = match_robot_instances(template, scene_names)
        candidates.append((report, actuator_suffixes))
        if report.complete_matches:
            return report, actuator_suffixes

    def score(candidate) -> int:
        report, _ = candidate
        return max(
            (sum(len(names) for names in match.matched_names.values()) for match in report.partial_matches),
            default=0,
        )

    return max(candidates, key=score)


def _raise_if_g1_scene_has_no_actuators(report, *, asset_path: str) -> None:
    if report.complete_matches or report.scene_names.actuators:
        return
    g1_joints = [name for name in report.scene_names.joints if name.startswith("g1_") or "_hip_" in name]
    if g1_joints:
        raise RuntimeError(
            "The OrcaLab scene contains G1 joints but no motor actuators. "
            f"Use an actuated G1 + Dex3-1 prefab; configured path={asset_path!r}."
        )


def _build_g1_robot_config() -> dict:
    return {
        "base_joint_name": "floating_base_joint",
        "leg_joint_names": list(G1_JOINT_SUFFIXES),
        "actuator_names": list(G1_ACTUATOR_SUFFIXES),
        "neutral_joint_angles": dict(zip(G1_JOINT_SUFFIXES, G1_DEFAULT_DOF_ANGLES)),
        "joint_limits": deepcopy(G1_JOINT_LIMITS),
        "kps": list(G1_MOTOR_KP),
        "kds": list(G1_MOTOR_KD),
        "motor_effort_limit_list": list(G1_MOTOR_EFFORT_LIMITS),
        "imu_site_name": "imu_in_pelvis",
        "contact_site_names": ["lf-tc-front", "lf-tc-back", "rf-tc-front", "rf-tc-back"],
        "sensor_foot_touch_names": ["lf-touch-front", "lf-touch-back", "rf-touch-front", "rf-touch-back"],
        "base_contact_body_names": ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"],
        "foot_body_names": [
            "left_ankle_roll_link", "left_ankle_roll_link",
            "right_ankle_roll_link", "right_ankle_roll_link",
        ],
        "action_scale": 0.5,
    }
