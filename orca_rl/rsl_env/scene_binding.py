from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

from .model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from .robot_configs import GO2_CONFIG


@dataclass(frozen=True)
class SceneBinding:
    agent_name: str
    robot_config: dict


G1_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_old_usda"

G1_JOINT_SUFFIXES = [
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
]

G1_ACTUATOR_SUFFIXES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

G1_DEFAULT_DOF_ANGLES = [
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

G1_MOTOR_KP = [
    100.0,
    100.0,
    100.0,
    200.0,
    20.0,
    20.0,
    100.0,
    100.0,
    100.0,
    200.0,
    20.0,
    20.0,
    400.0,
    400.0,
    400.0,
    90.0,
    60.0,
    20.0,
    60.0,
    4.0,
    4.0,
    0.0,
    90.0,
    60.0,
    20.0,
    60.0,
    4.0,
    4.0,
    4.0,
]

G1_MOTOR_KD = [
    2.5,
    2.5,
    2.5,
    5.0,
    0.2,
    0.1,
    2.5,
    2.5,
    2.5,
    5.0,
    0.2,
    0.1,
    5.0,
    5.0,
    5.0,
    2.0,
    1.0,
    0.4,
    1.0,
    0.2,
    0.2,
    0.2,
    2.0,
    1.0,
    0.4,
    1.0,
    0.2,
    0.2,
    0.2,
]

G1_MOTOR_EFFORT_LIMITS = [
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
]

G1_JOINT_LIMITS = [
    [-2.5307, 2.8798],
    [-0.5236, 2.9671],
    [-2.7576, 2.7576],
    [-0.087267, 2.8798],
    [-0.87267, 0.5236],
    [-0.2618, 0.2618],
    [-2.5307, 2.8798],
    [-2.9671, 0.5236],
    [-2.7576, 2.7576],
    [-0.087267, 2.8798],
    [-0.87267, 0.5236],
    [-0.2618, 0.2618],
    [-2.618, 2.618],
    [-0.52, 0.52],
    [-0.52, 0.52],
    [-3.0892, 2.6704],
    [-1.5882, 2.2515],
    [-2.618, 2.618],
    [-1.0472, 2.0944],
    [-1.972222054, 1.972222054],
    [-1.61443, 1.61443],
    [-1.61443, 1.61443],
    [-3.0892, 2.6704],
    [-2.2515, 1.5882],
    [-2.618, 2.618],
    [-1.0472, 2.0944],
    [-1.972222054, 1.972222054],
    [-1.61443, 1.61443],
    [-1.61443, 1.61443],
]


def resolve_go2_scene_binding(
    orcagym_addr: str,
    time_step: float,
    min_count: int = 1,
    max_count: int = 1,
) -> SceneBinding:
    robot_config = deepcopy(GO2_CONFIG)
    template = build_suffix_template(
        model_name="go2",
        joints=[robot_config["base_joint_name"], *list(robot_config["leg_joint_names"])],
        actuators=list(robot_config["actuator_names"]),
        sites=[robot_config["imu_site_name"], *list(robot_config["contact_site_names"])],
        bodies=[*list(robot_config.get("base_contact_body_names", [])), *list(robot_config.get("foot_body_names", []))],
        sensors=list(robot_config.get("sensor_foot_touch_names", [])),
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    match = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )[0]

    agent_name = match.agent_name
    robot_config["model_name"] = "go2"
    robot_config["log_agent_names"] = [agent_name]
    robot_config["visualize_command_agent_names"] = [agent_name]
    robot_config["playable_agent_name"] = agent_name
    return SceneBinding(agent_name=agent_name, robot_config=robot_config)


def resolve_g1_scene_binding(
    orcagym_addr: str,
    time_step: float,
    min_count: int = 1,
    max_count: int = 1,
    *,
    spawn_if_missing: bool = False,
    spawn_agent_name: str = "g1_000",
    asset_path: str = G1_AGENT_ASSET_PATH,
) -> SceneBinding:
    robot_config = _build_g1_robot_config()
    template = build_suffix_template(
        model_name="G1",
        joints=["floating_base_joint", *G1_JOINT_SUFFIXES],
        actuators=G1_ACTUATOR_SUFFIXES,
        sensors=["imu_quat", "imu_gyro"],
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    if spawn_if_missing and not report.complete_matches:
        try:
            publish_g1_scene(orcagym_addr=orcagym_addr, agent_name=spawn_agent_name, asset_path=asset_path)
        except Exception as exc:
            raise RuntimeError(
                "G1 auto-publish failed. The current OrcaStudio asset library does not know the configured "
                f"spawnable path: {asset_path!r}. Drag one G1 actor into the layout manually, or update "
                "`scene_binding.asset_path` to the spawnable path available in this OrcaStudio project."
            ) from exc
        report = scan_scene_for_template(
            orcagym_addr=orcagym_addr,
            time_step=time_step,
            template=template,
        )

    match = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )[0]

    agent_name = match.agent_name
    robot_config["model_name"] = "G1"
    robot_config["log_agent_names"] = [agent_name]
    robot_config["visualize_command_agent_names"] = [agent_name]
    robot_config["playable_agent_name"] = agent_name
    return SceneBinding(agent_name=agent_name, robot_config=robot_config)


def publish_g1_scene(orcagym_addr: str, agent_name: str, asset_path: str = G1_AGENT_ASSET_PATH) -> None:
    from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
    from orca_gym.utils.rotations import euler2quat

    temp_scene = OrcaGymScene(orcagym_addr)
    try:
        temp_scene.publish_scene()
        time.sleep(1)
    finally:
        temp_scene.close()
    time.sleep(1)

    scene = OrcaGymScene(orcagym_addr)
    try:
        agent = Actor(
            name=agent_name,
            asset_path=asset_path.replace("//", "/"),
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(agent)
        scene.publish_scene()
        time.sleep(3)
    finally:
        scene.close()
    time.sleep(1)


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
            "left_ankle_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "right_ankle_roll_link",
        ],
        "action_scale": 0.5,
    }
