from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import time

from .local_mjcf import build_local_mjcf_batch, prepare_local_terrain_cfg, resolve_existing_xml_path
from .model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from .robot_configs import GO2_CONFIG


@dataclass(frozen=True)
class SceneBinding:
    agent_names: list[str]
    robot_config: dict
    model_xml_path: str | None = None
    source: str = "orcalab_scene"

    @property
    def agent_name(self) -> str:
        return self.agent_names[0]


G1_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda"
GO2_AGENT_ASSET_PATH = "assets/e071469a36d3c8aa/unitree_robots/prefabs/go2_usda"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

G1_LOCAL_XML_CANDIDATES = [
    os.environ.get("ORCA_RL_G1_XML", ""),
    _PROJECT_ROOT / "third_party" / "unitree_rl_mjlab" / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml",
    "/home/huan-hu/OrcaPlayground/examples/g1/g1_29dof_old.xml",
    "/home/huan-hu/下载/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml",
]
GO2_LOCAL_XML_CANDIDATES = [
    os.environ.get("ORCA_RL_GO2_XML", ""),
    _PROJECT_ROOT
    / "third_party"
    / "unitree_rl_mjlab"
    / "src"
    / "assets"
    / "robots"
    / "unitree_go2"
    / "xmls"
    / "scene_go2.xml",
    "/home/huan-hu/下载/unitree_rl_mjlab/src/assets/robots/unitree_go2/xmls/scene_go2.xml",
]

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
    max_count: int | None = 1,
    num_envs: int | None = None,
    *,
    spawn_if_missing: bool = False,
    max_auto_spawn_count: int = 1,
    spawn_agent_name: str = "go2_000",
    asset_path: str = GO2_AGENT_ASSET_PATH,
    local_xml_path: str | None = None,
    local_clone_spacing: float = 2.0,
    local_xml_output_dir: str | None = None,
    terrain_cfg: dict | None = None,
    terrain_seed: int = 1,
    extra_actors: list[dict] | None = None,
) -> SceneBinding:
    desired_count = int(num_envs or min_count)
    if num_envs is not None:
        min_count = desired_count
        max_count = desired_count
    robot_config = deepcopy(GO2_CONFIG)
    if local_xml_path is not None:
        source_xml_path = resolve_existing_xml_path(local_xml_path, GO2_LOCAL_XML_CANDIDATES)
        agent_names = [f"{spawn_agent_name.rsplit('_', 1)[0]}_{index:03d}" for index in range(desired_count)]
        local_terrain_cfg = prepare_local_terrain_cfg(
            terrain_cfg,
            agent_count=desired_count,
            spacing=float(local_clone_spacing),
        )
        model_xml_path = build_local_mjcf_batch(
            source_xml_path=source_xml_path,
            agent_names=agent_names,
            spacing=float(local_clone_spacing),
            output_dir=local_xml_output_dir,
            terrain_cfg=local_terrain_cfg,
            terrain_seed=int(terrain_seed),
        )
        robot_config["model_name"] = "go2"
        robot_config["log_agent_names"] = agent_names
        robot_config["visualize_command_agent_names"] = agent_names
        robot_config["playable_agent_name"] = agent_names[0]
        robot_config["local_source_xml_path"] = source_xml_path
        if local_terrain_cfg is not None:
            robot_config["local_terrain_cfg"] = local_terrain_cfg
        return SceneBinding(
            agent_names=agent_names,
            robot_config=robot_config,
            model_xml_path=model_xml_path,
            source="local_mjcf",
        )

    template = build_suffix_template(
        model_name="go2",
        joints=[robot_config["base_joint_name"], *list(robot_config["leg_joint_names"])],
        actuators=list(robot_config["actuator_names"]),
        bodies=[*list(robot_config.get("base_contact_body_names", [])), *list(robot_config.get("foot_body_names", []))],
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    if spawn_if_missing and len(report.complete_matches) < desired_count:
        max_auto_spawn_count = max(0, int(max_auto_spawn_count))
        missing_count = desired_count - len(report.complete_matches)
        if missing_count > max_auto_spawn_count:
            raise RuntimeError(
                "Refusing to auto-publish a large GO2 batch into the OrcaLab scene. "
                f"requested={desired_count}, existing={len(report.complete_matches)}, missing={missing_count}, "
                f"max_auto_spawn_count={max_auto_spawn_count}. Prepare a batched scene explicitly or raise "
                "`scene_binding.max_auto_spawn_count` after accepting the scene compile/render cost."
            )
        try:
            publish_go2_scene(
                orcagym_addr=orcagym_addr,
                agent_name=spawn_agent_name,
                asset_path=asset_path,
                agent_count=desired_count,
                extra_actors=extra_actors,
            )
        except Exception as exc:
            raise RuntimeError(
                "GO2 auto-publish failed. The current OrcaStudio asset library does not know the configured "
                f"spawnable path: {asset_path!r}. Drag one GO2 actor into the layout manually, or update "
                "`scene_binding.asset_path` to the spawnable path available in this OrcaStudio project."
            ) from exc
        report = scan_scene_for_template(
            orcagym_addr=orcagym_addr,
            time_step=time_step,
            template=template,
        )

    matches = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )

    agent_names = [match.agent_name for match in matches[:desired_count]]
    robot_config["model_name"] = "go2"
    robot_config["log_agent_names"] = agent_names
    robot_config["visualize_command_agent_names"] = agent_names
    robot_config["playable_agent_name"] = agent_names[0]
    return SceneBinding(agent_names=agent_names, robot_config=robot_config)


def resolve_g1_scene_binding(
    orcagym_addr: str,
    time_step: float,
    min_count: int = 1,
    max_count: int | None = 1,
    num_envs: int | None = None,
    *,
    spawn_if_missing: bool = False,
    max_auto_spawn_count: int = 1,
    spawn_agent_name: str = "g1_000",
    asset_path: str = G1_AGENT_ASSET_PATH,
    local_xml_path: str | None = None,
    local_clone_spacing: float = 2.0,
    local_xml_output_dir: str | None = None,
    terrain_cfg: dict | None = None,
    terrain_seed: int = 1,
    extra_actors: list[dict] | None = None,
) -> SceneBinding:
    desired_count = int(num_envs or min_count)
    if num_envs is not None:
        min_count = desired_count
        max_count = desired_count
    robot_config = _build_g1_robot_config()
    if local_xml_path is not None:
        source_xml_path = resolve_existing_xml_path(local_xml_path, G1_LOCAL_XML_CANDIDATES)
        agent_names = [f"{spawn_agent_name.rsplit('_', 1)[0]}_{index:03d}" for index in range(desired_count)]
        local_terrain_cfg = prepare_local_terrain_cfg(
            terrain_cfg,
            agent_count=desired_count,
            spacing=float(local_clone_spacing),
        )
        model_xml_path = build_local_mjcf_batch(
            source_xml_path=source_xml_path,
            agent_names=agent_names,
            spacing=float(local_clone_spacing),
            output_dir=local_xml_output_dir,
            terrain_cfg=local_terrain_cfg,
            terrain_seed=int(terrain_seed),
        )
        robot_config["model_name"] = "G1"
        robot_config["log_agent_names"] = agent_names
        robot_config["visualize_command_agent_names"] = agent_names
        robot_config["playable_agent_name"] = agent_names[0]
        robot_config["local_source_xml_path"] = source_xml_path
        if local_terrain_cfg is not None:
            robot_config["local_terrain_cfg"] = local_terrain_cfg
        return SceneBinding(
            agent_names=agent_names,
            robot_config=robot_config,
            model_xml_path=model_xml_path,
            source="local_mjcf",
        )

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
    if spawn_if_missing and len(report.complete_matches) < desired_count:
        max_auto_spawn_count = max(0, int(max_auto_spawn_count))
        missing_count = desired_count - len(report.complete_matches)
        if missing_count > max_auto_spawn_count:
            raise RuntimeError(
                "Refusing to auto-publish a large G1 batch into the OrcaLab scene. "
                f"requested={desired_count}, existing={len(report.complete_matches)}, missing={missing_count}, "
                f"max_auto_spawn_count={max_auto_spawn_count}. Prepare a batched scene explicitly or raise "
                "`scene_binding.max_auto_spawn_count` after accepting the scene compile/render cost."
            )
        try:
            publish_g1_scene(
                orcagym_addr=orcagym_addr,
                agent_name=spawn_agent_name,
                asset_path=asset_path,
                agent_count=desired_count,
                extra_actors=extra_actors,
            )
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

    _raise_if_g1_scene_has_no_actuators(report, asset_path=asset_path)
    matches = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )

    agent_names = [match.agent_name for match in matches[:desired_count]]
    robot_config["model_name"] = "G1"
    robot_config["log_agent_names"] = agent_names
    robot_config["visualize_command_agent_names"] = agent_names
    robot_config["playable_agent_name"] = agent_names[0]
    return SceneBinding(agent_names=agent_names, robot_config=robot_config)


def _raise_if_g1_scene_has_no_actuators(report, *, asset_path: str) -> None:
    if report.complete_matches or report.scene_names.actuators:
        return
    g1_like_joints = sorted(name for name in report.scene_names.joints if name.startswith("g1_") or "_hip_" in name)
    if not g1_like_joints:
        return
    sensors = sorted(report.scene_names.sensors)
    raise RuntimeError(
        "The current OrcaLab G1 scene contains joints but no motor actuators, so RL policy play cannot send ctrl. "
        f"Detected {len(g1_like_joints)} G1-like joints, 0 actuators, sensors={sensors}. "
        "Remove this non-actuated G1 from the OrcaLab layout and import/publish an XML-backed G1 asset with 29 "
        f"motor actuators, or run play with `--local-mujoco` for the local MJCF debug path. "
        f"Configured G1 asset_path={asset_path!r}."
    )


def publish_g1_scene(
    orcagym_addr: str,
    agent_name: str,
    asset_path: str = G1_AGENT_ASSET_PATH,
    agent_count: int = 1,
    extra_actors: list[dict] | None = None,
) -> None:
    _publish_unitree_scene(
        orcagym_addr=orcagym_addr,
        agent_name=agent_name,
        asset_path=asset_path,
        agent_count=agent_count,
        extra_actors=extra_actors,
    )


def publish_go2_scene(
    orcagym_addr: str,
    agent_name: str,
    asset_path: str = GO2_AGENT_ASSET_PATH,
    agent_count: int = 1,
    extra_actors: list[dict] | None = None,
) -> None:
    _publish_unitree_scene(
        orcagym_addr=orcagym_addr,
        agent_name=agent_name,
        asset_path=asset_path,
        agent_count=agent_count,
        extra_actors=extra_actors,
    )


def _publish_unitree_scene(
    *,
    orcagym_addr: str,
    agent_name: str,
    asset_path: str,
    agent_count: int,
    extra_actors: list[dict] | None = None,
) -> None:
    from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
    from orca_gym.utils.rotations import euler2quat
    import numpy as np

    temp_scene = OrcaGymScene(orcagym_addr)
    try:
        temp_scene.publish_scene()
        time.sleep(1)
    finally:
        temp_scene.close()
    time.sleep(1)

    scene = OrcaGymScene(orcagym_addr)
    try:
        grid_width = int(np.ceil(np.sqrt(max(1, agent_count))))
        spacing = 1.5
        x0 = -0.5 * spacing * (grid_width - 1)
        y0 = -0.5 * spacing * (grid_width - 1)
        for index in range(max(1, agent_count)):
            name = agent_name if agent_count == 1 else f"{agent_name.rsplit('_', 1)[0]}_{index:03d}"
            agent = Actor(
                name=name,
                asset_path=asset_path.replace("//", "/"),
                position=[x0 + spacing * (index % grid_width), y0 + spacing * (index // grid_width), 0],
                rotation=euler2quat([0, 0, 0]),
                scale=1.0,
            )
            scene.add_actor(agent)
        for extra_actor in extra_actors or []:
            scene.add_actor(
                Actor(
                    name=str(extra_actor["name"]),
                    asset_path=str(extra_actor["asset_path"]).replace("//", "/"),
                    position=np.asarray(extra_actor.get("position", [0.0, 0.0, 0.6]), dtype=np.float64),
                    rotation=euler2quat(extra_actor.get("rotation_euler", [0.0, 0.0, 0.0])),
                    scale=float(extra_actor.get("scale", 1.0)),
                )
            )
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
