from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .math_utils import quat_wxyz_to_rotmat, yaw_quat_wxyz


_DEFAULT_COMMAND_ASSET_PATH = "assets/001d46537b9e555b/commandarrow/prefabs/command_arrow_usda"
_DEFAULT_HEADING_ASSET_PATH = "assets/001d46537b9e555b/heading_arrow/prefabs/heading_arrow_usda"
_DEFAULT_COMMAND_ACTOR_NAME = "cmd_arrow_000"
_DEFAULT_HEADING_ACTOR_NAME = "heading_arrow_000"


@dataclass(frozen=True)
class CommandArrowConfig:
    enabled: bool = False
    asset_path: str = _DEFAULT_COMMAND_ASSET_PATH
    actor_name: str = _DEFAULT_COMMAND_ACTOR_NAME
    scale: float = 0.55
    joint_name: str | None = None
    mode: str = "command"
    agent_index: int = 0
    z_offset: float = 0.45
    forward_offset: float = 0.0
    lateral_offset: float = 0.0
    tail_local_x: float = -0.25
    min_linear_speed: float = 0.03
    yaw_when_standing: bool = True
    yaw_arrow_radius: float = 0.35
    qpos_log_interval: int = 240

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "CommandArrowConfig":
        raw = dict(cfg or {})
        mode = str(raw.get("mode") or "command").strip().lower()
        uses_heading_asset = mode in {"heading", "velocity"}
        default_asset_path = _DEFAULT_HEADING_ASSET_PATH if uses_heading_asset else _DEFAULT_COMMAND_ASSET_PATH
        default_actor_name = _DEFAULT_HEADING_ACTOR_NAME if uses_heading_asset else _DEFAULT_COMMAND_ACTOR_NAME
        return cls(
            enabled=bool(raw.get("enabled", False)),
            asset_path=str(raw.get("asset_path") or default_asset_path),
            actor_name=str(raw.get("actor_name") or default_actor_name),
            scale=float(raw.get("scale", 0.5 if uses_heading_asset else 0.55)),
            joint_name=_none_if_empty(raw.get("joint_name")),
            mode=mode,
            agent_index=int(raw.get("agent_index", 0)),
            z_offset=float(raw.get("z_offset", 0.45)),
            forward_offset=float(raw.get("forward_offset", 0.0)),
            lateral_offset=float(raw.get("lateral_offset", 0.0)),
            tail_local_x=float(raw.get("tail_local_x", -0.25)),
            min_linear_speed=float(raw.get("min_linear_speed", 0.03)),
            yaw_when_standing=bool(raw.get("yaw_when_standing", True)),
            yaw_arrow_radius=float(raw.get("yaw_arrow_radius", 0.35)),
            qpos_log_interval=max(1, int(raw.get("qpos_log_interval", 240))),
        )


class CommandArrowVisualizer:
    """Drive a pre-placed OrcaLab command arrow actor through a freejoint.

    Asset protocol:
    - The arrow mesh points along local +X.
    - The actor has a freejoint that can be updated through OrcaGym set_joint_qpos.
    - Collision is disabled, or at least the visual body does not collide with robot/terrain.
    """

    def __init__(self, task: Any, cfg: CommandArrowConfig) -> None:
        self.task = task
        self.cfg = cfg
        self.joint_name = self._resolve_joint_name()
        self.agent_index = self._resolve_agent_index()
        self._frame = 0
        self._warned_missing = False
        self._warned_agent = False

    @property
    def available(self) -> bool:
        return self.joint_name is not None and self.agent_index is not None

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "available": self.available,
            "asset_path": self.cfg.asset_path,
            "actor_name": self.cfg.actor_name,
            "scale": self.cfg.scale,
            "joint_name": self.joint_name,
            "agent_index": self.agent_index,
            "mode": self.cfg.mode,
        }

    def update(self) -> None:
        if not self.cfg.enabled:
            return
        self._frame += 1
        if self.joint_name is None:
            self._warn_missing_joint()
            return
        if self.agent_index is None:
            self._warn_missing_agent()
            return

        agent = self.task.agents[self.agent_index]
        base_qpos = np.asarray(
            self.task.data.qpos[agent.base_qpos_offset : agent.base_qpos_offset + 7],
            dtype=np.float64,
        )
        if base_qpos.size != 7:
            return

        base_qvel = np.asarray(
            self.task.data.qvel[agent.base_qvel_offset : agent.base_qvel_offset + 6],
            dtype=np.float64,
        )
        qpos = self._build_arrow_qpos(base_qpos=base_qpos, base_qvel=base_qvel, command=agent.command)
        self.task.set_joint_qpos({self.joint_name: qpos})
        try:
            self.task.set_joint_qvel({self.joint_name: np.zeros(6, dtype=np.float64)})
        except Exception:
            pass
        self.task.update_data()

    def _build_arrow_qpos(
        self,
        *,
        base_qpos: np.ndarray,
        base_qvel: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        base_pos = base_qpos[:3]
        base_quat = base_qpos[3:7]
        base_rot = quat_wxyz_to_rotmat(base_quat)
        base_yaw = math.atan2(float(base_rot[1, 0]), float(base_rot[0, 0]))
        if self.cfg.mode == "heading":
            heading_world = base_yaw
            offset_body = np.array([self.cfg.forward_offset, self.cfg.lateral_offset, 0.0], dtype=np.float64)
        elif self.cfg.mode == "velocity":
            velocity_xy_world = np.asarray(base_qvel[:2], dtype=np.float64)
            velocity_speed = float(np.linalg.norm(velocity_xy_world))
            if velocity_speed >= self.cfg.min_linear_speed:
                heading_world = math.atan2(float(velocity_xy_world[1]), float(velocity_xy_world[0]))
            else:
                heading_world = base_yaw
            offset_body = np.array([self.cfg.forward_offset, self.cfg.lateral_offset, 0.0], dtype=np.float64)
        else:
            command = np.asarray(command, dtype=np.float64).reshape(3)
            local_xy = np.array([command[0], command[1]], dtype=np.float64)
            local_speed = float(np.linalg.norm(local_xy))
            if local_speed >= self.cfg.min_linear_speed:
                heading_body = math.atan2(float(local_xy[1]), float(local_xy[0]))
                heading_world = base_yaw + heading_body
                offset_body = np.array([self.cfg.forward_offset, self.cfg.lateral_offset, 0.0], dtype=np.float64)
            elif self.cfg.yaw_when_standing and abs(float(command[2])) >= self.cfg.min_linear_speed:
                heading_body = math.copysign(math.pi * 0.5, float(command[2]))
                heading_world = base_yaw + heading_body
                offset_body = np.array([self.cfg.yaw_arrow_radius, 0.0, 0.0], dtype=np.float64)
            else:
                heading_world = base_yaw
                offset_body = np.array([self.cfg.forward_offset, self.cfg.lateral_offset, 0.0], dtype=np.float64)

        offset_world = base_rot @ offset_body
        tail_target = base_pos + offset_world + np.array([0.0, 0.0, self.cfg.z_offset], dtype=np.float64)
        heading_quat = yaw_quat_wxyz(heading_world)
        heading_rot = quat_wxyz_to_rotmat(heading_quat)
        position = tail_target - heading_rot @ np.array([self.cfg.tail_local_x, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([position, heading_quat]).astype(np.float64)

    def _resolve_agent_index(self) -> int | None:
        if not getattr(self.task, "agents", None):
            return None
        index = int(self.cfg.agent_index)
        if 0 <= index < len(self.task.agents):
            return index
        return None

    def _resolve_joint_name(self) -> str | None:
        joint_dict = self.task.model.get_joint_dict()
        candidates = []
        if self.cfg.joint_name:
            candidates.append(self.cfg.joint_name)
        actor = self.cfg.actor_name
        candidates.extend(
            [
                actor,
                f"{actor}_joint",
                f"{actor}_freejoint",
                f"{actor}_base_joint",
                f"{actor}_floating_base_joint",
                f"{actor}_root",
                f"{actor}_root_joint",
                "command_arrow",
                "command_arrow_joint",
                "command_arrow_freejoint",
                "arrow_x",
                "arrow_x_joint",
                "arrow_x_freejoint",
                "heading_arrow",
                "heading_arrow_joint",
                "heading_arrow_freejoint",
            ]
        )
        for candidate in dict.fromkeys(candidates):
            if candidate in joint_dict and self._is_freejoint(candidate):
                return candidate

        needle_parts = tuple(
            part
            for part in (
                self.cfg.actor_name.lower(),
                "cmd_arrow" if self.cfg.mode == "command" else "",
                "command_arrow" if self.cfg.mode == "command" else "",
                "heading_arrow" if self.cfg.mode in {"heading", "velocity"} else "",
                "arrow_x",
            )
            if part
        )
        for joint_name in joint_dict:
            lowered = joint_name.lower()
            if any(part in lowered for part in needle_parts) and self._is_freejoint(joint_name):
                return joint_name
        return None

    def _is_freejoint(self, joint_name: str) -> bool:
        try:
            qpos_lengths, qvel_lengths, _ = self.task.query_joint_lengths([joint_name])
        except Exception:
            return False
        return int(qpos_lengths[0]) == 7 and int(qvel_lengths[0]) == 6

    def _warn_missing_joint(self) -> None:
        if self._warned_missing or self._frame % self.cfg.qpos_log_interval != 1:
            return
        self._warned_missing = True
        print(
            f"[orca_rl.debug] {self.cfg.mode} arrow enabled but no freejoint was found. "
            f"Expected actor/joint around {self.cfg.actor_name!r}. "
            f"Place asset {self.cfg.asset_path!r} in the OrcaLab scene with a freejoint, "
            "or pass --command-arrow-joint."
        )

    def _warn_missing_agent(self) -> None:
        if self._warned_agent:
            return
        self._warned_agent = True
        print(
            "[orca_rl.debug] Command arrow enabled but agent_index is out of range: "
            f"{self.cfg.agent_index}."
        )


def maybe_make_command_arrow_visualizer(task: Any, cfg: dict[str, Any] | None) -> CommandArrowVisualizer | None:
    arrow_cfg = CommandArrowConfig.from_mapping(cfg)
    if not arrow_cfg.enabled:
        return None
    return CommandArrowVisualizer(task, arrow_cfg)


def make_debug_arrow_visualizers(task: Any, debug_cfg: dict[str, Any] | None) -> list[CommandArrowVisualizer]:
    if not isinstance(debug_cfg, dict):
        return []
    visualizers = []
    for key in ("command_arrow", "heading_arrow"):
        arrow_cfg = CommandArrowConfig.from_mapping(debug_cfg.get(key))
        if arrow_cfg.enabled:
            visualizers.append(CommandArrowVisualizer(task, arrow_cfg))
    return visualizers


def ensure_command_arrow_actor(
    *,
    orcagym_addr: str,
    actor_name: str = _DEFAULT_COMMAND_ACTOR_NAME,
    asset_path: str = _DEFAULT_COMMAND_ASSET_PATH,
) -> bool:
    """Publish the command arrow actor when it is not already in the OrcaLab scene."""

    from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
    from orca_gym.utils.rotations import euler2quat

    from .model_scanner import probe_scene_model

    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=0.005)
    actor_key = actor_name.lower()
    existing_names = set().union(
        scene_names.joints,
        scene_names.bodies,
        scene_names.sites,
        scene_names.sensors,
    )
    if any(actor_key in name.lower() for name in existing_names):
        return False

    scene = OrcaGymScene(orcagym_addr)
    try:
        scene.add_actor(
            Actor(
                name=actor_name,
                asset_path=asset_path.replace("//", "/"),
                position=np.array([0.0, 0.0, 0.6], dtype=np.float64),
                rotation=euler2quat([0.0, 0.0, 0.0]),
                scale=1.0,
            )
        )
        scene.publish_scene()
    finally:
        scene.close()
    time.sleep(1.0)
    return True


def default_command_arrow_asset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets/debug/command_arrow.usdz"


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
