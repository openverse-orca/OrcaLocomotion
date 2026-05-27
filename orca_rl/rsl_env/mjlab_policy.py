from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .math_utils import quat_wxyz_to_rotmat


_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_GO2_MJLAB_BASE_HEIGHT = 0.32
_G1_MJLAB_FOOT_FRICTION = 0.6
_GO2_MJLAB_FOOT_FRICTION = np.array([0.6, 0.02, 0.01], dtype=np.float64)
_GO2_MJLAB_FOOT_SOLIMP_HEAD = np.array([0.9, 0.95, 0.023], dtype=np.float64)


class MjlabRslRlActorPolicy(nn.Module):
    """Deterministic actor loader for Unitree/mjlab RSL-RL checkpoints."""

    def __init__(self, actor_state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        first_weight = actor_state_dict["mlp.0.weight"]
        self.input_dim = int(first_weight.shape[1])
        self.output_dim = int(actor_state_dict[_last_mlp_weight_key(actor_state_dict)].shape[0])

        self.register_buffer("obs_mean", actor_state_dict["obs_normalizer._mean"].clone().float())
        self.register_buffer("obs_std", actor_state_dict["obs_normalizer._std"].clone().float())
        self.layers = _build_mlp(actor_state_dict)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, device: str | torch.device = "cpu") -> "MjlabRslRlActorPolicy":
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Mjlab checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu")
        actor_state_dict = payload.get("actor_state_dict")
        if not isinstance(actor_state_dict, dict):
            raise ValueError(f"Checkpoint does not contain an actor_state_dict: {path}")
        policy = cls(actor_state_dict)
        policy.to(device)
        policy.eval()
        return policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.obs_mean) / (self.obs_std + 1.0e-2)
        return self.layers(obs)

    @torch.inference_mode()
    def act_numpy(self, obs: np.ndarray, *, device: str | torch.device | None = None) -> np.ndarray:
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        return self(obs_t).detach().cpu().numpy().astype(np.float32)


@dataclass(frozen=True)
class MjlabG1ActionSpec:
    scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    armature: np.ndarray
    frictionloss: np.ndarray
    joint_limits: np.ndarray
    neutral_qpos: np.ndarray
    arm_joint_mask: np.ndarray


@dataclass(frozen=True)
class MjlabGo2ActionSpec:
    scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    armature: np.ndarray
    frictionloss: np.ndarray
    joint_limits: np.ndarray
    default_qpos: np.ndarray


class MjlabG1OrcaPlayBridge:
    """Build mjlab G1 observations and apply mjlab G1 actions inside Orca envs."""

    def __init__(self, env: Any, *, expected_obs_dim: int, arm_mode: str = "neutral") -> None:
        self.env = env
        self.expected_obs_dim = int(expected_obs_dim)
        self.arm_mode = str(arm_mode)
        if self.arm_mode not in {"neutral", "policy"}:
            raise ValueError(f"Unsupported G1 arm mode: {arm_mode!r}")
        if self.expected_obs_dim < 98:
            raise ValueError(
                "Unitree/mjlab G1 actor observations are at least 98-D "
                f"(flat policy), got checkpoint input_dim={self.expected_obs_dim}."
            )
        self.action_spec = _load_unitree_mjlab_g1_action_spec(_g1_joint_names(env))
        self.alignment_report = _align_orca_g1_runtime_to_mjlab(env, self.action_spec)
        if self.arm_mode == "neutral":
            _initialize_g1_arms_neutral(env, self.action_spec)

    def get_observations(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        for task in self.env.tasks:
            foot_positions = task._query_all_foot_positions()
            contact_state = task._query_batched_contacts()
            for index in range(task.num_envs):
                agent = task.agents[index]
                state = task._read_state(
                    index,
                    update_air_time=False,
                    foot_contacts=contact_state["foot_contacts"][index],
                    foot_pos=foot_positions[index],
                )
                observations.append(
                    _build_mjlab_g1_actor_obs(
                        state=state,
                        base_ang_vel_body=_read_mjlab_imu_gyro(task, agent, index),
                        nominal_qpos=task._nominal_qpos[index],
                        episode_length=int(task.episode_lengths[index]),
                        step_dt=float(task.control_dt),
                        expected_dim=self.expected_obs_dim,
                    )
                )
        return np.stack(observations, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64).reshape(self.env.num_envs, self.env.num_actions)
        start = 0
        for task in self.env.tasks:
            stop = start + task.num_envs
            _step_task_with_mjlab_g1_actions(task, actions[start:stop], self.action_spec, arm_mode=self.arm_mode)
            self.env.episode_length_buf[start:stop] = torch.as_tensor(
                task.episode_lengths,
                dtype=torch.long,
                device=self.env.device,
            )
            start = stop
        return self.get_observations()


class MjlabGo2OrcaPlayBridge:
    """Build mjlab GO2 observations and apply mjlab GO2 actions inside Orca envs."""

    def __init__(self, env: Any, *, expected_obs_dim: int) -> None:
        self.env = env
        self.expected_obs_dim = int(expected_obs_dim)
        if self.expected_obs_dim < 47:
            raise ValueError(
                "Unitree/mjlab GO2 actor observations are at least 47-D "
                f"(flat policy), got checkpoint input_dim={self.expected_obs_dim}."
            )
        self.action_spec = _load_unitree_mjlab_go2_action_spec(_go2_joint_names(env))
        self.alignment_report = _align_orca_go2_runtime_to_mjlab(env, self.action_spec)

    def get_observations(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        for task in self.env.tasks:
            foot_positions = task._query_all_foot_positions()
            contact_state = task._query_batched_contacts()
            for index in range(task.num_envs):
                agent = task.agents[index]
                state = task._read_state(
                    index,
                    update_air_time=False,
                    foot_contacts=contact_state["foot_contacts"][index],
                    foot_pos=foot_positions[index],
                )
                observations.append(
                    _build_mjlab_go2_actor_obs(
                        state=state,
                        base_ang_vel_body=_read_mjlab_imu_gyro(task, agent, index),
                        default_qpos=self.action_spec.default_qpos,
                        episode_length=int(task.episode_lengths[index]),
                        step_dt=float(task.control_dt),
                        expected_dim=self.expected_obs_dim,
                    )
                )
        return np.stack(observations, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64).reshape(self.env.num_envs, self.env.num_actions)
        start = 0
        for task in self.env.tasks:
            stop = start + task.num_envs
            _step_task_with_mjlab_go2_actions(task, actions[start:stop], self.action_spec)
            self.env.episode_length_buf[start:stop] = torch.as_tensor(
                task.episode_lengths,
                dtype=torch.long,
                device=self.env.device,
            )
            start = stop
        return self.get_observations()


def make_mjlab_orca_play_bridge(
    env: Any,
    *,
    expected_obs_dim: int,
    robot: str | None = None,
    g1_arm_mode: str = "neutral",
) -> Any:
    robot_name = (robot or _infer_robot_name(env)).strip().lower()
    if robot_name == "g1":
        return MjlabG1OrcaPlayBridge(env, expected_obs_dim=expected_obs_dim, arm_mode=g1_arm_mode)
    if robot_name in {"go2", "unitree_go2"}:
        return MjlabGo2OrcaPlayBridge(env, expected_obs_dim=expected_obs_dim)
    raise ValueError(f"Unsupported Unitree/mjlab Orca play bridge robot: {robot_name!r}")


def find_latest_unitree_mjlab_checkpoint(
    project_root: str | Path,
    *,
    robot: str = "g1",
    terrain: str | None = None,
) -> Path:
    robot_name = str(robot).strip().lower()
    terrain_name = str(terrain or "flat").strip().lower()
    project_root = Path(project_root).expanduser().resolve()
    default_names = _unitree_mjlab_default_checkpoint_names(terrain_name)
    candidate_paths = []
    default_name = default_names.get(robot_name)
    if default_name:
        candidate_paths.extend(
            [
                project_root / "checkpoints" / default_name,
                project_root / default_name,
            ]
        )
    for path in candidate_paths:
        if path.exists():
            return path
    candidates = []
    root = project_root / "third_party" / "unitree_rl_mjlab" / "logs" / "rsl_rl"
    candidates.extend(root.glob(f"{robot_name}_velocity/*/model_*.pt"))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if not candidates:
        searched = [str(path) for path in candidate_paths]
        searched.append(str(root / f"{robot_name}_velocity" / "*" / "model_*.pt"))
        raise FileNotFoundError(
            f"Cannot find a Unitree/mjlab {robot_name.upper()} {terrain_name} checkpoint under "
            f"{searched}. Pass --checkpoint explicitly after training."
        )
    return candidates[-1]


def _unitree_mjlab_default_checkpoint_names(terrain: str) -> dict[str, str]:
    if terrain == "rough":
        return {
            "g1": "test_model_G1_mjlab_Rough.pt",
            "go2": "test_model_Go2_mjlab_Rough.pt",
            "unitree_go2": "test_model_Go2_mjlab_Rough.pt",
        }
    return {
        "g1": "test_model_G1_mjlab_Flat.pt",
        "go2": "test_model_Go2_mjlab_Flat.pt",
        "unitree_go2": "test_model_Go2_mjlab_Flat.pt",
    }


def _build_mlp(actor_state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
    weight_keys = sorted(
        (key for key in actor_state_dict if re.fullmatch(r"mlp\.\d+\.weight", key)),
        key=lambda key: int(key.split(".")[1]),
    )
    modules: list[nn.Module] = []
    for index, weight_key in enumerate(weight_keys):
        layer_id = weight_key.split(".")[1]
        bias_key = f"mlp.{layer_id}.bias"
        weight = actor_state_dict[weight_key]
        bias = actor_state_dict[bias_key]
        layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        layer.weight.data.copy_(weight.float())
        layer.bias.data.copy_(bias.float())
        modules.append(layer)
        if index != len(weight_keys) - 1:
            modules.append(nn.ELU())
    return nn.Sequential(*modules)


def _last_mlp_weight_key(actor_state_dict: dict[str, torch.Tensor]) -> str:
    return max(
        (key for key in actor_state_dict if re.fullmatch(r"mlp\.\d+\.weight", key)),
        key=lambda key: int(key.split(".")[1]),
    )


def _build_mjlab_g1_actor_obs(
    *,
    state: Any,
    base_ang_vel_body: np.ndarray | None = None,
    nominal_qpos: np.ndarray,
    episode_length: int,
    step_dt: float,
    expected_dim: int,
) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(state.base_quat)
    if base_ang_vel_body is None:
        base_ang_vel_body = rot.T @ state.base_ang_vel_world
    else:
        base_ang_vel_body = np.asarray(base_ang_vel_body, dtype=np.float64).reshape(3)
    projected_gravity = rot.T @ _GRAVITY_W
    phase = _mjlab_phase(
        episode_length=episode_length,
        step_dt=step_dt,
        period=0.6,
        command=state.command,
    )
    terms = [
        base_ang_vel_body,
        projected_gravity,
        state.command,
        phase,
        state.qpos - nominal_qpos,
        state.qvel,
        state.last_action,
    ]
    obs = np.concatenate(terms).astype(np.float64)
    if expected_dim > obs.size:
        height_scan = np.asarray(getattr(state, "height_scan", np.zeros(0)), dtype=np.float64).reshape(-1)
        if height_scan.size:
            height_scan = height_scan / 5.0
        obs = np.concatenate([obs, height_scan])
    if obs.size < expected_dim:
        obs = np.pad(obs, (0, expected_dim - obs.size))
    elif obs.size > expected_dim:
        obs = obs[:expected_dim]
    return obs.astype(np.float32)


def _build_mjlab_go2_actor_obs(
    *,
    state: Any,
    base_ang_vel_body: np.ndarray | None = None,
    default_qpos: np.ndarray,
    episode_length: int,
    step_dt: float,
    expected_dim: int,
) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(state.base_quat)
    if base_ang_vel_body is None:
        base_ang_vel_body = rot.T @ state.base_ang_vel_world
    else:
        base_ang_vel_body = np.asarray(base_ang_vel_body, dtype=np.float64).reshape(3)
    projected_gravity = rot.T @ _GRAVITY_W
    phase = _mjlab_phase(
        episode_length=episode_length,
        step_dt=step_dt,
        period=0.6,
        command=state.command,
    )
    terms = [
        base_ang_vel_body,
        projected_gravity,
        state.command,
        phase,
        state.qpos - default_qpos,
        state.qvel,
        state.last_action,
    ]
    obs = np.concatenate(terms).astype(np.float64)
    if expected_dim > obs.size:
        height_scan = np.asarray(getattr(state, "height_scan", np.zeros(0)), dtype=np.float64).reshape(-1)
        if height_scan.size:
            height_scan = height_scan / 5.0
        obs = np.concatenate([obs, height_scan])
    if obs.size < expected_dim:
        obs = np.pad(obs, (0, expected_dim - obs.size))
    elif obs.size > expected_dim:
        obs = obs[:expected_dim]
    return obs.astype(np.float32)


def _mjlab_phase(*, episode_length: int, step_dt: float, period: float, command: np.ndarray) -> np.ndarray:
    global_phase = (float(episode_length) * float(step_dt)) % period / period
    phase = np.array(
        [np.sin(global_phase * np.pi * 2.0), np.cos(global_phase * np.pi * 2.0)],
        dtype=np.float64,
    )
    if np.linalg.norm(command) < 0.1:
        phase[:] = 0.0
    return phase


def _read_mjlab_imu_gyro(task: Any, agent: Any, agent_index: int) -> np.ndarray | None:
    if not hasattr(task, "query_sensor_data"):
        return None
    candidates = []
    if hasattr(task, "sensor"):
        try:
            candidates.append(task.sensor("imu_gyro", agent_index))
        except Exception:
            pass
    candidates.extend(
        [
            f"{agent.agent_name}_imu_gyro",
            f"{agent.agent_name}/imu_gyro",
            "imu_gyro",
        ]
    )
    sensor_dict = getattr(task.model, "_sensor_dict", {})
    for sensor_name in dict.fromkeys(candidates):
        if sensor_dict and sensor_name not in sensor_dict:
            continue
        try:
            sensor_data = task.query_sensor_data([sensor_name])
            gyro = np.asarray(sensor_data[sensor_name], dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if gyro.size >= 3:
            return gyro[:3].copy()
    return None


def _count_mjlab_imu_gyro_sensors(task: Any) -> int:
    sensor_dict = getattr(task.model, "_sensor_dict", {})
    count = 0
    for index, agent in enumerate(getattr(task, "agents", [])):
        candidates = []
        if hasattr(task, "sensor"):
            try:
                candidates.append(task.sensor("imu_gyro", index))
            except Exception:
                pass
        candidates.append(f"{agent.agent_name}_imu_gyro")
        if any(sensor_name in sensor_dict for sensor_name in candidates):
            count += 1
    return count


def _step_task_with_mjlab_g1_actions(
    task: Any,
    actions: np.ndarray,
    spec: MjlabG1ActionSpec,
    *,
    arm_mode: str,
) -> None:
    actions = np.asarray(actions, dtype=np.float64).reshape(task.num_envs, task.num_actions)
    task._resample_commands()
    for index, agent in enumerate(task.agents):
        agent.previous_action = agent.last_action.copy()
        agent.last_action = actions[index].copy()

    target_qpos = task._nominal_qpos + actions * spec.scale.reshape(1, -1)
    if arm_mode == "neutral" and np.any(spec.arm_joint_mask):
        target_qpos[:, spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]

    task.ctrl[:] = 0.0
    if getattr(task, "_mjlab_position_actuator_aligned", False):
        task.ctrl[task._flat_actuator_ids] = target_qpos.reshape(-1)
    else:
        qpos = task.data.qpos[task._leg_qpos_indices]
        qvel = task.data.qvel[task._leg_qvel_indices]
        clipped_target = np.clip(target_qpos, task._joint_limit_low, task._joint_limit_high)
        torque = spec.kp.reshape(1, -1) * (clipped_target - qpos) - spec.kd.reshape(1, -1) * qvel
        torque = np.clip(torque, -spec.effort_limit.reshape(1, -1), spec.effort_limit.reshape(1, -1))
        task.ctrl[task._flat_actuator_ids] = torque.reshape(-1)
    if task._mjwarp_runtime is not None:
        for _ in range(task.decimation):
            task._mjwarp_runtime.set_ctrl(task.ctrl)
            task._mjwarp_runtime.step(nstep=task.frame_skip)
        task._mjwarp_runtime.sync_to_cpu(task.gym._mjData, recompute_contacts=True)
        task.update_data()
    else:
        for _ in range(task.decimation):
            task.set_ctrl(task.ctrl)
            task.mj_step(nstep=task.frame_skip)
        task.update_data()
    if task._render_mode == "human":
        task.render()
    task.episode_lengths += 1
    task._maybe_apply_push_disturbances()


def _step_task_with_mjlab_go2_actions(task: Any, actions: np.ndarray, spec: MjlabGo2ActionSpec) -> None:
    actions = np.asarray(actions, dtype=np.float64).reshape(task.num_envs, task.num_actions)
    task._resample_commands()
    for index, agent in enumerate(task.agents):
        agent.previous_action = agent.last_action.copy()
        agent.last_action = actions[index].copy()

    target_qpos = spec.default_qpos.reshape(1, -1) + actions * spec.scale.reshape(1, -1)

    task.ctrl[:] = 0.0
    if getattr(task, "_mjlab_position_actuator_aligned", False):
        task.ctrl[task._flat_actuator_ids] = target_qpos.reshape(-1)
    else:
        qpos = task.data.qpos[task._leg_qpos_indices]
        qvel = task.data.qvel[task._leg_qvel_indices]
        clipped_target = np.clip(target_qpos, task._joint_limit_low, task._joint_limit_high)
        torque = spec.kp.reshape(1, -1) * (clipped_target - qpos) - spec.kd.reshape(1, -1) * qvel
        torque = np.clip(torque, -spec.effort_limit.reshape(1, -1), spec.effort_limit.reshape(1, -1))
        task.ctrl[task._flat_actuator_ids] = torque.reshape(-1)
    if task._mjwarp_runtime is not None:
        for _ in range(task.decimation):
            task._mjwarp_runtime.set_ctrl(task.ctrl)
            task._mjwarp_runtime.step(nstep=task.frame_skip)
        task._mjwarp_runtime.sync_to_cpu(task.gym._mjData, recompute_contacts=True)
        task.update_data()
    else:
        for _ in range(task.decimation):
            task.set_ctrl(task.ctrl)
            task.mj_step(nstep=task.frame_skip)
        task.update_data()
    if task._render_mode == "human":
        task.render()
    task.episode_lengths += 1
    task._maybe_apply_push_disturbances()


def _g1_joint_names(env: Any) -> list[str]:
    if not env.tasks:
        raise ValueError("The Orca env has no tasks.")
    names = list(env.tasks[0].robot_config.get("leg_joint_names") or [])
    if len(names) != 29:
        raise ValueError("Unitree/mjlab play bridge currently supports the 29-DoF G1 action ABI only.")
    return names


def _go2_joint_names(env: Any) -> list[str]:
    if not env.tasks:
        raise ValueError("The Orca env has no tasks.")
    names = list(env.tasks[0].robot_config.get("leg_joint_names") or [])
    if len(names) != 12:
        raise ValueError("Unitree/mjlab GO2 play bridge expects the 12-DoF GO2 action ABI.")
    return names


def _infer_robot_name(env: Any) -> str:
    if not getattr(env, "tasks", None):
        raise ValueError("The Orca env has no tasks.")
    robot_name = str(getattr(env.tasks[0], "cfg", {}).get("robot", "")).strip().lower()
    if robot_name:
        return robot_name
    model_name = str(env.tasks[0].robot_config.get("model_name", "")).strip().lower()
    if model_name:
        return model_name
    raise ValueError("Cannot infer robot name for Unitree/mjlab play bridge.")


def _load_unitree_mjlab_g1_action_spec(joint_names: list[str]) -> MjlabG1ActionSpec:
    try:
        from src.assets.robots.unitree_g1 import g1_constants
    except ImportError as exc:
        raise ImportError(
            "Unitree/mjlab G1 action bridge requires the vendored unitree_rl_mjlab package. "
            "Install it with: python -m pip install -e third_party/unitree_rl_mjlab"
        ) from exc

    scale = _resolve_regex_values(g1_constants.G1_ACTION_SCALE, joint_names)
    kp = np.zeros(len(joint_names), dtype=np.float64)
    kd = np.zeros(len(joint_names), dtype=np.float64)
    effort = np.zeros(len(joint_names), dtype=np.float64)
    armature = np.zeros(len(joint_names), dtype=np.float64)
    frictionloss = np.zeros(len(joint_names), dtype=np.float64)
    joint_limits = _load_mjlab_g1_joint_limits(g1_constants, joint_names)
    for joint_index, joint_name in enumerate(joint_names):
        matched = False
        for actuator_cfg in g1_constants.G1_ARTICULATION.actuators:
            patterns = getattr(actuator_cfg, "target_names_expr", ())
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                kp[joint_index] = float(actuator_cfg.stiffness)
                kd[joint_index] = float(actuator_cfg.damping)
                effort[joint_index] = float(actuator_cfg.effort_limit)
                armature[joint_index] = float(getattr(actuator_cfg, "armature", 0.0))
                frictionloss[joint_index] = float(getattr(actuator_cfg, "frictionloss", 0.0))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot resolve Unitree/mjlab G1 actuator config for joint: {joint_name}")
    return MjlabG1ActionSpec(
        scale=scale,
        kp=kp,
        kd=kd,
        effort_limit=effort,
        armature=armature,
        frictionloss=frictionloss,
        joint_limits=joint_limits,
        neutral_qpos=_g1_dangling_arm_qpos(joint_names, joint_limits),
        arm_joint_mask=np.asarray([_is_g1_arm_joint(joint_name) for joint_name in joint_names], dtype=bool),
    )


def _load_unitree_mjlab_go2_action_spec(joint_names: list[str]) -> MjlabGo2ActionSpec:
    try:
        from src.assets.robots.unitree_go2 import go2_constants
    except ImportError as exc:
        raise ImportError(
            "Unitree/mjlab GO2 action bridge requires the vendored unitree_rl_mjlab package. "
            "Install it with: python -m pip install -e third_party/unitree_rl_mjlab"
        ) from exc

    scale = np.full(len(joint_names), 0.25, dtype=np.float64)
    kp = np.zeros(len(joint_names), dtype=np.float64)
    kd = np.zeros(len(joint_names), dtype=np.float64)
    effort = np.zeros(len(joint_names), dtype=np.float64)
    armature = np.zeros(len(joint_names), dtype=np.float64)
    frictionloss = np.zeros(len(joint_names), dtype=np.float64)
    joint_limits = _load_mjlab_go2_joint_limits(go2_constants, joint_names)
    default_qpos = _resolve_initial_joint_positions(go2_constants.INIT_STATE, joint_names)
    for joint_index, joint_name in enumerate(joint_names):
        matched = False
        for actuator_cfg in go2_constants.GO2_ARTICULATION.actuators:
            patterns = getattr(actuator_cfg, "target_names_expr", ())
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                kp[joint_index] = float(actuator_cfg.stiffness)
                kd[joint_index] = float(actuator_cfg.damping)
                effort[joint_index] = float(actuator_cfg.effort_limit)
                armature[joint_index] = float(getattr(actuator_cfg, "armature", 0.0))
                frictionloss[joint_index] = float(getattr(actuator_cfg, "frictionloss", 0.0))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot resolve Unitree/mjlab GO2 actuator config for joint: {joint_name}")
    return MjlabGo2ActionSpec(
        scale=scale,
        kp=kp,
        kd=kd,
        effort_limit=effort,
        armature=armature,
        frictionloss=frictionloss,
        joint_limits=joint_limits,
        default_qpos=default_qpos,
    )


def _load_mjlab_g1_joint_limits(g1_constants: Any, joint_names: list[str]) -> np.ndarray:
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("Unitree/mjlab G1 alignment requires the mujoco Python package.") from exc

    model = g1_constants.get_spec().compile()
    limits = np.zeros((len(joint_names), 2), dtype=np.float64)
    for index, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Cannot find Unitree/mjlab G1 joint in source XML: {joint_name}")
        limits[index] = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    return limits


def _is_g1_arm_joint(joint_name: str) -> bool:
    return any(part in str(joint_name) for part in ("shoulder", "elbow", "wrist"))


def _g1_dangling_arm_qpos(joint_names: list[str], joint_limits: np.ndarray) -> np.ndarray:
    qpos = np.zeros(len(joint_names), dtype=np.float64)
    for index, joint_name in enumerate(joint_names):
        if "elbow" in str(joint_name):
            low, high = np.asarray(joint_limits[index], dtype=np.float64)
            qpos[index] = float(np.clip(0.2, low, high))
    return qpos


def _initialize_g1_arms_neutral(env: Any, spec: MjlabG1ActionSpec) -> None:
    if not np.any(spec.arm_joint_mask):
        return
    for task in getattr(env, "tasks", []):
        if hasattr(task, "_nominal_qpos"):
            task._nominal_qpos[:, spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]
        joint_qpos: dict[str, np.ndarray] = {}
        joint_qvel: dict[str, np.ndarray] = {}
        for agent in getattr(task, "agents", []):
            if hasattr(agent, "nominal_qpos"):
                agent.nominal_qpos[spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]
            if hasattr(agent, "obs_builder"):
                agent.obs_builder.nominal_qpos[spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]
            if hasattr(agent, "reward_manager"):
                agent.reward_manager.nominal_qpos[spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]
            if hasattr(agent, "action_mapper"):
                agent.action_mapper.nominal_qpos[spec.arm_joint_mask] = spec.neutral_qpos[spec.arm_joint_mask]
            for joint_index, joint_name in enumerate(agent.leg_joint_names):
                if bool(spec.arm_joint_mask[joint_index]):
                    joint_qpos[joint_name] = np.array([spec.neutral_qpos[joint_index]], dtype=np.float64)
                    joint_qvel[joint_name] = np.zeros(1, dtype=np.float64)
        if joint_qpos:
            task.set_joint_qpos(joint_qpos)
            task.set_joint_qvel(joint_qvel)
            task.mj_forward()
            task.update_data()


def _load_mjlab_go2_joint_limits(go2_constants: Any, joint_names: list[str]) -> np.ndarray:
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("Unitree/mjlab GO2 alignment requires the mujoco Python package.") from exc

    model = go2_constants.get_spec().compile()
    limits = np.zeros((len(joint_names), 2), dtype=np.float64)
    for index, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Cannot find Unitree/mjlab GO2 joint in source XML: {joint_name}")
        limits[index] = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    return limits


def _resolve_initial_joint_positions(initial_state: Any, joint_names: list[str]) -> np.ndarray:
    joint_pos = getattr(initial_state, "joint_pos", None)
    if not isinstance(joint_pos, dict):
        raise ValueError("Unitree/mjlab initial state does not expose a joint_pos dict.")
    values = np.zeros(len(joint_names), dtype=np.float64)
    for index, joint_name in enumerate(joint_names):
        for pattern, value in joint_pos.items():
            if re.fullmatch(pattern, joint_name):
                values[index] = float(value)
                break
    return values


def _align_orca_g1_runtime_to_mjlab(env: Any, spec: MjlabG1ActionSpec) -> dict[str, int]:
    report = {
        "tasks": 0,
        "agents": 0,
        "joints": 0,
        "actuators": 0,
        "position_actuator_tasks": 0,
        "imu_gyro_sensors": 0,
        "contact_geoms": 0,
        "foot_contact_geoms": 0,
        "nonfoot_contact_geoms": 0,
    }
    for task in getattr(env, "tasks", []):
        model = _task_mujoco_model(task)
        if model is None:
            continue
        if getattr(task, "_mjwarp_runtime", None) is not None:
            print(
                "[orca_rl.play] mjlab alignment skipped for an mjwarp task; "
                "play alignment currently targets the CPU MuJoCo model."
            )
            continue
        report["tasks"] += 1
        report["imu_gyro_sensors"] += _count_mjlab_imu_gyro_sensors(task)
        contact_report = _align_g1_contact_model_to_mjlab(model, task)
        for key, value in contact_report.items():
            report[key] += value
        for agent in task.agents:
            _align_agent_to_mjlab_position_actuators(model, task, agent, spec)
            agent.joint_limits = spec.joint_limits.copy()
            agent.torque_limits = np.stack([-spec.effort_limit, spec.effort_limit], axis=1)
            report["agents"] += 1
            report["joints"] += len(agent.leg_joint_names)
            report["actuators"] += len(agent.actuator_names)
        task._joint_limit_low = np.broadcast_to(spec.joint_limits[:, 0], task._joint_limit_low.shape).copy()
        task._joint_limit_high = np.broadcast_to(spec.joint_limits[:, 1], task._joint_limit_high.shape).copy()
        task._torque_low = np.broadcast_to(-spec.effort_limit, task._torque_low.shape).copy()
        task._torque_high = np.broadcast_to(spec.effort_limit, task._torque_high.shape).copy()
        task._mjlab_position_actuator_aligned = True
        report["position_actuator_tasks"] += 1
        task.mj_forward()
        task.update_data()
    return report


def _align_orca_go2_runtime_to_mjlab(env: Any, spec: MjlabGo2ActionSpec) -> dict[str, int]:
    report = {
        "tasks": 0,
        "agents": 0,
        "joints": 0,
        "actuators": 0,
        "position_actuator_tasks": 0,
        "contact_geoms": 0,
        "foot_contact_geoms": 0,
        "nonfoot_contact_geoms": 0,
        "base_height_resets": 0,
        "imu_gyro_sensors": 0,
    }
    for task in getattr(env, "tasks", []):
        model = _task_mujoco_model(task)
        if model is None:
            continue
        if getattr(task, "_mjwarp_runtime", None) is not None:
            print(
                "[orca_rl.play] mjlab GO2 alignment skipped for an mjwarp task; "
                "play alignment currently targets the CPU MuJoCo model."
            )
            continue
        report["tasks"] += 1
        contact_report = _align_go2_contact_model_to_mjlab(model, task)
        for key, value in contact_report.items():
            report[key] += value
        report["imu_gyro_sensors"] += _count_mjlab_imu_gyro_sensors(task)
        for agent in task.agents:
            _align_agent_to_mjlab_position_actuators(model, task, agent, spec)
            agent.joint_limits = spec.joint_limits.copy()
            agent.torque_limits = np.stack([-spec.effort_limit, spec.effort_limit], axis=1)
            agent.nominal_qpos = spec.default_qpos.copy()
            report["agents"] += 1
            report["joints"] += len(agent.leg_joint_names)
            report["actuators"] += len(agent.actuator_names)
        task._nominal_qpos = np.broadcast_to(spec.default_qpos, task._nominal_qpos.shape).copy()
        task._joint_limit_low = np.broadcast_to(spec.joint_limits[:, 0], task._joint_limit_low.shape).copy()
        task._joint_limit_high = np.broadcast_to(spec.joint_limits[:, 1], task._joint_limit_high.shape).copy()
        task._torque_low = np.broadcast_to(-spec.effort_limit, task._torque_low.shape).copy()
        task._torque_high = np.broadcast_to(spec.effort_limit, task._torque_high.shape).copy()
        task._mjlab_position_actuator_aligned = True
        report["position_actuator_tasks"] += 1
        report["base_height_resets"] += _align_go2_base_height_to_mjlab(task)
        task.mj_forward()
        task.update_data()
    return report


def _align_agent_to_mjlab_position_actuators(
    model: Any,
    task: Any,
    agent: Any,
    spec: MjlabG1ActionSpec | MjlabGo2ActionSpec,
) -> None:
    import mujoco

    joint_dict = task.model.get_joint_dict()
    actuator_dict = task.model.get_actuator_dict()
    for joint_index, (joint_name, actuator_name) in enumerate(zip(agent.leg_joint_names, agent.actuator_names)):
        joint_id = int(joint_dict[joint_name]["JointId"])
        actuator_id = int(actuator_dict[actuator_name]["ActuatorId"])
        qvel_offset = int(agent.leg_qvel_indices[joint_index])
        low, high = spec.joint_limits[joint_index]
        kp = float(spec.kp[joint_index])
        kd = float(spec.kd[joint_index])
        effort = float(spec.effort_limit[joint_index])

        model.jnt_range[joint_id] = (low, high)
        model.jnt_limited[joint_id] = True
        model.dof_armature[qvel_offset] = float(spec.armature[joint_index])
        model.dof_damping[qvel_offset] = 0.0
        model.dof_frictionloss[qvel_offset] = float(spec.frictionloss[joint_index])

        model.actuator_dyntype[actuator_id] = int(mujoco.mjtDyn.mjDYN_NONE)
        model.actuator_gaintype[actuator_id] = int(mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_biastype[actuator_id] = int(mujoco.mjtBias.mjBIAS_AFFINE)
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcelimited[actuator_id] = True
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_ctrllimited[actuator_id] = False
        model.actuator_actlimited[actuator_id] = False
        delta = effort / kp if kp > 0.0 else effort
        model.actuator_ctrlrange[actuator_id] = (low - delta, high + delta)

        _update_orca_model_dicts(task, joint_name, actuator_name, joint_id, actuator_id, low, high)


def _align_g1_contact_model_to_mjlab(model: Any, task: Any) -> dict[str, int]:
    import mujoco

    report = {
        "contact_geoms": 0,
        "foot_contact_geoms": 0,
        "nonfoot_contact_geoms": 0,
    }
    foot_body_names = {body for agent in task.agents for body in agent.foot_body_names}
    robot_body_names = {body for agent in task.agents for body in agent.robot_body_names}
    agent_prefixes = tuple(f"{agent.agent_name}_" for agent in task.agents)
    sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)

    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name not in robot_body_names and not body_name.startswith(agent_prefixes):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue

        report["contact_geoms"] += 1
        if _is_g1_runtime_foot_geom(model, geom_id, body_name, foot_body_names, sphere_type):
            model.geom_condim[geom_id] = 3
            model.geom_priority[geom_id] = max(int(model.geom_priority[geom_id]), 1)
            model.geom_friction[geom_id, 0] = _G1_MJLAB_FOOT_FRICTION
            report["foot_contact_geoms"] += 1
        else:
            model.geom_condim[geom_id] = 1
            report["nonfoot_contact_geoms"] += 1

    return report


def _is_g1_runtime_foot_geom(
    model: Any,
    geom_id: int,
    body_name: str,
    foot_body_names: set[str],
    sphere_type: int,
) -> bool:
    if body_name not in foot_body_names:
        return False
    if int(model.geom_type[geom_id]) != sphere_type:
        return False
    radius = float(model.geom_size[geom_id, 0])
    local_z = float(model.geom_pos[geom_id, 2])
    return 0.001 <= radius <= 0.03 and local_z < 0.0


def _align_go2_contact_model_to_mjlab(model: Any, task: Any) -> dict[str, int]:
    import mujoco

    report = {
        "contact_geoms": 0,
        "foot_contact_geoms": 0,
        "nonfoot_contact_geoms": 0,
    }
    foot_body_names = {body for agent in task.agents for body in agent.foot_body_names}
    robot_body_names = {body for agent in task.agents for body in agent.robot_body_names}
    agent_prefixes = tuple(f"{agent.agent_name}_" for agent in task.agents)
    sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)

    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name not in robot_body_names and not body_name.startswith(agent_prefixes):
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue

        report["contact_geoms"] += 1
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 0

        if _is_go2_runtime_foot_geom(model, geom_id, body_name, foot_body_names, sphere_type):
            model.geom_condim[geom_id] = 3
            model.geom_priority[geom_id] = max(int(model.geom_priority[geom_id]), 1)
            model.geom_friction[geom_id, : _GO2_MJLAB_FOOT_FRICTION.size] = _GO2_MJLAB_FOOT_FRICTION
            solimp = np.asarray(model.geom_solimp[geom_id], dtype=np.float64).copy()
            solimp[: _GO2_MJLAB_FOOT_SOLIMP_HEAD.size] = _GO2_MJLAB_FOOT_SOLIMP_HEAD
            model.geom_solimp[geom_id] = solimp
            report["foot_contact_geoms"] += 1
        else:
            model.geom_condim[geom_id] = 1
            report["nonfoot_contact_geoms"] += 1

    return report


def _is_go2_runtime_foot_geom(
    model: Any,
    geom_id: int,
    body_name: str,
    foot_body_names: set[str],
    sphere_type: int,
) -> bool:
    if body_name not in foot_body_names:
        return False
    if int(model.geom_type[geom_id]) != sphere_type:
        return False
    radius = float(model.geom_size[geom_id, 0])
    local_z = float(model.geom_pos[geom_id, 2])
    return 0.015 <= radius <= 0.035 and local_z < -0.15


def _align_go2_base_height_to_mjlab(task: Any) -> int:
    reset_cfg = task.cfg.setdefault("reset", {})
    reset_cfg["base_height"] = _GO2_MJLAB_BASE_HEIGHT
    reset_cfg["xy_noise"] = 0.0
    reset_cfg["yaw_noise"] = 0.0
    reset_cfg["joint_noise"] = 0.0

    joint_qpos: dict[str, np.ndarray] = {}
    joint_qvel: dict[str, np.ndarray] = {}
    changed = 0
    for agent in task.agents:
        base_qpos = np.asarray(
            task.data.qpos[agent.base_qpos_offset : agent.base_qpos_offset + 7],
            dtype=np.float64,
        ).copy()
        if base_qpos.size != 7:
            continue
        if abs(float(base_qpos[2]) - _GO2_MJLAB_BASE_HEIGHT) > 1e-9:
            changed += 1
        base_qpos[2] = _GO2_MJLAB_BASE_HEIGHT
        agent.initial_base_qpos[2] = _GO2_MJLAB_BASE_HEIGHT
        joint_qpos[agent.base_joint_name] = base_qpos
        joint_qvel[agent.base_joint_name] = np.zeros(6, dtype=np.float64)

    if joint_qpos:
        task.set_joint_qpos(joint_qpos)
        task.set_joint_qvel(joint_qvel)
    return changed


def _update_orca_model_dicts(
    task: Any,
    joint_name: str,
    actuator_name: str,
    joint_id: int,
    actuator_id: int,
    joint_low: float,
    joint_high: float,
) -> None:
    joint_dict = task.model.get_joint_dict()
    actuator_dict = task.model.get_actuator_dict()
    if joint_name in joint_dict:
        joint_dict[joint_name]["JointId"] = joint_id
        joint_dict[joint_name]["Range"] = np.array([joint_low, joint_high], dtype=np.float64)
    if actuator_name in actuator_dict:
        model = _task_mujoco_model(task)
        if model is not None:
            actuator_dict[actuator_name]["ActuatorId"] = actuator_id
            actuator_dict[actuator_name]["CtrlLimited"] = False
            actuator_dict[actuator_name]["ForceLimited"] = True
            actuator_dict[actuator_name]["CtrlRange"] = model.actuator_ctrlrange[actuator_id].copy()
            actuator_dict[actuator_name]["ForceRange"] = model.actuator_forcerange[actuator_id].copy()
            actuator_dict[actuator_name]["GainPrm"] = model.actuator_gainprm[actuator_id].copy()
            actuator_dict[actuator_name]["BiasPrm"] = model.actuator_biasprm[actuator_id].copy()


def _task_mujoco_model(task: Any) -> Any | None:
    gym = getattr(task, "gym", None)
    return getattr(gym, "_mjModel", None)


def _resolve_regex_values(pattern_values: dict[str, float], names: list[str]) -> np.ndarray:
    values = np.zeros(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        for pattern, value in pattern_values.items():
            if re.fullmatch(pattern, name):
                values[index] = float(value)
                break
        else:
            raise ValueError(f"Cannot resolve Unitree/mjlab action scale for joint: {name}")
    return values
