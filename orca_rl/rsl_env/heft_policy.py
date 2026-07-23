from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort


HEFT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

HEFT_DEFAULT_QPOS = np.asarray(
    [
        -0.28, -0.28, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.5, 0.5,
        0.35, 0.35,
        -0.23, -0.23,
        0.16, -0.16,
        0.0, 0.0,
        0.0, 0.0,
        0.87, 0.87,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
    ],
    dtype=np.float64,
)

HEFT_ACTION_SCALE = np.asarray(
    [0.5] * 11
    + [1.0, 1.0]
    + [0.5, 0.5]
    + [1.0, 1.0]
    + [0.5, 0.5]
    + [1.0] * 10,
    dtype=np.float64,
)

HEFT_KP = np.asarray(
    [
        99.09842777666113, 99.09842777666113, 40.17923847137318,
        99.09842777666113, 99.09842777666113, 28.50124619574858,
        40.17923847137318, 40.17923847137318, 28.50124619574858,
        99.09842777666113, 99.09842777666113,
        14.25062309787429, 14.25062309787429,
        28.50124619574858, 28.50124619574858,
        14.25062309787429, 14.25062309787429,
        28.50124619574858, 28.50124619574858,
        14.25062309787429, 14.25062309787429,
        14.25062309787429, 14.25062309787429,
        14.25062309787429, 14.25062309787429,
        8.611032447370201, 8.611032447370201,
        8.611032447370201, 8.611032447370201,
    ],
    dtype=np.float64,
)

HEFT_KD = np.asarray(
    [
        6.3088018534966395, 6.3088018534966395, 2.5578897650279457,
        6.3088018534966395, 6.3088018534966395, 1.814445686584846,
        2.5578897650279457, 2.5578897650279457, 1.814445686584846,
        6.3088018534966395, 6.3088018534966395,
        0.907222843292423, 0.907222843292423,
        1.814445686584846, 1.814445686584846,
        0.907222843292423, 0.907222843292423,
        1.814445686584846, 1.814445686584846,
        0.907222843292423, 0.907222843292423,
        0.907222843292423, 0.907222843292423,
        0.907222843292423, 0.907222843292423,
        0.548195351665136, 0.548195351665136,
        0.548195351665136, 0.548195351665136,
    ],
    dtype=np.float64,
)

HEFT_ARMATURE = np.asarray(
    [
        0.01017752, 0.01017752, 0.01017752,
        0.025101925, 0.025101925, 0.00721945,
        0.01017752, 0.01017752, 0.00721945,
        0.025101925, 0.025101925,
        0.003609725, 0.003609725,
        0.00721945, 0.00721945,
        0.003609725, 0.003609725,
        0.00721945, 0.00721945,
        0.003609725, 0.003609725,
        0.003609725, 0.003609725,
        0.003609725, 0.003609725,
        0.00425, 0.00425, 0.00425, 0.00425,
    ],
    dtype=np.float64,
)

HEFT_EFFORT_LIMIT = np.asarray(
    [
        88.0, 88.0, 88.0,
        139.0, 139.0, 35.0,
        88.0, 88.0, 35.0,
        139.0, 139.0,
        25.0, 25.0,
        35.0, 35.0,
        25.0, 25.0,
        35.0, 35.0,
        25.0, 25.0,
        25.0, 25.0,
        25.0, 25.0,
        5.0, 5.0, 5.0, 5.0,
    ],
    dtype=np.float64,
)

HEFT_FUTURE_STEPS = np.asarray([0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16], dtype=np.int64)
HEFT_HISTORY_STEPS = np.asarray([0, 1, 2, 3, 4, 8, 12, 16, 20], dtype=np.int64)
HEFT_PREV_ACTION_STEPS = 8
HEFT_OBSERVATION_DIM = 1729


@dataclass(frozen=True)
class HeftMotion:
    joint_pos: np.ndarray
    root_quat: np.ndarray
    root_pos: np.ndarray


class HeftMotionLibrary:
    def __init__(self, motion_dir: str | Path) -> None:
        self.motion_dir = Path(motion_dir).expanduser().resolve()
        self._motions: dict[str, HeftMotion] = {
            "default": HeftMotion(
                joint_pos=HEFT_DEFAULT_QPOS.reshape(1, -1).astype(np.float32),
                root_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                root_pos=np.asarray([[0.0, 0.0, 0.78]], dtype=np.float32),
            )
        }
        self.add_directory(self.motion_dir)

    def add_directory(self, motion_dir: str | Path) -> None:
        directory = Path(motion_dir).expanduser().resolve()
        for path in sorted(directory.glob("*.npz")):
            self._motions[path.stem] = self._load_npz(path)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._motions)

    def get(self, name: str) -> HeftMotion:
        try:
            return self._motions[str(name)]
        except KeyError as exc:
            raise ValueError(f"Unknown HEFT motion {name!r}; available={list(self._motions)}") from exc

    @staticmethod
    def _load_npz(path: Path) -> HeftMotion:
        with np.load(path, allow_pickle=True) as data:
            fps = float(np.asarray(data["fps"]).reshape(()))
            if not np.isclose(fps, 50.0):
                raise ValueError(f"HEFT motion must be 50 Hz, got {fps:g} Hz: {path}")
            source_names = [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                for item in data["joint_names"].tolist()
            ]
            source_index = {name: index for index, name in enumerate(source_names)}
            missing = [name for name in HEFT_JOINT_NAMES if name not in source_index]
            if missing:
                raise ValueError(f"HEFT motion is missing joints {missing}: {path}")
            indices = np.asarray([source_index[name] for name in HEFT_JOINT_NAMES], dtype=np.int64)
            # The upstream config uses end=-1, so retain the same frame range.
            joint_pos = np.asarray(data["dof_pos"][:-1, indices], dtype=np.float32)
            root_pos = np.asarray(data["root_pos"][:-1], dtype=np.float32)
            root_xyzw = np.asarray(data["root_rot"][:-1], dtype=np.float32)
        root_quat = np.concatenate([root_xyzw[:, 3:4], root_xyzw[:, :3]], axis=1)
        return HeftMotion(joint_pos=joint_pos, root_quat=root_quat, root_pos=root_pos)


class HeftOnnxPolicy:
    def __init__(self, path: str | Path, *, threads: int = 4) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"HEFT ONNX policy does not exist: {self.path}")
        data_path = self.path.with_suffix(self.path.suffix + ".data")
        if not data_path.exists():
            raise FileNotFoundError(f"HEFT external ONNX weights do not exist: {data_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, HEFT_OBSERVATION_DIM]:
            raise ValueError(f"Unexpected HEFT ONNX input ABI: {[(item.name, item.shape) for item in inputs]}")
        self.input_name = inputs[0].name
        self.action_output_name = self._resolve_action_output_name()

    def _resolve_action_output_name(self) -> str:
        metadata_path = self.path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            out_keys = list(metadata.get("out_keys", ()))
            if "action" in out_keys:
                return self.session.get_outputs()[out_keys.index("action")].name
        outputs = self.session.get_outputs()
        for item in outputs:
            if item.name == "action" and item.shape == [1, 29]:
                return item.name
        candidates = [item.name for item in outputs if item.shape == [1, 29]]
        if not candidates:
            raise ValueError(f"Cannot locate the HEFT 29-D action output in {self.path}")
        return candidates[-1]

    def act(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32).reshape(-1, HEFT_OBSERVATION_DIM)
        actions = []
        for observation in observations:
            output = self.session.run(
                [self.action_output_name],
                {self.input_name: observation.reshape(1, -1)},
            )[0]
            action = np.asarray(output[0], dtype=np.float32)
            if action.shape != (29,) or not np.all(np.isfinite(action)):
                raise FloatingPointError("HEFT returned a non-finite or malformed action.")
            actions.append(np.clip(action, -10.0, 10.0))
        return np.stack(actions, axis=0)


class _HeftRuntime:
    def __init__(self) -> None:
        self.root_angvel = np.zeros((21, 3), dtype=np.float32)
        self.projected_gravity = np.zeros((21, 3), dtype=np.float32)
        self.joint_pos = np.zeros((21, 29), dtype=np.float32)
        self.joint_vel = np.zeros((21, 29), dtype=np.float32)
        self.prev_action = np.zeros((HEFT_PREV_ACTION_STEPS, 29), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.boot_value = 25
        self.ref_joint_pos = np.zeros((1, 29), dtype=np.float32)
        self.ref_root_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.ref_root_pos = np.asarray([[0.0, 0.0, 0.78]], dtype=np.float32)
        self.ref_idx = 0
        self.motion_name = "default"

    def reset_history(self) -> None:
        self.root_angvel.fill(0.0)
        self.projected_gravity.fill(0.0)
        self.joint_pos.fill(0.0)
        self.joint_vel.fill(0.0)
        self.prev_action.fill(0.0)
        self.last_action.fill(0.0)
        self.boot_value = 25



class HeftG1OrcaPlayBridge:
    """Run the upstream HEFT G1 PMG policy directly in an OrcaLab G1 scene."""

    def __init__(
        self,
        env: Any,
        *,
        policy_path: str | Path,
        motion_dir: str | Path | None,
        transition_steps: int = 100,
        onnx_threads: int = 4,
    ) -> None:
        if int(getattr(env, "num_actions", 0)) != 29:
            raise ValueError(f"HEFT G1 requires a 29-D body action ABI, got {getattr(env, 'num_actions', None)}")
        self.env = env
        self.policy = HeftOnnxPolicy(policy_path, threads=onnx_threads)
        self.motions = HeftMotionLibrary(motion_dir) if motion_dir is not None else None
        self.transition_steps = max(0, int(transition_steps))
        self.runtime = [_HeftRuntime() for _ in range(int(env.num_envs))]
        self._task_offsets: list[int] = []
        self._env_to_heft: list[np.ndarray] = []
        self._heft_to_env: list[np.ndarray] = []
        offset = 0
        for task in env.tasks:
            names = list(task.robot_config.get("leg_joint_names") or ())
            if len(names) != 29 or set(names) != set(HEFT_JOINT_NAMES):
                raise ValueError("The OrcaLab G1 joint set does not match the HEFT 29-DoF joint set.")
            env_to_heft = np.asarray([names.index(name) for name in HEFT_JOINT_NAMES], dtype=np.int64)
            heft_to_env = np.asarray([HEFT_JOINT_NAMES.index(name) for name in names], dtype=np.int64)
            self._task_offsets.append(offset)
            self._env_to_heft.append(env_to_heft)
            self._heft_to_env.append(heft_to_env)
            self._align_task(task, heft_to_env)
            offset += int(task.num_envs)
        if offset != len(self.runtime):
            raise ValueError(f"HEFT environment accounting mismatch: tasks={offset}, env={len(self.runtime)}")
        self.current_motion = "default"

    def reset(self, motion_name: str | None = None, *, reset_env: bool = True) -> None:
        if reset_env:
            self.env.reset()
        selected = self.current_motion if motion_name is None else str(motion_name)
        self.current_motion = selected
        for runtime in self.runtime:
            runtime.reset_history()
        self.set_motion(selected)

    def set_motion(self, motion_name: str) -> None:
        if self.motions is None:
            raise RuntimeError("This HEFT bridge uses an online teacher and has no file-backed motion library.")
        motion = self.motions.get(motion_name)
        states = self._read_states()
        for runtime, state in zip(self.runtime, states):
            runtime.ref_joint_pos, runtime.ref_root_quat, runtime.ref_root_pos = self._make_reference(
                motion,
                current_joint_pos=state[0],
                current_root_quat=state[2],
                current_root_pos=state[3],
            )
            runtime.ref_idx = 0
            runtime.motion_name = str(motion_name)
        self.current_motion = str(motion_name)

    def get_observations(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        for runtime, state in zip(self.runtime, self._read_states()):
            qpos, qvel, root_quat, _root_pos, root_angvel = state
            if runtime.ref_idx < runtime.ref_joint_pos.shape[0] - 1:
                runtime.ref_idx += 1
            self._push_history(runtime.root_angvel, root_angvel)
            self._push_history(runtime.projected_gravity, _quat_apply_inv(root_quat, _GRAVITY))
            self._push_history(runtime.joint_pos, qpos)
            self._push_history(runtime.joint_vel, qvel)
            self._push_history(runtime.prev_action, runtime.last_action)
            runtime.boot_value = max(runtime.boot_value - 1, 0)
            observation = self._build_observation(runtime, qpos, root_quat)
            if observation.shape != (HEFT_OBSERVATION_DIM,):
                raise ValueError(f"HEFT observation ABI mismatch: {observation.shape}")
            if not np.all(np.isfinite(observation)):
                raise FloatingPointError("Non-finite state reached the HEFT observation.")
            observations.append(observation)
        return np.stack(observations, axis=0)

    def act(self) -> np.ndarray:
        return self.policy.act(self.get_observations())

    def step(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float64).reshape(len(self.runtime), 29)
        start = 0
        for task_index, task in enumerate(self.env.tasks):
            stop = start + int(task.num_envs)
            action_heft = actions[start:stop]
            action_env = action_heft[:, self._heft_to_env[task_index]]
            target_heft = HEFT_DEFAULT_QPOS.reshape(1, -1) + action_heft * HEFT_ACTION_SCALE.reshape(1, -1)
            target_env = target_heft[:, self._heft_to_env[task_index]]

            for local_index, agent in enumerate(task.agents):
                agent.previous_action = agent.last_action.copy()
                agent.last_action = action_env[local_index].copy()
                self.runtime[start + local_index].last_action[:] = action_heft[local_index]

            task.prepare_control_buffer()
            task.ctrl[task._flat_actuator_ids] = target_env.reshape(-1)
            for _ in range(task.decimation):
                task.set_ctrl(task.ctrl)
                task.mj_step(nstep=task.frame_skip)
            task.update_data()
            if task._render_mode == "human":
                task.render()
            task.episode_lengths += 1
            self.env.episode_length_buf[start:stop] = task.episode_lengths
            start = stop

    def progress(self) -> tuple[int, int]:
        if not self.runtime:
            return 0, 0
        runtime = self.runtime[0]
        return int(runtime.ref_idx), int(runtime.ref_joint_pos.shape[0])

    def _read_states(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        states = []
        for task_index, task in enumerate(self.env.tasks):
            reorder = self._env_to_heft[task_index]
            for local_index, agent in enumerate(task.agents):
                base_qpos = task.data.qpos[task._base_qpos_indices[local_index]].copy()
                base_qvel = task.data.qvel[task._base_qvel_indices[local_index]].copy()
                qpos = task.data.qpos[agent.leg_qpos_indices].copy()[reorder]
                qvel = task.data.qvel[agent.leg_qvel_indices].copy()[reorder]
                states.append(
                    (
                        qpos.astype(np.float32),
                        qvel.astype(np.float32),
                        _quat_normalize(base_qpos[3:7]).astype(np.float32),
                        base_qpos[:3].astype(np.float32),
                        base_qvel[3:6].astype(np.float32),
                    )
                )
        return states

    def _make_reference(
        self,
        motion: HeftMotion,
        *,
        current_joint_pos: np.ndarray,
        current_root_quat: np.ndarray,
        current_root_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_yaw = _yaw_quat(motion.root_quat[0])
        current_yaw = _yaw_quat(current_root_quat)
        yaw_delta = _quat_mul(current_yaw, _quat_conjugate(source_yaw))
        root_offset = motion.root_pos - motion.root_pos[0]
        aligned_pos = _quat_apply(yaw_delta, root_offset) + current_root_pos.reshape(1, 3)
        # HEFT treats reference height as absolute rather than relative.
        aligned_pos[:, 2] = motion.root_pos[:, 2]
        aligned_quat = _quat_mul(yaw_delta, motion.root_quat)

        if self.transition_steps:
            joint_transition = _lerp_exclusive(current_joint_pos, motion.joint_pos[0], self.transition_steps)
            pos_target = aligned_pos[0].copy()
            pos_transition = _lerp_exclusive(current_root_pos, pos_target, self.transition_steps)
            quat_transition = _slerp_exclusive(current_root_quat, aligned_quat[0], self.transition_steps)
            joint_pos = np.concatenate([joint_transition, motion.joint_pos], axis=0)
            root_quat = np.concatenate([quat_transition, aligned_quat], axis=0)
            root_pos = np.concatenate([pos_transition, aligned_pos], axis=0)
        else:
            joint_pos = motion.joint_pos.copy()
            root_quat = aligned_quat.copy()
            root_pos = aligned_pos.copy()
        return (
            np.asarray(joint_pos, dtype=np.float32),
            np.asarray(root_quat, dtype=np.float32),
            np.asarray(root_pos, dtype=np.float32),
        )

    @staticmethod
    def _build_observation(runtime: _HeftRuntime, current_qpos: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        indices = np.clip(
            runtime.ref_idx + HEFT_FUTURE_STEPS,
            0,
            runtime.ref_joint_pos.shape[0] - 1,
        )
        target_joint = runtime.ref_joint_pos[indices]
        target_quat = runtime.ref_root_quat[indices]
        target_pos = runtime.ref_root_pos[indices]

        position_delta = target_pos[1:] - target_pos[0:1]
        position_delta_body = _quat_apply_inv(target_quat[0], position_delta)
        relative_quat = _quat_mul(_quat_conjugate(current_quat), target_quat)
        relative_rot = _quat_to_matrix(relative_quat)
        rot6d = relative_rot[:, :, :2].transpose(0, 2, 1).reshape(-1)
        tracking_command = np.concatenate([position_delta_body.reshape(-1), rot6d])

        target_joint_obs = np.concatenate(
            [target_joint.reshape(-1), (target_joint - current_qpos.reshape(1, -1)).reshape(-1)]
        )
        target_root_z = target_pos[:, 2]
        target_gravity = _quat_apply_inv(target_quat, _GRAVITY).reshape(-1)

        return np.concatenate(
            [
                np.asarray([runtime.boot_value / 25.0], dtype=np.float32),
                tracking_command,
                target_joint_obs,
                target_root_z,
                target_gravity,
                runtime.root_angvel[HEFT_HISTORY_STEPS].reshape(-1),
                runtime.projected_gravity[HEFT_HISTORY_STEPS].reshape(-1),
                runtime.joint_pos[HEFT_HISTORY_STEPS].reshape(-1),
                runtime.joint_vel[HEFT_HISTORY_STEPS].reshape(-1),
                runtime.prev_action.reshape(-1),
            ]
        ).astype(np.float32)

    @staticmethod
    def _push_history(history: np.ndarray, value: np.ndarray) -> None:
        history[1:] = history[:-1]
        history[0] = np.asarray(value, dtype=np.float32)

    @staticmethod
    def _align_task(task: Any, heft_to_env: np.ndarray) -> None:
        model = getattr(getattr(task, "gym", None), "_mjModel", None)
        if model is None:
            raise ValueError("HEFT play requires a CPU MuJoCo model.")
        default_env = HEFT_DEFAULT_QPOS[heft_to_env]
        kp_env = HEFT_KP[heft_to_env]
        kd_env = HEFT_KD[heft_to_env]
        armature_env = HEFT_ARMATURE[heft_to_env]
        effort_env = HEFT_EFFORT_LIMIT[heft_to_env]
        joint_dict = task.model.get_joint_dict()
        actuator_dict = task.model.get_actuator_dict()
        foot_bodies = {name for agent in task.agents for name in agent.foot_body_names}

        for agent in task.agents:
            agent.nominal_qpos = default_env.copy()
            agent.kp = kp_env.copy()
            agent.kd = kd_env.copy()
            agent.torque_limits = np.stack([-effort_env, effort_env], axis=1)
            for index, (joint_name, actuator_name) in enumerate(zip(agent.leg_joint_names, agent.actuator_names)):
                joint_id = int(joint_dict[joint_name]["JointId"])
                actuator_id = int(actuator_dict[actuator_name]["ActuatorId"])
                dof_id = int(agent.leg_qvel_indices[index])
                kp = float(kp_env[index])
                kd = float(kd_env[index])
                effort = float(effort_env[index])
                model.dof_armature[dof_id] = float(armature_env[index])
                model.dof_damping[dof_id] = 0.0
                model.dof_frictionloss[dof_id] = 0.0
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
                low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
                delta = effort / kp if kp > 0.0 else effort
                model.actuator_ctrlrange[actuator_id] = (low - delta, high + delta)

        for geom_id in range(model.ngeom):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
            if body_name in foot_bodies and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                model.geom_friction[geom_id, 0] = 0.75
                model.geom_condim[geom_id] = 3

        task._nominal_qpos = np.broadcast_to(default_env, task._nominal_qpos.shape).copy()
        task._kp = np.broadcast_to(kp_env, task._kp.shape).copy()
        task._kd = np.broadcast_to(kd_env, task._kd.shape).copy()
        task._torque_low = np.broadcast_to(-effort_env, task._torque_low.shape).copy()
        task._torque_high = np.broadcast_to(effort_env, task._torque_high.shape).copy()
        task._heft_position_actuator_aligned = True
        task.mj_forward()
        task.update_data()



class HeftG1RecordedCommandBridge(HeftG1OrcaPlayBridge):
    """Select file-backed motion captured from the original G1 velocity policy."""

    _STAND_MOTION = "heft_stand"
    _COMMAND_MOTIONS = {
        "w": "heft_forward",
        "s": "heft_backward",
        "a": "heft_left",
        "d": "heft_right",
        "z": "heft_yaw_left",
        "c": "heft_yaw_right",
    }
    _WALK_MOTIONS = {
        "f1": "walk1_subject1",
        "f2": "walk2_subject1",
        "f3": "walk3_subject1",
    }

    def __init__(
        self,
        env: Any,
        *,
        policy_path: str | Path,
        motion_dir: str | Path,
        walk_motion_dir: str | Path | None = None,
        transition_steps: int = 20,
        loop_guard_frames: int = 16,
        onnx_threads: int = 4,
    ) -> None:
        super().__init__(
            env,
            policy_path=policy_path,
            motion_dir=motion_dir,
            transition_steps=transition_steps,
            onnx_threads=onnx_threads,
        )
        if int(env.num_envs) != 1:
            raise ValueError("Recorded G1 keyboard playback currently requires exactly one environment.")
        if walk_motion_dir is not None:
            self.motions.add_directory(walk_motion_dir)
        required = {self._STAND_MOTION, *self._COMMAND_MOTIONS.values(), *self._WALK_MOTIONS.values()}
        missing = sorted(required.difference(self.motions.names))
        if missing:
            raise FileNotFoundError(f"Missing recorded G1 command motions in {motion_dir}: {missing}")
        self.commands = np.zeros((1, 3), dtype=np.float64)
        self.manual_motion: str | None = None
        self.requested_motion = self._STAND_MOTION
        self.loop_guard_frames = max(1, int(loop_guard_frames))
        self.current_motion = self._STAND_MOTION

    def reset(self, motion_name: str | None = None, *, reset_env: bool = True) -> None:
        del motion_name
        self.commands.fill(0.0)
        self.manual_motion = None
        self.requested_motion = self._STAND_MOTION
        super().reset(self._STAND_MOTION, reset_env=reset_env)

    def set_commands(self, commands: np.ndarray) -> None:
        command = np.asarray(commands, dtype=np.float64).reshape(1, 3)
        if not np.all(np.isfinite(command)):
            raise ValueError("Recorded G1 velocity commands must be finite.")
        self.commands[:] = command
        if np.max(np.abs(command)) >= 0.025:
            self.manual_motion = None
            self.requested_motion = self._motion_for_command(command[0])
        elif self.manual_motion is None:
            self.requested_motion = self._STAND_MOTION

    def select_motion_key(self, selection: str) -> None:
        selection = str(selection)
        self.commands.fill(0.0)
        if selection == "stand":
            self.manual_motion = None
            self.requested_motion = self._STAND_MOTION
            return
        try:
            self.manual_motion = self._WALK_MOTIONS[selection]
        except KeyError as exc:
            raise ValueError(f"Unknown HEFT walk selection: {selection!r}") from exc
        self.requested_motion = self.manual_motion

    def get_observations(self) -> np.ndarray:
        runtime = self.runtime[0]
        remaining = int(runtime.ref_joint_pos.shape[0] - runtime.ref_idx - 1)
        if self.requested_motion != self.current_motion or remaining <= self.loop_guard_frames:
            self.set_motion(self.requested_motion)
        return super().get_observations()

    def _motion_for_command(self, command: np.ndarray) -> str:
        vx, vy, wz = np.asarray(command, dtype=np.float64)
        normalized = np.asarray([vx / 0.5, vy / 0.5, wz / 0.8], dtype=np.float64)
        axis = int(np.argmax(np.abs(normalized)))
        if abs(float(normalized[axis])) < 0.05:
            return self._STAND_MOTION
        if axis == 0:
            key = "w" if vx > 0.0 else "s"
        elif axis == 1:
            key = "a" if vy > 0.0 else "d"
        else:
            key = "z" if wz > 0.0 else "c"
        return self._COMMAND_MOTIONS[key]



_GRAVITY = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _quat_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1.0e-9)


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    vector = np.asarray(vector)
    zeros = np.zeros(vector.shape[:-1] + (1,), dtype=np.result_type(quat, vector))
    pure = np.concatenate([zeros, vector], axis=-1)
    return _quat_mul(_quat_mul(quat, pure), _quat_conjugate(quat))[..., 1:]


def _quat_apply_inv(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _quat_apply(_quat_conjugate(quat), vector)


def _yaw_quat(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(np.asarray(quat, dtype=np.float64))
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-9)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _lerp_exclusive(start: np.ndarray, stop: np.ndarray, steps: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, steps + 2, dtype=np.float64)[1:-1, None]
    return start.reshape(1, -1) * (1.0 - alpha) + stop.reshape(1, -1) * alpha


def _slerp_exclusive(start: np.ndarray, stop: np.ndarray, steps: int) -> np.ndarray:
    q0 = _quat_normalize(start)
    q1 = _quat_normalize(stop)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    alpha = np.linspace(0.0, 1.0, steps + 2, dtype=np.float64)[1:-1]
    if dot > 0.9995:
        result = q0[None, :] + alpha[:, None] * (q1 - q0)[None, :]
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-9)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - alpha) * theta)[:, None] / sin_theta * q0[None, :]
        + np.sin(alpha * theta)[:, None] / sin_theta * q1[None, :]
    )


__all__ = [
    "HEFT_JOINT_NAMES",
    "HEFT_OBSERVATION_DIM",
    "HeftG1OrcaPlayBridge",
    "HeftG1RecordedCommandBridge",
    "HeftMotionLibrary",
    "HeftOnnxPolicy",
]
