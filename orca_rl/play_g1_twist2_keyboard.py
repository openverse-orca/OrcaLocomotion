from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Protocol

import numpy as np

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    load_task_and_train_cfg,
)

ensure_project_root_on_path()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "orca_rl/assets/ckpts/twist2/twist2_1017_20k.onnx"
DEFAULT_TWIST2_XML = PROJECT_ROOT / "orca_rl/assets/robots/unitree_g1/xmls/g1_twist2_sim2sim_29dof.xml"

DEFAULT_DOF_POS = np.array(
    [
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

DEFAULT_MIMIC_OBS = np.concatenate(
    [
        np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float64),
        DEFAULT_DOF_POS,
    ]
)

MUJOCO_RESET_DOF_POS = DEFAULT_DOF_POS.copy()
MUJOCO_RESET_DOF_POS[16] = 0.2
MUJOCO_RESET_DOF_POS[23] = -0.2

ACTION_SCALE = np.full(29, 0.5, dtype=np.float64)
ANKLE_IDX = np.array([4, 5, 10, 11], dtype=np.int64)
ARM_IDX = np.arange(15, 29, dtype=np.int64)

TWIST2_STIFFNESS = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        150,
        150,
        150,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
    ],
    dtype=np.float64,
)

TWIST2_DAMPING = np.array(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        0.2,
        0.2,
        0.2,
        5,
        5,
        5,
        5,
        0.2,
        0.2,
        0.2,
    ],
    dtype=np.float64,
)

TWIST2_TORQUE_LIMITS = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        150,
        150,
        150,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
    ],
    dtype=np.float64,
)

DOF_INDEX = {
    "torso_yaw": 12,
    "torso_roll": 13,
    "torso_pitch": 14,
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

HELP = """
TWIST2 keyboard play for OrcaLab G1

Base:
  w/s  forward/back velocity      a/d  left/right velocity
  q/e  yaw velocity               space zero velocities
  r/f  body height up/down        t    reset mimic target

Torso:
  j/l  torso yaw                  i/k  torso pitch

Arms:
  1/2  left shoulder pitch        3/4  right shoulder pitch
  5/6  left elbow                 7/8  right elbow
  9/0  both arms inward/outward

Other:
  ?    print help                 Q    quit
"""


class OnnxPolicy:
    def __init__(self, path: str | Path, *, device: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "TWIST2 ONNX play needs onnxruntime. Install it in the Orca runtime env, "
                "for example: python -m pip install onnxruntime-gpu"
            ) from exc

        policy_path = Path(path).expanduser().resolve()
        if not policy_path.exists():
            raise FileNotFoundError(f"TWIST2 policy does not exist: {policy_path}")
        providers: list[str] = []
        available = ort.get_available_providers()
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_dim = int(np.prod(self.session.get_inputs()[0].shape[1:]))
        self.output_dim = int(np.prod(self.session.get_outputs()[0].shape[1:]))
        print(f"TWIST2 ONNX policy loaded: {policy_path}")
        print(f"ONNX providers: {self.session.get_providers()}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        outputs = self.session.run(None, {self.input_name: obs})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


@dataclass
class Twist2ObsState:
    history: deque[np.ndarray]
    last_action: np.ndarray


class Twist2Keyboard:
    def __init__(self, *, backend: str, vel_step: float, yaw_step: float, joint_step: float) -> None:
        self.backend_name = backend
        self.vel_step = float(vel_step)
        self.yaw_step = float(yaw_step)
        self.joint_step = float(joint_step)
        self.mimic_obs = DEFAULT_MIMIC_OBS.copy()
        self.quit_requested = False
        self._active_keys: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self._old_settings = None

    @property
    def name(self) -> str:
        return self.backend_name

    def __enter__(self) -> "Twist2Keyboard":
        if self.backend_name == "global":
            try:
                from pynput import keyboard as pynput_keyboard
            except ImportError as exc:
                raise RuntimeError(
                    "Global keyboard backend needs pynput. Install with: python -m pip install pynput"
                ) from exc
            self._pynput_keyboard = pynput_keyboard
            self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        elif self.backend_name == "terminal" and sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def poll(self) -> tuple[np.ndarray, bool]:
        if self.backend_name == "terminal":
            key = _read_terminal_key()
            if key:
                self._apply_key(key, pressed=True)
        with self._lock:
            return self.mimic_obs.copy(), self.quit_requested

    def _on_press(self, key) -> None:
        self._apply_key(self._key_name(key), pressed=True)

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            self._active_keys.discard(name)

    def _key_name(self, key) -> str:
        keyboard_module = self._pynput_keyboard
        if key == keyboard_module.Key.space:
            return "space"
        char = getattr(key, "char", None)
        return str(char) if char is not None else str(key)

    def _apply_key(self, key: str, *, pressed: bool) -> None:
        if not pressed:
            return
        with self._lock:
            if key in {"Q", "q", "\x03", "\x04"}:
                self.quit_requested = True
                return
            obs = self.mimic_obs
            dof = obs[6:]
            if key == "w":
                obs[0] = _clip(obs[0] + self.vel_step, -1.0, 1.0)
            elif key == "s":
                obs[0] = _clip(obs[0] - self.vel_step, -1.0, 1.0)
            elif key == "a":
                obs[1] = _clip(obs[1] + self.vel_step, -0.8, 0.8)
            elif key == "d":
                obs[1] = _clip(obs[1] - self.vel_step, -0.8, 0.8)
            elif key == "q":
                obs[5] = _clip(obs[5] + self.yaw_step, -1.5, 1.5)
            elif key == "e":
                obs[5] = _clip(obs[5] - self.yaw_step, -1.5, 1.5)
            elif key in {" ", "space"}:
                obs[0] = obs[1] = obs[5] = 0.0
            elif key == "r":
                obs[2] = _clip(obs[2] + 0.02, 0.65, 0.95)
            elif key == "f":
                obs[2] = _clip(obs[2] - 0.02, 0.65, 0.95)
            elif key == "t":
                obs[:] = DEFAULT_MIMIC_OBS
            elif key == "j":
                dof[DOF_INDEX["torso_yaw"]] = _clip(dof[DOF_INDEX["torso_yaw"]] + self.joint_step, -0.6, 0.6)
            elif key == "l":
                dof[DOF_INDEX["torso_yaw"]] = _clip(dof[DOF_INDEX["torso_yaw"]] - self.joint_step, -0.6, 0.6)
            elif key == "i":
                dof[DOF_INDEX["torso_pitch"]] = _clip(dof[DOF_INDEX["torso_pitch"]] + self.joint_step, -0.6, 0.6)
            elif key == "k":
                dof[DOF_INDEX["torso_pitch"]] = _clip(dof[DOF_INDEX["torso_pitch"]] - self.joint_step, -0.6, 0.6)
            elif key == "1":
                dof[DOF_INDEX["left_shoulder_pitch"]] = _clip(dof[DOF_INDEX["left_shoulder_pitch"]] + self.joint_step, -1.0, 1.2)
            elif key == "2":
                dof[DOF_INDEX["left_shoulder_pitch"]] = _clip(dof[DOF_INDEX["left_shoulder_pitch"]] - self.joint_step, -1.0, 1.2)
            elif key == "3":
                dof[DOF_INDEX["right_shoulder_pitch"]] = _clip(dof[DOF_INDEX["right_shoulder_pitch"]] + self.joint_step, -1.0, 1.2)
            elif key == "4":
                dof[DOF_INDEX["right_shoulder_pitch"]] = _clip(dof[DOF_INDEX["right_shoulder_pitch"]] - self.joint_step, -1.0, 1.2)
            elif key == "5":
                dof[DOF_INDEX["left_elbow"]] = _clip(dof[DOF_INDEX["left_elbow"]] + self.joint_step, 0.0, 1.6)
            elif key == "6":
                dof[DOF_INDEX["left_elbow"]] = _clip(dof[DOF_INDEX["left_elbow"]] - self.joint_step, 0.0, 1.6)
            elif key == "7":
                dof[DOF_INDEX["right_elbow"]] = _clip(dof[DOF_INDEX["right_elbow"]] + self.joint_step, 0.0, 1.6)
            elif key == "8":
                dof[DOF_INDEX["right_elbow"]] = _clip(dof[DOF_INDEX["right_elbow"]] - self.joint_step, 0.0, 1.6)
            elif key == "9":
                dof[DOF_INDEX["left_shoulder_roll"]] = _clip(dof[DOF_INDEX["left_shoulder_roll"]] - self.joint_step, -1.2, 1.2)
                dof[DOF_INDEX["right_shoulder_roll"]] = _clip(dof[DOF_INDEX["right_shoulder_roll"]] + self.joint_step, -1.2, 1.2)
            elif key == "0":
                dof[DOF_INDEX["left_shoulder_roll"]] = _clip(dof[DOF_INDEX["left_shoulder_roll"]] + self.joint_step, -1.2, 1.2)
                dof[DOF_INDEX["right_shoulder_roll"]] = _clip(dof[DOF_INDEX["right_shoulder_roll"]] - self.joint_step, -1.2, 1.2)
            elif key == "?":
                print(HELP)


class NoKeyboard:
    name = "none"

    def __init__(self) -> None:
        self.mimic_obs = DEFAULT_MIMIC_OBS.copy()

    def __enter__(self) -> "NoKeyboard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def poll(self) -> tuple[np.ndarray, bool]:
        return self.mimic_obs.copy(), False


class KeyboardLike(Protocol):
    name: str

    def __enter__(self) -> "KeyboardLike": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...

    def poll(self) -> tuple[np.ndarray, bool]: ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TWIST2 ONNX policy on OrcaLab G1 with keyboard mimic commands.")
    parser.add_argument("--config", default="Unitree-G1-Flat", help="Registered Orca G1 task or cfg path.")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="TWIST2 ONNX policy path.")
    parser.add_argument("--device", default="cuda", help="ONNX device preference: cuda or cpu.")
    parser.add_argument("--remote", default=None, help="Override OrcaGym address, e.g. localhost:50051.")
    parser.add_argument("--local-mujoco", action="store_true", help="Run against local MuJoCo XML.")
    parser.add_argument(
        "--model-source",
        choices=("twist2", "orca"),
        default="twist2",
        help="Local MJCF source. twist2 uses a small adapter around TWIST2's g1_sim2sim_29dof.xml.",
    )
    parser.add_argument("--twist2-xml", default=str(DEFAULT_TWIST2_XML), help="Orca-compatible TWIST2 MuJoCo XML used when --model-source twist2.")
    parser.add_argument("--keyboard-backend", choices=("auto", "global", "terminal", "none"), default="auto")
    parser.add_argument("--frequency", type=float, default=100.0, help="TWIST2 policy/control frequency.")
    parser.add_argument("--vel-step", type=float, default=0.1)
    parser.add_argument("--yaw-step", type=float, default=0.15)
    parser.add_argument("--joint-step", type=float, default=0.05)
    parser.add_argument("--no-twist2-reset", action="store_true", help="Do not reset G1 joints to TWIST2 default pose.")
    parser.add_argument(
        "--arm-source",
        choices=("mimic", "default", "policy"),
        default="policy",
        help="Arm target source. policy matches TWIST2 sim2sim and is needed for balance.",
    )
    parser.add_argument(
        "--body-source",
        choices=("policy", "default", "mimic"),
        default="policy",
        help="Leg/waist target source. default is useful to verify that the Orca G1 can stand before enabling policy.",
    )
    parser.add_argument("--policy-scale", type=float, default=1.0, help="Scale TWIST2 policy actions before target mapping.")
    parser.add_argument("--policy-ramp-seconds", type=float, default=0.0, help="Blend policy actions in from zero at startup.")
    parser.add_argument("--state-log-interval", type=float, default=0.5, help="Seconds between status logs. Use 0 to disable.")
    parser.add_argument("--debug-first-action", action="store_true", help="Print the first TWIST2 obs/action and exit.")
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()

    try:
        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.run_play import _apply_play_scene_mode
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    _set_control_frequency(task_cfg, float(args.frequency))
    if args.local_mujoco and args.model_source == "twist2":
        task_cfg.setdefault("scene_binding", {})["local_xml_path"] = str(Path(args.twist2_xml).expanduser().resolve())
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    task_cfg.setdefault("randomization", {})["enabled"] = False
    if not args.local_mujoco:
        check_orcagym_addresses(task_cfg)

    policy = OnnxPolicy(args.policy, device=str(args.device))
    if policy.input_dim != 1432 or policy.output_dim != 29:
        raise ValueError(f"Expected TWIST2 ONNX shape 1432 -> 29, got {policy.input_dim} -> {policy.output_dim}")

    env = make_locomotion_vec_env(task_cfg, device="cpu", render_mode="human", headless=False)
    try:
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=str(args.device),
            checkpoint=str(Path(args.policy).expanduser()),
        )
        _validate_g1_env(env)
        if not args.no_twist2_reset:
            _reset_env_to_twist2_default(env)
        obs_states = [Twist2ObsState(deque(maxlen=10), np.zeros(29, dtype=np.float32)) for _ in range(env.num_envs)]
        for state in obs_states:
            for _ in range(10):
                state.history.append(np.zeros(127, dtype=np.float32))

        if args.debug_first_action:
            mimic_obs = DEFAULT_MIMIC_OBS.copy()
            obs_buf = _build_twist2_observation(env.tasks[0], 0, mimic_obs, obs_states[0])
            action = policy.act(obs_buf)
            target = DEFAULT_DOF_POS + np.clip(action, -10.0, 10.0) * ACTION_SCALE
            print(f"first_obs min={obs_buf.min():+.4f} max={obs_buf.max():+.4f} norm={np.linalg.norm(obs_buf):.4f}")
            print(f"first_action min={action.min():+.4f} max={action.max():+.4f} norm={np.linalg.norm(action):.4f}")
            for i, (a, q) in enumerate(zip(action, target)):
                print(f"{i:02d} action={a:+.4f} target={q:+.4f}")
            return

        keyboard = _make_keyboard(args)
        dt = 1.0 / max(1.0, float(args.frequency))
        print(HELP)
        print(f"OrcaLab TWIST2 play ready. Keyboard backend={keyboard.name}. Policy dt={dt:.3f}s.")

        start_time = time.perf_counter()
        last_status = 0.0
        with keyboard:
            while args.seconds is None or time.perf_counter() - start_time < float(args.seconds):
                step_start = time.perf_counter()
                mimic_obs, quit_requested = keyboard.poll()
                if quit_requested:
                    break
                action_batch = []
                mimic_batch = []
                state_index = 0
                for task in env.tasks:
                    for env_index in range(task.num_envs):
                        obs_buf = _build_twist2_observation(task, env_index, mimic_obs, obs_states[state_index])
                        raw_action = policy.act(obs_buf)
                        obs_states[state_index].last_action = raw_action.astype(np.float32)
                        ramp = _policy_ramp(start_time, seconds=float(args.policy_ramp_seconds))
                        action = raw_action * ramp * float(args.policy_scale)
                        action_batch.append(action)
                        mimic_batch.append(mimic_obs)
                        state_index += 1
                _step_env_with_twist2_actions(
                    env,
                    np.stack(action_batch, axis=0),
                    np.stack(mimic_batch, axis=0),
                    arm_source=str(args.arm_source),
                    body_source=str(args.body_source),
                )
                now = time.perf_counter()
                if float(args.state_log_interval) > 0.0 and now - last_status > float(args.state_log_interval):
                    _print_status(env, mimic_obs, np.stack(action_batch, axis=0))
                    last_status = now
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _make_keyboard(args: argparse.Namespace) -> KeyboardLike:
    backend = str(args.keyboard_backend)
    if backend == "none":
        return NoKeyboard()
    if backend == "auto":
        try:
            import pynput  # noqa: F401
        except ImportError:
            backend = "terminal"
        else:
            backend = "global"
    return Twist2Keyboard(
        backend=backend,
        vel_step=float(args.vel_step),
        yaw_step=float(args.yaw_step),
        joint_step=float(args.joint_step),
    )


def _set_control_frequency(task_cfg: dict, frequency: float) -> None:
    sim_cfg = task_cfg.setdefault("sim", {})
    hz = max(1.0, float(frequency))
    time_step = float(sim_cfg.get("time_step", 0.001))
    frame_skip = max(1, int(round(1.0 / (hz * time_step))))
    sim_cfg["time_step"] = time_step
    sim_cfg["frame_skip"] = frame_skip
    sim_cfg["decimation"] = 1


def _validate_g1_env(env) -> None:
    if not getattr(env, "tasks", None):
        raise ValueError("Orca env has no tasks.")
    task = env.tasks[0]
    joint_names = list(task.robot_config.get("leg_joint_names") or [])
    if len(joint_names) != 29:
        raise ValueError(f"TWIST2 Orca play expects a 29DOF G1 env, got {len(joint_names)} controlled joints.")


def _reset_env_to_twist2_default(env) -> None:
    for task in env.tasks:
        joint_qpos: dict[str, np.ndarray] = {}
        joint_qvel: dict[str, np.ndarray] = {}
        for agent in task.agents:
            base_qpos = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            joint_qpos[agent.base_joint_name] = base_qpos
            joint_qvel[agent.base_joint_name] = np.zeros(6, dtype=np.float64)
            agent.initial_base_qpos = base_qpos.copy()
            for joint_name, value in zip(agent.leg_joint_names, MUJOCO_RESET_DOF_POS):
                joint_qpos[joint_name] = np.array([value], dtype=np.float64)
                joint_qvel[joint_name] = np.zeros(1, dtype=np.float64)
            agent.last_action.fill(0.0)
            agent.previous_action.fill(0.0)
            agent.last_torque.fill(0.0)
        task.set_joint_qpos(joint_qpos)
        task.set_joint_qvel(joint_qvel)
        task.ctrl[:] = 0.0
        task.set_ctrl(task.ctrl)
        task.mj_forward()
        task.update_data()
        if task._render_mode == "human":
            task.render()


def _build_twist2_observation(task, env_index: int, mimic_obs: np.ndarray, state: Twist2ObsState) -> np.ndarray:
    agent = task.agents[env_index]
    base_qpos = task.data.qpos[task._base_qpos_indices[env_index]].copy()
    base_qvel = task.data.qvel[task._base_qvel_indices[env_index]].copy()
    dof_pos = task.data.qpos[agent.leg_qpos_indices].copy()
    dof_vel = task.data.qvel[agent.leg_qvel_indices].copy()

    quat = base_qpos[3:7]
    rpy = _quat_to_euler_wxyz(quat)
    ang_vel = base_qvel[3:6]
    obs_body_dof_vel = dof_vel.copy()
    obs_body_dof_vel[ANKLE_IDX] = 0.0
    obs_proprio = np.concatenate(
        [
            ang_vel * 0.25,
            rpy[:2],
            dof_pos - DEFAULT_DOF_POS,
            obs_body_dof_vel * 0.05,
            state.last_action,
        ]
    ).astype(np.float32)
    obs_full = np.concatenate([np.asarray(mimic_obs, dtype=np.float32), obs_proprio]).astype(np.float32)
    obs_hist = np.array(state.history, dtype=np.float32).reshape(-1)
    state.history.append(obs_full.copy())
    obs_buf = np.concatenate([obs_full, obs_hist, np.asarray(mimic_obs, dtype=np.float32)]).astype(np.float32)
    if obs_buf.shape[0] != 1432:
        raise ValueError(f"Expected TWIST2 obs dim 1432, got {obs_buf.shape[0]}")
    return obs_buf


def _step_env_with_twist2_actions(
    env,
    actions: np.ndarray,
    mimic_obs: np.ndarray,
    *,
    arm_source: str,
    body_source: str,
) -> None:
    actions = np.asarray(actions, dtype=np.float64).reshape(env.num_envs, 29)
    mimic_obs = np.asarray(mimic_obs, dtype=np.float64).reshape(env.num_envs, 35)
    start = 0
    for task in env.tasks:
        stop = start + task.num_envs
        _step_task_with_twist2_actions(
            task,
            actions[start:stop],
            mimic_obs[start:stop],
            arm_source=arm_source,
            body_source=body_source,
        )
        env.episode_length_buf[start:stop] += 1
        start = stop


def _step_task_with_twist2_actions(
    task,
    actions: np.ndarray,
    mimic_obs: np.ndarray,
    *,
    arm_source: str,
    body_source: str,
) -> None:
    actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(task.num_envs, 29), -10.0, 10.0)
    mimic_obs = np.asarray(mimic_obs, dtype=np.float64).reshape(task.num_envs, 35)
    for index, agent in enumerate(task.agents):
        agent.previous_action = agent.last_action.copy()
        agent.last_action = actions[index].copy()

    target_qpos = DEFAULT_DOF_POS.reshape(1, -1) + actions * ACTION_SCALE.reshape(1, -1)
    body_idx = np.setdiff1d(np.arange(29, dtype=np.int64), ARM_IDX)
    if body_source == "default":
        target_qpos[:, body_idx] = DEFAULT_DOF_POS[body_idx]
    elif body_source == "mimic":
        target_qpos[:, body_idx] = mimic_obs[:, 6 + body_idx]
    if arm_source == "mimic":
        target_qpos[:, ARM_IDX] = mimic_obs[:, 6 + ARM_IDX]
    elif arm_source == "default":
        target_qpos[:, ARM_IDX] = DEFAULT_DOF_POS[ARM_IDX]
    target_qpos = np.clip(target_qpos, task._joint_limit_low, task._joint_limit_high)
    if task._mjwarp_runtime is not None:
        for _ in range(task.decimation):
            qpos = task.data.qpos[task._leg_qpos_indices]
            qvel = task.data.qvel[task._leg_qvel_indices]
            torque = TWIST2_STIFFNESS.reshape(1, -1) * (target_qpos - qpos) - TWIST2_DAMPING.reshape(1, -1) * qvel
            torque = np.clip(torque, -TWIST2_TORQUE_LIMITS.reshape(1, -1), TWIST2_TORQUE_LIMITS.reshape(1, -1))
            task.ctrl[:] = 0.0
            task.ctrl[task._flat_actuator_ids] = torque.reshape(-1)
            task._mjwarp_runtime.set_ctrl(task.ctrl)
            task._mjwarp_runtime.step(nstep=task.frame_skip)
        task._mjwarp_runtime.sync_to_cpu(task.gym._mjData, recompute_contacts=True)
        task.update_data()
    else:
        mj_data = getattr(task.gym, "_mjData", None)
        for _ in range(task.decimation * task.frame_skip):
            if mj_data is not None:
                qpos = mj_data.qpos[task._leg_qpos_indices]
                qvel = mj_data.qvel[task._leg_qvel_indices]
            else:
                qpos = task.data.qpos[task._leg_qpos_indices]
                qvel = task.data.qvel[task._leg_qvel_indices]
            torque = TWIST2_STIFFNESS.reshape(1, -1) * (target_qpos - qpos) - TWIST2_DAMPING.reshape(1, -1) * qvel
            torque = np.clip(torque, -TWIST2_TORQUE_LIMITS.reshape(1, -1), TWIST2_TORQUE_LIMITS.reshape(1, -1))
            task.ctrl[:] = 0.0
            task.ctrl[task._flat_actuator_ids] = torque.reshape(-1)
            task.set_ctrl(task.ctrl)
            task.mj_step(nstep=1)
        task.update_data()
    if task._render_mode == "human":
        task.render()
    task.episode_lengths += 1


def _quat_to_euler_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _policy_ramp(start_time: float, *, seconds: float) -> float:
    duration = max(0.0, float(seconds))
    if duration <= 1.0e-6:
        return 1.0
    return _clip((time.perf_counter() - start_time) / duration, 0.0, 1.0)


def _read_terminal_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    return sys.stdin.read(1)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _print_status(env, mimic_obs: np.ndarray, actions: np.ndarray) -> None:
    task = env.tasks[0]
    base = task.data.qpos[task._base_qpos_indices[0]].copy()
    rpy = _quat_to_euler_wxyz(base[3:7])
    action_norm = float(np.linalg.norm(np.asarray(actions, dtype=np.float64).reshape(env.num_envs, 29)[0]))
    sys.stdout.write(
        f"\rz={base[2]:.3f} roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} "
        f"cmd=({mimic_obs[0]:+.2f},{mimic_obs[1]:+.2f},{mimic_obs[5]:+.2f}) "
        f"target_z={mimic_obs[2]:.2f} action_norm={action_norm:.2f}   "
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
