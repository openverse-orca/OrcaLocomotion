from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orca_rl.utils import ensure_project_root_on_path, explain_missing_runtime_dependency, load_task_and_train_cfg


ensure_project_root_on_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/test_model_G1_mjlab_Flat.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets/heft/recorded_commands"


@dataclass(frozen=True)
class CommandClip:
    key: str
    name: str
    command: tuple[float, float, float]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record separate 50 Hz W/S/A/D/Z/C motion clips from the original G1 29-DoF mjlab policy."
    )
    parser.add_argument("--config", default="Unitree-G1-Flat")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--linear-speed", type=float, default=0.5)
    parser.add_argument("--yaw-speed", type=float, default=0.8)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--record-seconds", type=float, default=6.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from orca_rl import make_locomotion_vec_env
        from orca_rl.run_play import _apply_play_scene_mode
        from orca_rl.rsl_env.heft_policy import HEFT_JOINT_NAMES
        from orca_rl.rsl_env.mjlab_policy import MjlabRslRlActorPolicy, make_mjlab_orca_play_bridge
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    linear_speed = abs(float(args.linear_speed))
    yaw_speed = abs(float(args.yaw_speed))
    clips = (
        CommandClip("0", "g1_mjlab_stand", (0.0, 0.0, 0.0)),
        CommandClip("w", "g1_mjlab_w_forward", (linear_speed, 0.0, 0.0)),
        CommandClip("s", "g1_mjlab_s_backward", (-linear_speed, 0.0, 0.0)),
        CommandClip("a", "g1_mjlab_a_left", (0.0, linear_speed, 0.0)),
        CommandClip("d", "g1_mjlab_d_right", (0.0, -linear_speed, 0.0)),
        CommandClip("z", "g1_mjlab_z_yaw_left", (0.0, 0.0, yaw_speed)),
        CommandClip("c", "g1_mjlab_c_yaw_right", (0.0, 0.0, -yaw_speed)),
    )

    task_cfg, _ = load_task_and_train_cfg(args.config)
    task_cfg["num_envs"] = 1
    _apply_play_scene_mode(task_cfg, local_mujoco=True)
    task_cfg.setdefault("sim", {})["render_mode"] = "none"
    task_cfg["sim"]["headless"] = True
    env = make_locomotion_vec_env(task_cfg, device=args.device, render_mode="none", headless=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        policy = MjlabRslRlActorPolicy.from_checkpoint(args.checkpoint, device=args.device)
        bridge = make_mjlab_orca_play_bridge(
            env,
            expected_obs_dim=policy.input_dim,
            robot="g1",
            g1_arm_mode="policy",
        )
        dt = float(env.tasks[0].control_dt)
        if not np.isclose(dt, 0.02):
            raise ValueError(f"HEFT command recordings require 50 Hz control, got dt={dt:.6f}s")
        settle_frames = max(0, int(round(float(args.settle_seconds) / dt)))
        warmup_frames = max(0, int(round(float(args.warmup_seconds) / dt)))
        record_frames = max(2, int(round(float(args.record_seconds) / dt)))

        for clip in clips:
            env.reset()
            _run_frames(env, bridge, policy, np.zeros(3), settle_frames, device=args.device)
            command = np.asarray(clip.command, dtype=np.float64)
            _run_frames(env, bridge, policy, command, warmup_frames, device=args.device)
            joint_pos, root_quat, root_pos = _capture_frames(
                env,
                bridge,
                policy,
                command,
                record_frames,
                joint_names=HEFT_JOINT_NAMES,
                device=args.device,
            )
            _save_heft_clip(
                output_dir / f"{clip.name}.npz",
                joint_names=HEFT_JOINT_NAMES,
                joint_pos=joint_pos,
                root_quat_wxyz=root_quat,
                root_pos=root_pos,
                command=command,
                key=clip.key,
            )
            displacement = root_pos[-1, :2] - root_pos[0, :2]
            yaw_delta = _unwrap_yaw(root_quat)[-1] - _unwrap_yaw(root_quat)[0]
            duration = max(dt, (root_pos.shape[0] - 1) * dt)
            print(
                f"[record] {clip.key.upper():>2} {clip.name}: frames={root_pos.shape[0]} "
                f"vxy=({displacement[0] / duration:+.3f},{displacement[1] / duration:+.3f})m/s "
                f"wz={yaw_delta / duration:+.3f}rad/s"
            )
    finally:
        env.close()


def _set_command(env, command: np.ndarray) -> None:
    command_batch = np.repeat(np.asarray(command, dtype=np.float64).reshape(1, 3), env.num_envs, axis=0)
    start = 0
    for task in env.tasks:
        stop = start + task.num_envs
        task.set_manual_commands(command_batch[start:stop])
        start = stop


def _policy_step(env, bridge, policy, command: np.ndarray, *, device: str) -> None:
    _set_command(env, command)
    observations = bridge.get_observations()
    actions = policy.act_numpy(observations, device=device)
    bridge.step(actions)


def _run_frames(env, bridge, policy, command: np.ndarray, frames: int, *, device: str) -> None:
    for _ in range(int(frames)):
        _policy_step(env, bridge, policy, command, device=device)


def _capture_frames(
    env,
    bridge,
    policy,
    command: np.ndarray,
    frames: int,
    *,
    joint_names: tuple[str, ...],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = env.tasks[0]
    agent = task.agents[0]
    configured_names = list(task.robot_config.get("leg_joint_names") or ())
    reorder = np.asarray([configured_names.index(name) for name in joint_names], dtype=np.int64)
    joint_frames: list[np.ndarray] = []
    quat_frames: list[np.ndarray] = []
    position_frames: list[np.ndarray] = []
    for _ in range(int(frames)):
        _policy_step(env, bridge, policy, command, device=device)
        base_qpos = np.asarray(task.data.qpos[task._base_qpos_indices[0]], dtype=np.float64)
        joint_qpos = np.asarray(task.data.qpos[agent.leg_qpos_indices], dtype=np.float64)[reorder]
        if not np.isfinite(base_qpos).all() or not np.isfinite(joint_qpos).all():
            raise FloatingPointError("The original G1 policy produced a non-finite recording frame.")
        joint_frames.append(joint_qpos.astype(np.float32))
        quat_frames.append(_normalize_quat(base_qpos[3:7]).astype(np.float32))
        position_frames.append(base_qpos[:3].astype(np.float32))
    return np.stack(joint_frames), np.stack(quat_frames), np.stack(position_frames)


def _save_heft_clip(
    path: Path,
    *,
    joint_names: tuple[str, ...],
    joint_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_pos: np.ndarray,
    command: np.ndarray,
    key: str,
) -> None:
    # HeftMotionLibrary follows the upstream end=-1 convention, so append one
    # duplicate terminal frame that the loader intentionally drops.
    dof_pos = np.concatenate([joint_pos, joint_pos[-1:]], axis=0)
    root_pos_saved = np.concatenate([root_pos, root_pos[-1:]], axis=0)
    root_xyzw = np.concatenate([root_quat_wxyz[:, 1:], root_quat_wxyz[:, :1]], axis=1)
    root_rot = np.concatenate([root_xyzw, root_xyzw[-1:]], axis=0)
    np.savez_compressed(
        path,
        fps=np.asarray(50.0, dtype=np.float32),
        joint_names=np.asarray(joint_names),
        dof_pos=dof_pos.astype(np.float32),
        root_pos=root_pos_saved.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        command=np.asarray(command, dtype=np.float32),
        key=np.asarray(str(key)),
        source=np.asarray("orca_rl original G1 29DoF mjlab policy"),
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-9)


def _unwrap_yaw(quaternions: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.unwrap(yaw)


if __name__ == "__main__":
    main()
