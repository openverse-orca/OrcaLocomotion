"""Drive HEFT from a full XRoboToolkit PICO skeleton with arms-only output."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from orca_rl.play_g1_heft_velocity import DEFAULT_POLICY, _configure_heft_scene
from orca_rl.heft_keyboard import make_keyboard_backend
from orca_rl.pico_upper_body import (
    G1SkeletonArmsRetargeter,
    HEFT_JOINT_NAMES,
    PicoUpperBodyStream,
    PicoUpperBodyViser,
    load_xrobotoolkit_sdk,
)
from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
)


ensure_project_root_on_path()


def _smoothstep(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def hand_forward_reference(
    standing_reference: np.ndarray,
    *,
    side: str,
    elapsed_s: float,
    ramp_seconds: float,
) -> np.ndarray:
    """A conservative single-arm forward reach for PICO-offline smoke tests.

    It starts from the captured HEFT standing pose, modifies only one shoulder
    pitch joint, and uses a smooth two-second ramp so the online policy does
    not receive a reference discontinuity.
    """
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    reference = np.asarray(standing_reference, dtype=np.float32).reshape(-1).copy()
    if reference.shape != (len(HEFT_JOINT_NAMES),):
        raise ValueError(f"standing_reference must have shape (29,), got {reference.shape}")
    target = reference.copy()
    # In the HEFT/G1 convention this moves the straight hanging arm toward
    # +X (forward in the browser diagnostic) without changing the legs, waist
    # or opposite arm.  The runtime clamps it to the model's physical limit.
    joint_index = HEFT_JOINT_NAMES.index(f"{normalized_side}_shoulder_pitch_joint")
    target[joint_index] -= 1.20
    alpha = _smoothstep(float(elapsed_s) / max(float(ramp_seconds), 1e-3))
    return reference + alpha * (target - reference)


_DEMO_ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)


def arm_sequence_reference(
    standing_reference: np.ndarray,
    *,
    side: str,
    elapsed_s: float,
    duration_s: float,
) -> tuple[np.ndarray, str]:
    """Return a smooth, conservative 7-DoF single-arm 60-second test motion."""
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    reference = np.asarray(standing_reference, dtype=np.float32).reshape(-1).copy()
    if reference.shape != (len(HEFT_JOINT_NAMES),):
        raise ValueError(f"standing_reference must have shape (29,), got {reference.shape}")
    duration = max(float(duration_s), 1.0)
    # Times are normalized so --demo-sequence-seconds may change the total
    # duration without changing the motion's proportions. Values are deltas
    # from the captured standing reference in the named arm-joint order.
    right_keyframes = (
        (0.00, "stand", (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00)),
        # The first four components deliberately lead from the shoulder and
        # then straighten the elbow.  This makes the upper arm visibly leave
        # the torso instead of producing a mostly wrist/elbow-only motion.
        (0.10, "shoulder_prepare", (-0.52, -0.14, -0.06, 0.06, 0.00, 0.05, 0.00)),
        (0.22, "shoulder_forward", (-1.20, -0.10, -0.14, 0.38, 0.10, 0.18, -0.08)),
        (0.36, "shoulder_sweep", (-1.10, -0.48, 0.18, 0.34, -0.28, -0.16, 0.32)),
        (0.50, "shoulder_high", (-1.42, -0.16, -0.28, 0.18, 0.32, 0.32, -0.25)),
        (0.64, "wrist_circle", (-1.26, 0.12, 0.22, 0.30, -0.46, -0.28, 0.42)),
        (0.78, "present_forward", (-1.24, -0.06, -0.08, 0.48, 0.08, 0.14, 0.04)),
        (0.90, "retract", (-0.36, -0.02, 0.00, 0.06, 0.00, 0.03, 0.00)),
        (1.00, "stand", (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00)),
    )
    phase = min(max(float(elapsed_s) / duration, 0.0), 1.0)
    lower, upper = right_keyframes[-2], right_keyframes[-1]
    for candidate_lower, candidate_upper in zip(right_keyframes, right_keyframes[1:], strict=True):
        if phase <= candidate_upper[0]:
            lower, upper = candidate_lower, candidate_upper
            break
    span = max(float(upper[0] - lower[0]), 1e-6)
    blend = _smoothstep((phase - float(lower[0])) / span)
    right_delta = (1.0 - blend) * np.asarray(lower[2], dtype=np.float32) + blend * np.asarray(upper[2], dtype=np.float32)
    # Left/right arm frames are mirrored around the sagittal plane.  Pitch and
    # elbow flexion remain shared; roll, yaw and wrist roll/yaw mirror signs.
    if normalized_side == "left":
        mirror = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float32)
        delta = right_delta * mirror
    else:
        delta = right_delta
    for suffix, value in zip(_DEMO_ARM_JOINT_SUFFIXES, delta, strict=True):
        reference[HEFT_JOINT_NAMES.index(f"{normalized_side}_{suffix}")] += value
    return reference, str(upper[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-skeleton XRoboToolkit PICO retargeting with standing-body arms-only HEFT output."
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--remote", default=None, help="OrcaGym address override, e.g. localhost:50051.")
    parser.add_argument("--onnx-threads", type=int, default=4)
    parser.add_argument("--position-scale", type=float, default=0.8, help="PICO arm-displacement scale.")
    parser.add_argument("--ik-iterations", type=int, default=6)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--max-joint-step", type=float, default=0.08, help="Maximum arm-joint IK step in rad/frame.")
    parser.add_argument(
        "--max-reference-step",
        type=float,
        default=0.06,
        help="Maximum published arm-reference change in rad/PICO frame; protects HEFT from OOD jumps.",
    )
    parser.add_argument("--reference-horizon", type=int, default=7, help="HEFT online reference horizon in frames.")
    demo_mode = parser.add_mutually_exclusive_group()
    demo_mode.add_argument(
        "--demo-hand-forward",
        action="store_true",
        help="Run without PICO/XR SDK and send a smooth single-arm forward-reach reference to HEFT.",
    )
    demo_mode.add_argument(
        "--demo-arm-sequence",
        action="store_true",
        help="Run a no-PICO, smooth multi-phase single-arm reference sequence (default: 60 seconds).",
    )
    parser.add_argument(
        "--demo-side",
        choices=("left", "right"),
        default="right",
        help="Arm used by --demo-hand-forward (default: right).",
    )
    parser.add_argument(
        "--demo-ramp-seconds",
        type=float,
        default=2.0,
        help="Seconds to smoothly reach the demo pose (default: 2.0).",
    )
    parser.add_argument(
        "--demo-sequence-seconds",
        type=float,
        default=60.0,
        help="Duration of --demo-arm-sequence before it returns to standing (default: 60).",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the reference-style MJViser browser view (default: enabled).",
    )
    parser.add_argument("--viewer-host", default="0.0.0.0")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--viewer-fps", type=float, default=10.0)
    parser.add_argument(
        "--keyboard-backend",
        choices=("auto", "global", "terminal", "none"),
        default="auto",
        help="Keyboard source for Dex3 controls; O=open, C=close, Q/Esc=quit.",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Optional automatic stop for integration tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from orca_rl.heft_env import make_heft_env
        from orca_rl.rsl_env.heft_policy import HeftG1OrcaPlayBridge
        from orca_rl.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    demo_mode_active = bool(args.demo_hand_forward or args.demo_arm_sequence)
    xrt = None if demo_mode_active else load_xrobotoolkit_sdk()
    task_cfg = unitree_g1_flat_env_cfg(play=True).to_dict()
    apply_remote_override(task_cfg, args.remote)
    _configure_heft_scene(task_cfg)
    check_orcagym_addresses(task_cfg)

    env = make_heft_env(task_cfg)
    viewer = None
    stream = PicoUpperBodyStream()
    sdk_started = False
    clean_native_shutdown = False
    keyboard = make_keyboard_backend(
        str(args.keyboard_backend),
        initial_command=np.zeros(3, dtype=np.float64),
        command_speed=0.0,
        yaw_speed=0.0,
        max_speed=0.0,
        enable_hand_control=True,
    )
    try:
        bridge = HeftG1OrcaPlayBridge(
            env,
            policy_path=args.policy,
            motion_dir=None,
            onnx_threads=args.onnx_threads,
        )
        bridge.reset_online(reset_env=True, horizon=args.reference_horizon)
        retargeter = G1SkeletonArmsRetargeter(
            env.tasks[0],
            position_scale=args.position_scale,
            iterations=args.ik_iterations,
            damping=args.ik_damping,
            max_joint_step=args.max_joint_step,
            max_reference_step=args.max_reference_step,
        )
        retargeter.reset(bridge.current_joint_positions()[0])

        if args.visualize:
            viewer = PicoUpperBodyViser(
                retargeter.model,
                reference_joint_positions=retargeter.standing_reference,
                host=args.viewer_host,
                port=args.viewer_port,
            )

        if not demo_mode_active:
            assert xrt is not None
            xrt.init()
            sdk_started = True
            xrt.register_frame_callback(stream.on_frame)
        print("[orca_rl.pico] HEFT upper-body online reference ready")
        print(f"  robot=G1+Dex3-1 remote={task_cfg['orcagym_addresses'][0]}")
        if args.demo_arm_sequence:
            print(
                f"  DEMO: {args.demo_side} 7-DoF complex reach sequence for {args.demo_sequence_seconds:.1f}s; "
                "no PICO/XR SDK is used"
            )
        elif args.demo_hand_forward:
            print(
                f"  DEMO: {args.demo_side} arm smoothly reaches forward in {args.demo_ramp_seconds:.1f}s; "
                "no PICO/XR SDK is used"
            )
        else:
            print("  PICO input=full skeleton; output=14 arm qpos only; the other 15 G1 joints remain at startup standing pose")
        print(
            f"  Dex3-1={env.tasks[0].dex3_hand_pose} at start; keyboard={keyboard.name}; "
            "O=open, C=close, R=reset/recalibrate, Q/Esc=quit"
        )
        if not demo_mode_active:
            print("  hold a comfortable neutral standing pose for the first valid PICO frame to calibrate")
        if viewer is not None:
            print(f"  viewer_url=http://localhost:{args.viewer_port}")
            print(
                "  viewer shows the MuJoCo mesh when available; /teleop/reference_g1 "
                "is an always-visible red qpos diagnostic of the 14-DoF retarget output"
            )

        control_dt = float(env.tasks[0].control_dt)
        last_sequence = 0
        last_frame = None
        last_target = None
        last_status = 0.0
        last_view = 0.0
        start = time.perf_counter()
        max_runtime = args.seconds
        if max_runtime is None and args.demo_arm_sequence:
            max_runtime = max(float(args.demo_sequence_seconds), 1.0)
        demo_phase = "stand"
        with keyboard:
            while max_runtime is None or time.perf_counter() - start < float(max_runtime):
                step_start = time.perf_counter()
                keyboard_state = keyboard.poll()
                if keyboard_state.quit_requested:
                    print("\n[orca_rl.pico] quit requested")
                    break
                if keyboard_state.reset_requested:
                    bridge.reset_online(reset_env=True, horizon=args.reference_horizon)
                    retargeter.reset(bridge.current_joint_positions()[0])
                    last_frame = None
                    last_target = None
                    if demo_mode_active:
                        start = time.perf_counter()
                        print("\n[orca_rl.pico] reset; restarting the offline reference motion")
                    else:
                        print("\n[orca_rl.pico] reset; the next valid PICO frame will recalibrate the neutral pose")
                if keyboard_state.hand_command is not None:
                    changed = env.tasks[0].set_dex3_hand_pose(keyboard_state.hand_command)
                    if changed:
                        print(f"\n[orca_rl.pico] Dex3-1 -> {env.tasks[0].dex3_hand_pose}")

                if args.demo_arm_sequence:
                    target, demo_phase = arm_sequence_reference(
                        retargeter.standing_reference,
                        side=args.demo_side,
                        elapsed_s=time.perf_counter() - start,
                        duration_s=args.demo_sequence_seconds,
                    )
                    target = retargeter.clamp_arm_reference(target)
                    bridge.set_online_reference(target, horizon=args.reference_horizon)
                    last_target = target
                elif args.demo_hand_forward:
                    target = hand_forward_reference(
                        retargeter.standing_reference,
                        side=args.demo_side,
                        elapsed_s=time.perf_counter() - start,
                        ramp_seconds=args.demo_ramp_seconds,
                    )
                    target = retargeter.clamp_arm_reference(target)
                    bridge.set_online_reference(target, horizon=args.reference_horizon)
                    last_target = target
                else:
                    frame = stream.latest_after(last_sequence)
                    if frame is not None:
                        last_sequence = frame.sequence
                        target = retargeter.retarget(frame)
                        bridge.set_online_reference(target, horizon=args.reference_horizon)
                        last_frame = frame
                        last_target = target
                        if retargeter.calibrated and stream.accepted_frames == 1:
                            print("[orca_rl.pico] full-skeleton neutral pose calibrated; sending arms-only reference")

                actions = bridge.act()
                bridge.step(actions)

                now = time.perf_counter()
                if (
                    viewer is not None
                    and last_target is not None
                    and now - last_view >= 1.0 / max(args.viewer_fps, 1.0)
                ):
                    positions, rotations = (
                        retargeter.human_visualization(last_frame)
                        if last_frame is not None
                        else (None, None)
                    )
                    viewer.update(
                        retargeter.qpos_for_reference(last_target),
                        positions,
                        rotations,
                        reference_joint_positions=last_target,
                    )
                    last_view = now
                if now - last_status >= 1.0:
                    rejection = stream.last_rejection_reason
                    rejection_text = f" last_reject={rejection}" if rejection else ""
                    sys.stdout.write(
                        "\r[orca_rl.pico] "
                        f"mode={'demo-sequence' if args.demo_arm_sequence else ('demo' if args.demo_hand_forward else 'pico')} "
                        f"frames={stream.accepted_frames} rejected={stream.rejected_frames} "
                        f"calibrated={retargeter.calibrated} dex3={env.tasks[0].dex3_hand_pose} "
                        f"phase={demo_phase} max|action|={float(np.abs(actions).max()):5.2f}{rejection_text}   "
                    )
                    sys.stdout.flush()
                    last_status = now
                elapsed = time.perf_counter() - step_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
        clean_native_shutdown = True
    except KeyboardInterrupt:
        clean_native_shutdown = True
        print("\n[orca_rl.pico] stopped")
    finally:
        # Viser owns background threads.  Stop it before shutting down the
        # native XR SDK; closing the SDK while its callback can still be
        # observed from the viewer can abort in the C++ stream destructor.
        if viewer is not None:
            viewer.close()
        if sdk_started:
            try:
                assert xrt is not None
                xrt.clear_frame_callback()
            except Exception:
                pass
        # Keep the native SDK alive until process exit.  This matches the
        # reference teleop server.  Its ``close()`` calls PXREADeinit(), which
        # can race the SDK's gRPC feedback worker and segfault on this Linux
        # build; clearing the Python callback above is sufficient to stop
        # consuming frames in this process.
        env.close()
        if sdk_started and clean_native_shutdown:
            # The upstream callback SDK leaves its C++ feedback thread alive.
            # A regular CPython shutdown then calls that thread's destructor
            # and emits ``terminate called without an active exception``.
            # Exit after our Python-side resources are closed so the OS tears
            # down the native client atomically, without a core dump.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
