from __future__ import annotations

import argparse
import time

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    find_latest_checkpoint,
    load_task_and_train_cfg,
)

ensure_project_root_on_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Orca locomotion RSL-RL policy.")
    parser.add_argument(
        "--config",
        default="Unitree-GO2-Flat",
        help="Registered task name, or Python cfg file. Use file.py:factory_name for a non-default factory.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List registered task names and exit.")
    parser.add_argument("--ckpt", default=None, help="RSL-RL checkpoint path, e.g. model_1000.pt.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument(
        "--remote",
        default=None,
        help="Override OrcaGym address, e.g. localhost:50051.",
    )
    args = parser.parse_args()

    if args.list_tasks:
        from orca_rl.registry import list_tasks

        for spec in list_tasks():
            suffix = f" - {spec.description}" if spec.description else ""
            print(f"{spec.name}{suffix}")
        return

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    apply_remote_override(task_cfg, args.remote)
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    check_orcagym_addresses(task_cfg)

    try:
        import torch

        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.rsl_env.runtime_policy import load_inference_runner
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    checkpoint = args.ckpt or str(
        find_latest_checkpoint(task_name=str(train_cfg.get("experiment_name") or task_cfg.get("name", "")) or None)
    )
    try:
        env = make_locomotion_vec_env(task_cfg, device=device, render_mode="human", headless=False)
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc
    try:
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=checkpoint,
        )
        _runner, policy = load_inference_runner(env, train_cfg, checkpoint, log_dir=None, device=device)
        obs = env.get_observations().to(device)
        step = 0
        dt = float(task_cfg["sim"]["time_step"]) * int(task_cfg["sim"]["frame_skip"]) * int(task_cfg["sim"]["decimation"])
        while args.steps <= 0 or step < args.steps:
            start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
            obs, _rewards, _dones, _extras = env.step(actions.to(env.device))
            obs = obs.to(device)
            step += 1
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Interrupted RSL-RL playback.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
