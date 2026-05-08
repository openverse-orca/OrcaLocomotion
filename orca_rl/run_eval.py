from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Evaluate a trained Orca locomotion RSL-RL flat velocity policy.")
    parser.add_argument("--config", default="orca_rl/tasks/velocity/config/go2/env_cfgs.py")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--remote",
        default=None,
        help="Override OrcaGym address, e.g. localhost:50051.",
    )
    args = parser.parse_args()

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    apply_remote_override(task_cfg, args.remote)
    check_orcagym_addresses(task_cfg)

    try:
        import torch

        from orca_rl.diagnostics import print_runtime_summary
        from orca_rl.rsl_env import make_locomotion_vec_env
        from orca_rl.rsl_env.runtime_policy import load_inference_runner
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    device = args.device or task_cfg.get("eval", {}).get("device", "cpu")
    checkpoint = args.ckpt or str(find_latest_checkpoint(task_name=str(task_cfg.get("name", "")) or None))
    task_cfg.setdefault("sim", {})["render_mode"] = "none"
    try:
        env = make_locomotion_vec_env(task_cfg, device=device, render_mode="none")
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc
    try:
        print_runtime_summary(
            mode="eval",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=checkpoint,
        )
        _runner, policy = load_inference_runner(env, train_cfg, checkpoint, log_dir=None, device=device)
        obs = env.get_observations().to(device)
        total_reward = torch.zeros(env.num_envs, device=env.device)
        total_dones = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for _ in range(args.steps):
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
            obs, rewards, dones, _extras = env.step(actions.to(env.device))
            obs = obs.to(device)
            total_reward += rewards
            total_dones += dones.long()
        print(f"Eval steps: {args.steps}")
        print(f"Mean reward per env: {total_reward.mean().item():.4f}")
        print(f"Total dones: {int(total_dones.sum().item())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
