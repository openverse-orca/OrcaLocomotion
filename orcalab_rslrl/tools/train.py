from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .common import call_with_supported_kwargs, load_symbol, load_yaml
from .torch_backends import configure_torch_backends


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Orca training with RSL-RL")
    parser.add_argument("--task", default="G1-Velocity-Flat", help="Registered task id")
    parser.add_argument("--task-factory", help="Import path package.module:factory")
    parser.add_argument("--asset", help="Optional local robot model override")
    parser.add_argument("--runner-config", default="configs/train/ppo.yaml")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--log-dir", default="logs/rsl_rl")
    parser.add_argument("--resume")
    parser.add_argument("--video-hook", help="Optional package.module:func that logs an evaluation video")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--check-for-nan", action="store_true", help="Enable per-step NaN checks; slower due GPU sync")
    parser.add_argument("--disable-fast-rollout-logger", action="store_true", help="Use stock RSL-RL per-step logger")
    args = parser.parse_args()
    configure_torch_backends()
    if args.num_envs < 2:
        raise ValueError("Training requires --num-envs >= 2; use play for one world")
    os.environ["WANDB_MODE"] = args.wandb_mode
    cfg = load_yaml(args.runner_config)
    cfg["check_for_nan"] = bool(args.check_for_nan)
    iterations = args.iterations or int(cfg.pop("max_iterations", 1500))
    if args.task_factory:
        factory = load_symbol(args.task_factory)
    else:
        from ..tasks import make_task

        factory = lambda **kwargs: make_task(args.task, **kwargs)
    env = call_with_supported_kwargs(
        factory,
        num_envs=args.num_envs,
        device=args.device,
        headless=True,
        mjcf_path=args.asset,
    )
    from rsl_rl.runners import OnPolicyRunner
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=args.device)
    if not args.disable_fast_rollout_logger:
        from ..rslrl.fast_logger import install_fast_rollout_logger

        install_fast_rollout_logger(runner)
    if args.resume:
        runner.load(args.resume, map_location=args.device)
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        final_checkpoint = log_dir / "model_final.pt"
        payload = runner.alg.save()
        payload["iter"] = runner.current_learning_iteration
        payload["infos"] = {"saved_by": "orcalab-rslrl"}
        torch.save(payload, final_checkpoint)
        print(f"Saved final checkpoint: {final_checkpoint}")
        if args.video_hook:
            load_symbol(args.video_hook)(runner=runner, step=runner.current_learning_iteration)
    finally:
        env.close()


if __name__ == "__main__":
    main()
