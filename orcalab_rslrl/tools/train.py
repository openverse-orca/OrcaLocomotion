from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .common import call_with_supported_kwargs, load_symbol, load_yaml
from .torch_backends import configure_torch_backends


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Orca task with RSL-RL")
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
    parser.add_argument("--orcalab", action="store_true", help="Render training live in OrcaLab")
    parser.add_argument("--orca-addr", default="localhost:50051", help="OrcaLab bridge address")
    parser.add_argument(
        "--asset-path",
        default="assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda",
        help="Subscribed OrcaLab robot asset",
    )
    parser.add_argument(
        "--render-num-envs",
        type=int,
        default=16,
        help="Number of training environments shown in OrcaLab",
    )
    parser.add_argument("--render-fps", type=float, default=30.0, help="Maximum OrcaLab update rate")
    parser.add_argument("--agent-prefix", default="g1")
    parser.add_argument("--spacing", type=float, default=2.5, help="Grid spacing between rendered actors")
    parser.add_argument("--spawn-range", type=float, help="Optional half-width for the rendered actor layout")
    parser.add_argument(
        "--root-xy-scale",
        type=float,
        default=1.0,
        help="Visual-only root x/y scale used to keep rendered robots together",
    )
    parser.add_argument("--no-publish", action="store_true", help="Reuse an existing OrcaLab scene")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    configure_torch_backends()
    if args.num_envs < 2:
        raise ValueError("Training requires --num-envs >= 2; use play for one world")
    if args.render_num_envs < 1:
        raise ValueError("--render-num-envs must be positive")
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be positive")
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
    render_hook = None
    try:
        if args.orcalab:
            from .train_render import attach_orcalab_training_renderer

            render_hook = attach_orcalab_training_renderer(
                env,
                orca_addr=args.orca_addr,
                asset_path=args.asset_path,
                render_num_envs=args.render_num_envs,
                render_fps=args.render_fps,
                agent_prefix=args.agent_prefix,
                spacing=args.spacing,
                spawn_range=args.spawn_range,
                root_xy_scale=args.root_xy_scale,
                publish=not args.no_publish,
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
        try:
            if render_hook is not None:
                render_hook.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
