from __future__ import annotations

import argparse

from orca_rl.utils import (
    apply_logging_overrides,
    apply_remote_override,
    check_orcagym_addresses,
    check_wandb_dependency,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    load_task_and_train_cfg,
    make_log_dir,
    save_checkpoint_aliases,
)

ensure_project_root_on_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Orca locomotion velocity tasks with RSL-RL.")
    parser.add_argument(
        "--config",
        default="orca_rl/tasks/velocity/config/go2/env_cfgs.py",
        help="Python cfg file. Use file.py:factory_name to select a non-default factory such as rough terrain.",
    )
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Optional RSL-RL checkpoint path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-name", default=None)
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
    parser.add_argument("--wandb", action="store_true", help="Shortcut for --logger wandb.")
    parser.add_argument("--wandb-project", default=None, help="W&B project name. Defaults to the runner config value.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/user/team. Sets WANDB_USERNAME for RSL-RL.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument(
        "--remote",
        default=None,
        help="Override OrcaGym address, e.g. localhost:50051. Comma-separated values create multiple envs.",
    )
    args = parser.parse_args()

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    apply_remote_override(task_cfg, args.remote)
    logger_override = "wandb" if args.wandb else args.logger
    apply_logging_overrides(
        train_cfg,
        logger=logger_override,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    check_wandb_dependency(train_cfg)
    check_orcagym_addresses(task_cfg)

    try:
        from rsl_rl.runners import OnPolicyRunner

        from orca_rl import make_locomotion_vec_env, print_runtime_summary
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    device = args.device or task_cfg.get("device", "cuda:0")
    iterations = int(args.num_iterations or task_cfg.get("train", {}).get("num_learning_iterations", 1500))
    log_dir = make_log_dir(args.log_name, task_name=str(task_cfg.get("name", "locomotion")))

    try:
        env = make_locomotion_vec_env(
            task_cfg,
            device=device,
            render_mode=task_cfg.get("sim", {}).get("render_mode", "none"),
        )
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc
    try:
        print_runtime_summary(
            mode="train",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            log_dir=log_dir,
            iterations=iterations,
        )
        runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=device)
        runner.add_git_repo_to_log(str(log_dir.parents[2]))
        if args.resume:
            runner.load(args.resume, map_location=device)
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        saved_aliases = save_checkpoint_aliases(log_dir, runner.current_learning_iteration)
        print("Saved final checkpoint aliases: " + ", ".join(str(path) for path in saved_aliases))
        if task_cfg.get("export", {}).get("enabled", True):
            from orca_rl.rsl_env.runtime_policy import export_policy

            export_policy(
                runner,
                log_dir / "exported",
                onnx=bool(task_cfg.get("export", {}).get("onnx", True)),
                jit=bool(task_cfg.get("export", {}).get("jit", True)),
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
