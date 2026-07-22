from __future__ import annotations

import argparse

import torch

from .common import call_with_supported_kwargs
from .torch_backends import configure_torch_backends


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a registered Orca task")
    parser.add_argument("--task", default="G1-Velocity-Flat")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset", help="Optional local robot model override")
    args = parser.parse_args()
    configure_torch_backends()

    from ..tasks import make_task

    env = call_with_supported_kwargs(
        lambda **kwargs: make_task(args.task, **kwargs),
        num_envs=args.num_envs,
        device=args.device,
        headless=True,
        mjcf_path=args.asset,
    )
    try:
        obs = env.get_observations()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        next_obs, reward, done, info = env.step(actions)
        print(f"task: {args.task}")
        print(f"num_envs: {env.num_envs}")
        print(f"num_actions: {env.num_actions}")
        print(f"obs: { {name: tuple(value.shape) for name, value in obs.items()} }")
        print(f"next_obs: { {name: tuple(value.shape) for name, value in next_obs.items()} }")
        print(f"reward: {tuple(reward.shape)} done: {tuple(done.shape)}")
        print(f"log_keys: {sorted(info['log'])}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
