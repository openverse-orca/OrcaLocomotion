from __future__ import annotations

import argparse
import time

import torch

from .common import call_with_supported_kwargs
from .torch_backends import configure_torch_backends


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel Orca environment stepping")
    parser.add_argument("--task", default="G1-Velocity-Flat")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
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
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    try:
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - start
        print(f"num_envs={env.num_envs} steps={args.steps} fps={env.num_envs * args.steps / elapsed:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
