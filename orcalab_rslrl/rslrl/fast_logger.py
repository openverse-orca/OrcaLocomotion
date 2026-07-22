from __future__ import annotations

import types

import torch


def install_fast_rollout_logger(runner) -> None:
    """Keep RSL-RL iteration logs identical while removing per-step CPU syncs.

    The stock ``Logger.process_env_step`` calls ``dones.nonzero()`` +
    ``.cpu()`` every step. This patch accumulates episode reward/length of
    finished episodes on GPU and syncs once per iteration inside ``log()``,
    so ``Train/mean_reward`` / ``Mean episode length`` (console and W&B) are
    still populated, and ``extras[\"log\"]`` terms are forwarded untouched.
    """

    logger = runner.logger
    if logger.cfg["algorithm"]["rnd_cfg"]:
        return

    state: dict[str, torch.Tensor | None] = {"rew": None, "len": None, "done_rew": None, "done_len": None, "done_cnt": None}

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        if getattr(self, "writer", None) is None:
            return
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])
        rew = rewards.detach().reshape(-1)
        if state["rew"] is None:
            state["rew"] = torch.zeros_like(rew)
            state["len"] = torch.zeros_like(rew)
            state["done_rew"] = torch.zeros((), device=rew.device)
            state["done_len"] = torch.zeros((), device=rew.device)
            state["done_cnt"] = torch.zeros((), device=rew.device)
        state["rew"] += rew
        state["len"] += 1
        done = dones.detach().reshape(-1).bool()
        done_f = done.float()
        state["done_rew"] += (state["rew"] * done_f).sum()
        state["done_len"] += (state["len"] * done_f).sum()
        state["done_cnt"] += done_f.sum()
        state["rew"] = torch.where(done, torch.zeros_like(state["rew"]), state["rew"])
        state["len"] = torch.where(done, torch.zeros_like(state["len"]), state["len"])

    original_log = logger.log

    def log(self, *args, **kwargs):
        # One GPU->CPU sync per iteration: fold accumulated episode stats into
        # the deques the stock logger reads for console + W&B output.
        if state["done_cnt"] is not None:
            count = float(state["done_cnt"].item())
            if count > 0:
                self.rewbuffer.append(float(state["done_rew"].item()) / count)
                self.lenbuffer.append(float(state["done_len"].item()) / count)
                # Reassign instead of zero_(): the accumulators are inference
                # tensors (created inside the rollout's inference_mode).
                state["done_rew"] = torch.zeros_like(state["done_rew"])
                state["done_len"] = torch.zeros_like(state["done_len"])
                state["done_cnt"] = torch.zeros_like(state["done_cnt"])
        return original_log(*args, **kwargs)

    logger.process_env_step = types.MethodType(process_env_step, logger)
    logger.log = types.MethodType(log, logger)
