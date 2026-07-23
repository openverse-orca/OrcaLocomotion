from __future__ import annotations

import torch

from ..managers.terms import ObservationTermCfg
from ..orca.physics import OrcaPhysics
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv:
    """Owns the fixed step/reset lifecycle; tasks only configure terms."""

    def __init__(self, orca: OrcaPhysics, cfg: ManagerBasedRLEnvCfg):
        if cfg.decimation < 1 or cfg.episode_length_steps < 1:
            raise ValueError("decimation and episode_length_steps must be positive")
        self.orca, self.cfg = orca, cfg
        self.num_envs, self.device = orca.num_envs, orca.device
        self.physics_dt = float(getattr(orca, "timestep", 0.0))
        self.step_dt = self.physics_dt * cfg.decimation
        self.max_episode_length_s = cfg.episode_length_steps * self.step_dt
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Raw (unscaled) policy actions, Orca convention for obs and action-rate terms.
        self.last_action = torch.zeros_like(orca.state.ctrl)
        self.prev_action = torch.zeros_like(orca.state.ctrl)
        scale = cfg.action_scale
        self.action_scale = (
            torch.as_tensor(scale, dtype=torch.float32, device=self.device)
            if not isinstance(scale, (int, float))
            else torch.full_like(self.last_action[0], float(scale))
        )
        self.commands = {
            name: torch.zeros(self.num_envs, command.dim, device=self.device)
            for name, command in cfg.commands.items()
        }
        self.command_time_left = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name, command in cfg.commands.items()
            if command.resampling_time_range is not None
        }
        self.terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._reward_episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in cfg.rewards
        }
        self._interval_time_left = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name, event in cfg.events.items()
            if event.mode == "interval"
        }
        for name, event in cfg.events.items():
            if event.mode == "interval" and event.interval_range_s is None:
                raise ValueError(f"interval event {name!r} needs interval_range_s")
            if event.mode == "startup":
                event.func(self, None)
        for name in self._interval_time_left:
            self._interval_time_left[name].uniform_(*self.cfg.events[name].interval_range_s)
        self.extras: dict = {"log": {}}
        self.reset()

    def observations(self) -> dict[str, torch.Tensor]:
        # Actor and critic commonly share the same ObservationTermCfg object:
        # the actor applies noise while the critic consumes the clean value.
        # Cache only those raw values, before group-specific corruption/scale.
        raw_values: dict[int, torch.Tensor] = {}
        groups = {}
        for name, group in self.cfg.observations.items():
            values = []
            for term in group.terms.values():
                if isinstance(term, ObservationTermCfg):
                    cache_key = id(term)
                    value = raw_values.get(cache_key)
                    if value is None:
                        value = term.func(self)
                        raw_values[cache_key] = value
                    if group.enable_corruption and term.noise is not None:
                        value = value + torch.empty_like(value).uniform_(*term.noise)
                    if term.scale is not None:
                        value = value * term.scale
                else:
                    value = term(self)
                values.append(value)
            groups[name] = torch.cat(values, dim=-1) if group.concatenate_terms else values
        return groups

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device, torch.long)
        self.orca.reset(ids)
        for event in self.cfg.reset_events:
            event(self, ids)
        for name, event in self.cfg.events.items():
            if event.mode == "reset":
                event.func(self, ids)
        for name, command in self.cfg.commands.items():
            self.commands[name][ids] = command.func(self, ids).to(self.device)
            if name in self.command_time_left:
                self.command_time_left[name][ids] = torch.empty(
                    ids.numel(), device=self.device
                ).uniform_(*command.resampling_time_range)
        self.episode_length_buf[ids] = 0
        self.episode_reward_buf[ids] = 0
        self.last_action[ids] = 0
        self.prev_action[ids] = 0
        self.orca.forward()
        for name, command in self.cfg.commands.items():
            if command.update is not None:
                command.update(self, name)
        return self.observations()

    def _apply_action(self, actions: torch.Tensor) -> None:
        self.prev_action.copy_(self.last_action)
        self.last_action.copy_(actions)
        processed = actions
        if self.cfg.clip_actions is not None:
            processed = processed.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        processed = processed * self.action_scale
        if self.cfg.action_offset:
            processed = processed + self.orca.default_actuated_qpos()
        self.orca.write_action(processed)

    def _update_commands(self) -> None:
        for name, command in self.cfg.commands.items():
            if name in self.command_time_left:
                time_left = self.command_time_left[name]
                time_left -= self.step_dt
                expired = time_left <= 0.0
                sampler = getattr(getattr(command.func, "__self__", None), "sample_masked", None)
                if sampler is not None:
                    # The built-in velocity command avoids CUDA ``nonzero`` here;
                    # see UniformVelocityCommand.sample_masked().
                    sampled = sampler(self, expired).to(self.device)
                    self.commands[name].copy_(torch.where(expired[:, None], sampled, self.commands[name]))
                else:
                    # Third-party commands retain indexed, sample-only-expired
                    # semantics until they opt into ``sample_masked``.
                    expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
                    if expired_ids.numel():
                        self.commands[name][expired_ids] = command.func(self, expired_ids).to(self.device)
                fresh = torch.empty(self.num_envs, device=self.device).uniform_(
                    *command.resampling_time_range
                )
                time_left.copy_(torch.where(expired, fresh, time_left))
            if command.update is not None:
                command.update(self, name)

    def _update_interval_events(self) -> None:
        for name, time_left in self._interval_time_left.items():
            event = self.cfg.events[name]
            time_left -= self.step_dt
            expired = time_left <= 0.0
            event.func(self, expired)
            fresh = torch.empty(self.num_envs, device=self.device).uniform_(*event.interval_range_s)
            time_left.copy_(torch.where(expired, fresh, time_left))

    def step(self, actions: torch.Tensor):
        if actions.shape != self.last_action.shape:
            raise ValueError(f"actions {tuple(actions.shape)} != {tuple(self.last_action.shape)}")
        self.extras = {"log": {}}
        self._apply_action(actions.to(self.device))
        for _ in range(self.cfg.decimation):
            self.orca.step()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        for name, event in self.cfg.events.items():
            if event.mode == "post_step":
                event.func(self, None)
        self._update_interval_events()
        self._update_commands()

        # Terminations first so reward terms may read env.terminated_buf (Orca order).
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.cfg.episode_length_steps
        termination_values = {}
        for name, term_cfg in self.cfg.terminations.items():
            value = term_cfg.func(self).bool()
            termination_values[name] = value
            if term_cfg.time_out:
                truncated |= value
            else:
                terminated |= value
        self.terminated_buf, self.truncated_buf = terminated, truncated

        reward_scale = self.step_dt if self.cfg.reward_scale_by_dt else 1.0
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, term_cfg in self.cfg.rewards.items():
            value = term_cfg.func(self) * (term_cfg.weight * reward_scale)
            self._reward_episode_sums[name] += value
            reward += value
        self.episode_reward_buf += reward

        final_observation = self.observations()
        next_observation = final_observation
        done = terminated | truncated
        log = self.extras["log"]
        if self.cfg.gpu_done_reset:
            reset_observation = self._reset_done_mask(done)
            next_observation = {
                name: torch.where(done[:, None], reset_observation[name], value)
                for name, value in final_observation.items()
            }
        elif done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self._log_finished_episodes(done_ids, termination_values, log)
            reset_observation = self.reset(done_ids)
            next_observation = {name: value.clone() for name, value in final_observation.items()}
            for name, value in next_observation.items():
                value[done] = reset_observation[name][done]
        info = {
            "time_outs": truncated,
            "final_observation": final_observation,
            "log": log,
        }
        return next_observation, reward, terminated, truncated, info

    def _log_finished_episodes(
        self,
        done_ids: torch.Tensor,
        termination_values: dict[str, torch.Tensor],
        log: dict,
    ) -> None:
        """Orca-style episode metrics: per-term reward sums and termination counts."""
        denom = self.max_episode_length_s if self.max_episode_length_s > 0 else 1.0
        for name, sums in self._reward_episode_sums.items():
            log[f"Episode_Reward/{name}"] = sums[done_ids].mean() / denom
            sums[done_ids] = 0.0
        for name, value in termination_values.items():
            log[f"Episode_Termination/{name}"] = torch.count_nonzero(value[done_ids]).float()

    def get_command(self, name: str) -> torch.Tensor:
        return self.commands[name]

    def get_termination(self, name: str) -> torch.Tensor:
        return self.terminated_buf

    def _reset_done_mask(self, done: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.cfg.reset_events or any(e.mode == "reset" for e in self.cfg.events.values()):
            raise RuntimeError("gpu_done_reset does not support reset events; use indexed reset")
        self.orca.reset_mask(done)
        for name, command in self.cfg.commands.items():
            if command.dim and command.resample_on_done_mask:
                candidate = command.func(self, torch.arange(self.num_envs, device=self.device)).to(self.device)
                self.commands[name].copy_(torch.where(done[:, None], candidate, self.commands[name]))
        self.episode_length_buf.copy_(torch.where(done, torch.zeros_like(self.episode_length_buf), self.episode_length_buf))
        self.episode_reward_buf.copy_(torch.where(done, torch.zeros_like(self.episode_reward_buf), self.episode_reward_buf))
        for sums in self._reward_episode_sums.values():
            sums.copy_(torch.where(done, torch.zeros_like(sums), sums))
        self.last_action.copy_(torch.where(done[:, None], torch.zeros_like(self.last_action), self.last_action))
        self.prev_action.copy_(torch.where(done[:, None], torch.zeros_like(self.prev_action), self.prev_action))
        return self.observations()
