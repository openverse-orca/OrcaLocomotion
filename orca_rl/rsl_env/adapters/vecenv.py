from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from ..locomotion_task import OrcaLocomotionTask
from ..scene_resolvers import resolve_scene_binding


class OrcaRslRlVecEnv(VecEnv):
    """Config-driven RSL-RL VecEnv adapter for Orca locomotion tasks."""

    def __init__(
        self,
        task_cfg: dict[str, Any],
        *,
        device: str = "cpu",
        render_mode: str | None = None,
    ) -> None:
        self.cfg = task_cfg
        self.device = torch.device(device)
        self.tasks: list[OrcaLocomotionTask] = []
        self._last_logs: list[dict[str, float]] = []

        scene_cfg = dict(task_cfg.get("scene_binding", {}))
        resolver = _load_scene_binding_resolver(scene_cfg)
        addresses = task_cfg.get("orcagym_addresses") or ["localhost:50051"]
        for env_index, address in enumerate(addresses):
            binding = _call_scene_binding_resolver(
                resolver,
                scene_cfg=scene_cfg,
                orcagym_addr=address,
                time_step=float(task_cfg["sim"]["time_step"]),
            )
            task = OrcaLocomotionTask(
                cfg=task_cfg,
                orcagym_addr=address,
                agent_name=binding.agent_name,
                robot_config=binding.robot_config,
                render_mode=render_mode or task_cfg.get("sim", {}).get("render_mode", "none"),
                env_id=f"{task_cfg.get('name', 'locomotion')}-OrcaGym-{env_index:03d}",
            )
            self.tasks.append(task)

        if not self.tasks:
            raise ValueError("At least one OrcaGym address is required for RSL-RL training.")

        self.num_envs = len(self.tasks)
        self.num_actions = self.tasks[0].num_actions
        self.max_episode_length = self.tasks[0].max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def get_observations(self) -> TensorDict:
        observations = [task.get_observations(noisy=True) for task in self.tasks]
        return self._stack_observations(observations)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().to("cpu").numpy()
        if actions_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.num_actions)}, got {actions_np.shape}")

        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        logs = []

        for env_index, (task, action) in enumerate(zip(self.tasks, actions_np)):
            result = task.step(action)
            observations.append(result.observations)
            rewards[env_index] = result.reward
            dones[env_index] = result.done
            time_outs[env_index] = result.time_out
            logs.append(result.log)
            self.episode_length_buf[env_index] = task.episode_length

        self._last_logs = logs
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": self._aggregate_logs(logs),
        }
        return (
            self._stack_observations(observations),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def reset(self) -> TensorDict:
        observations = []
        for env_index, task in enumerate(self.tasks):
            obs, _ = task.reset_model()
            observations.append(obs)
            self.episode_length_buf[env_index] = task.episode_length
        return self._stack_observations(observations)

    def close(self) -> None:
        for task in self.tasks:
            task.close()

    def _stack_observations(self, observations: list[dict[str, np.ndarray]]) -> TensorDict:
        stacked = {
            key: torch.as_tensor(
                np.stack([obs[key] for obs in observations], axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            for key in observations[0].keys()
        }
        return TensorDict(stacked, batch_size=[self.num_envs], device=self.device)

    def _aggregate_logs(self, logs: list[dict[str, float]]) -> dict[str, torch.Tensor]:
        if not logs:
            return {}
        keys = sorted({key for log in logs for key in log.keys()})
        return {
            key: torch.as_tensor([log.get(key, 0.0) for log in logs], dtype=torch.float32, device=self.device)
            for key in keys
        }


def _load_scene_binding_resolver(scene_cfg: dict[str, Any]) -> Callable[..., Any]:
    resolver_path = scene_cfg.get("resolver")
    if not resolver_path:
        raise ValueError(
            "Task config must define `scene_binding.resolver`, e.g. "
            "`g1`, `go2`, or an import path."
        )
    return resolve_scene_binding(resolver_path)


def _call_scene_binding_resolver(
    resolver: Callable[..., Any],
    *,
    scene_cfg: dict[str, Any],
    orcagym_addr: str,
    time_step: float,
) -> Any:
    kwargs = {
        "orcagym_addr": orcagym_addr,
        "time_step": time_step,
        **{key: value for key, value in scene_cfg.items() if key != "resolver"},
    }
    signature = inspect.signature(resolver)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return resolver(**kwargs)
    accepted_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return resolver(**accepted_kwargs)
