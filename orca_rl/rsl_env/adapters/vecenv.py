from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from ..batched_locomotion_task import BatchedOrcaLocomotionTask
from ..rendering import resolve_rendering
from ..scene_resolvers import resolve_scene_binding


class OrcaRslRlVecEnv(VecEnv):
    """Config-driven RSL-RL VecEnv adapter for Orca locomotion tasks."""

    def __init__(
        self,
        task_cfg: dict[str, Any],
        *,
        device: str = "cpu",
        render_mode: str | None = None,
        headless: bool | None = None,
    ) -> None:
        self.cfg = task_cfg
        self.device = torch.device(device)
        self.tasks: list[BatchedOrcaLocomotionTask] = []
        self._last_logs: list[dict[str, float]] = []
        self.headless, self.render_mode = resolve_rendering(
            task_cfg,
            render_mode=render_mode,
            headless=headless,
        )
        self.cfg.setdefault("sim", {})["headless"] = self.headless
        self.cfg["sim"]["render_mode"] = self.render_mode

        scene_cfg = dict(task_cfg.get("scene_binding", {}))
        scene_cfg.setdefault("terrain_cfg", task_cfg.get("terrain"))
        scene_cfg.setdefault("terrain_seed", int(task_cfg.get("seed", 1)))
        resolver = _load_scene_binding_resolver(scene_cfg)
        addresses = task_cfg.get("orcagym_addresses") or ["localhost:50051"]
        desired_num_envs = int(task_cfg.get("num_envs") or len(addresses))
        if desired_num_envs <= 0:
            raise ValueError("Task config `num_envs` must be a positive integer.")
        env_counts = _split_num_envs(desired_num_envs, len(addresses))
        env_index = 0
        for address, env_count in zip(addresses, env_counts):
            if env_count <= 0:
                continue
            binding = _call_scene_binding_resolver(
                resolver,
                scene_cfg=scene_cfg,
                orcagym_addr=address,
                time_step=float(task_cfg["sim"]["time_step"]),
                num_envs=env_count,
            )
            task = BatchedOrcaLocomotionTask(
                cfg=task_cfg,
                orcagym_addr=address,
                agent_names=binding.agent_names,
                robot_config=binding.robot_config,
                model_xml_path=getattr(binding, "model_xml_path", None),
                render_mode=self.render_mode,
                headless=self.headless,
                env_id=f"{task_cfg.get('name', 'locomotion')}-OrcaGym-{env_index:03d}",
            )
            self.tasks.append(task)
            env_index += task.num_envs

        if not self.tasks:
            raise ValueError("At least one OrcaGym address is required for RSL-RL training.")
        if env_index != desired_num_envs:
            raise ValueError(f"Expected {desired_num_envs} envs, resolved {env_index} envs from OrcaLab scene.")

        self.num_envs = env_index
        self.num_actions = self.tasks[0].num_actions
        self.max_episode_length = self.tasks[0].max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def get_observations(self) -> TensorDict:
        observations = [task.get_observations(noisy=True) for task in self.tasks]
        return self._concat_observations(observations)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().to("cpu").numpy()
        if actions_np.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.num_actions)}, got {actions_np.shape}")

        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        logs = []

        start = 0
        for task in self.tasks:
            stop = start + task.num_envs
            result = task.step(actions_np[start:stop])
            observations.append(result.observations)
            rewards[start:stop] = result.rewards
            dones[start:stop] = result.dones
            time_outs[start:stop] = result.time_outs
            logs.extend(result.logs)
            self.episode_length_buf[start:stop] = torch.as_tensor(
                result.episode_lengths,
                dtype=torch.long,
                device=self.device,
            )
            start = stop

        self._last_logs = logs
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": self._aggregate_logs(logs),
        }
        return (
            self._concat_observations(observations),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def reset(self) -> TensorDict:
        observations = []
        start = 0
        for task in self.tasks:
            obs, _ = task.reset_model()
            observations.append(obs)
            stop = start + task.num_envs
            self.episode_length_buf[start:stop] = torch.as_tensor(
                task.episode_lengths,
                dtype=torch.long,
                device=self.device,
            )
            start = stop
        return self._concat_observations(observations)

    def close(self) -> None:
        for task in self.tasks:
            task.close()

    def _concat_observations(self, observations: list[dict[str, np.ndarray]]) -> TensorDict:
        concatenated = {
            key: torch.as_tensor(
                np.concatenate([obs[key] for obs in observations], axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            for key in observations[0].keys()
        }
        return TensorDict(concatenated, batch_size=[self.num_envs], device=self.device)

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
    num_envs: int,
) -> Any:
    kwargs = {
        "orcagym_addr": orcagym_addr,
        "time_step": time_step,
        "num_envs": num_envs,
        **{key: value for key, value in scene_cfg.items() if key != "resolver"},
    }
    signature = inspect.signature(resolver)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return resolver(**kwargs)
    accepted_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return resolver(**accepted_kwargs)


def _split_num_envs(num_envs: int, num_addresses: int) -> list[int]:
    if num_addresses <= 0:
        raise ValueError("At least one OrcaGym address is required.")
    base = num_envs // num_addresses
    remainder = num_envs % num_addresses
    return [base + (1 if index < remainder else 0) for index in range(num_addresses)]
