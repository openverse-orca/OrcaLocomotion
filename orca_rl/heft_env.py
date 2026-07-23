from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from .rsl_env.batched_locomotion_task import BatchedOrcaLocomotionTask
from .rsl_env.rendering import resolve_rendering
from .rsl_env.scene_resolvers import resolve_scene_binding


class HeftOrcaEnv:
    """Minimal OrcaLab environment surface needed by the HEFT player."""

    def __init__(self, task_cfg: dict[str, Any]) -> None:
        self.cfg = task_cfg
        self.device = "cpu"
        self.tasks: list[BatchedOrcaLocomotionTask] = []
        self.headless, self.render_mode = resolve_rendering(task_cfg, render_mode="human", headless=False)
        self.cfg.setdefault("sim", {})["headless"] = self.headless
        self.cfg["sim"]["render_mode"] = self.render_mode

        scene_cfg = dict(task_cfg.get("scene_binding", {}))
        resolver = resolve_scene_binding(scene_cfg.pop("resolver", "g1"))
        addresses = task_cfg.get("orcagym_addresses") or ["localhost:50051"]
        if len(addresses) != 1:
            raise ValueError("HEFT G1 + Dex3 playback supports exactly one OrcaGym address.")
        signature = inspect.signature(resolver)
        kwargs = {
            "orcagym_addr": addresses[0],
            "time_step": float(task_cfg["sim"]["time_step"]),
            "num_envs": 1,
            **scene_cfg,
        }
        binding = resolver(**{key: value for key, value in kwargs.items() if key in signature.parameters})
        task = BatchedOrcaLocomotionTask(
            cfg=task_cfg,
            orcagym_addr=addresses[0],
            agent_names=binding.agent_names,
            robot_config=binding.robot_config,
            render_mode="human",
            headless=False,
            env_id="heft-g1-dex3-OrcaGym-000",
        )
        self.tasks.append(task)
        self.num_envs = 1
        self.num_actions = task.num_actions
        self.episode_length_buf = np.zeros(1, dtype=np.int64)

    def reset(self) -> None:
        self.tasks[0].reset_model()
        self.episode_length_buf.fill(0)

    def close(self) -> None:
        for task in self.tasks:
            task.close()


def make_heft_env(task_cfg: dict[str, Any]) -> HeftOrcaEnv:
    return HeftOrcaEnv(task_cfg)
