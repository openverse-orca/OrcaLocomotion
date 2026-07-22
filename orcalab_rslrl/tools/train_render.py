from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class OrcaLabTrainingRenderHook:
    """Render a small view of a training batch without changing the rollout."""

    def __init__(
        self,
        env,
        renderer,
        *,
        num_envs: int,
        render_fps: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if num_envs < 1 or num_envs > env.num_envs:
            raise ValueError("render num_envs must be between 1 and the training num_envs")
        if render_fps <= 0:
            raise ValueError("render_fps must be positive")
        self.env = env
        self.renderer = renderer
        self.num_envs = int(num_envs)
        self.render_interval = 1.0 / float(render_fps)
        self.clock = clock
        self.sim_time = 0.0
        self.last_render: float | None = None
        self._original_step = env.step
        self._installed = False

    def install(self) -> None:
        if not self._installed:
            self.env.step = self.step
            self._installed = True

    def step(self, actions) -> Any:
        result = self._original_step(actions)
        self.sim_time += float(self.env.env.step_dt)
        now = self.clock()
        if self.last_render is None or now - self.last_render >= self.render_interval:
            qpos = self.env.env.orca.state.qpos[: self.num_envs].detach().cpu().numpy()
            self.renderer.render(qpos, self.sim_time)
            self.last_render = now
        return result

    def close(self) -> None:
        if self._installed:
            self.env.step = self._original_step
            self._installed = False
        self.renderer.close()


def attach_orcalab_training_renderer(
    env,
    *,
    orca_addr: str,
    asset_path: str,
    render_num_envs: int,
    render_fps: float,
    agent_prefix: str,
    spacing: float,
    spawn_range: float | None,
    root_xy_scale: float,
    publish: bool,
) -> OrcaLabTrainingRenderHook:
    from ..orcalab_batch_render import OrcaLabBatchRenderer

    inner = env.env
    visible_envs = min(int(render_num_envs), env.num_envs)
    renderer = OrcaLabBatchRenderer(
        orcagym_addr=orca_addr,
        num_envs=visible_envs,
        joint_qpos_addr=inner.orca.joint_qpos_addresses(),
        agent_prefix=agent_prefix,
        asset_path=asset_path,
        spacing=spacing,
        spawn_range=spawn_range,
        root_xy_scale=root_xy_scale,
        scene_timestep=inner.physics_dt,
        scene_profile="orca-train",
        publish=publish,
    )
    hook = OrcaLabTrainingRenderHook(
        env,
        renderer,
        num_envs=visible_envs,
        render_fps=render_fps,
    )
    hook.install()
    print(
        f"[orca-train] rendering {visible_envs}/{env.num_envs} environments "
        f"in OrcaLab at up to {render_fps:g} FPS"
    )
    return hook
