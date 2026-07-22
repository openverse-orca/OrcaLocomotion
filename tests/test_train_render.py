from types import SimpleNamespace

import pytest
import torch

from orcalab_rslrl.tools.train import _build_parser
from orcalab_rslrl.tools.train_render import OrcaLabTrainingRenderHook


class _Renderer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def render(self, qpos, sim_time):
        self.frames.append((qpos.copy(), sim_time))

    def close(self):
        self.closed = True


class _Env:
    def __init__(self):
        self.num_envs = 4
        self.env = SimpleNamespace(
            step_dt=0.02,
            orca=SimpleNamespace(
                state=SimpleNamespace(qpos=torch.arange(32, dtype=torch.float32).reshape(4, 8))
            ),
        )
        self.steps = 0

    def step(self, actions):
        self.steps += 1
        return actions


def test_train_cli_accepts_orcalab_render_options():
    args = _build_parser().parse_args(
        ["--orcalab", "--render-num-envs", "3", "--render-fps", "12", "--no-publish"]
    )
    assert args.orcalab is True
    assert args.render_num_envs == 3
    assert args.render_fps == 12
    assert args.no_publish is True


def test_training_render_hook_throttles_slices_and_restores_step():
    env = _Env()
    renderer = _Renderer()
    times = iter((1.0, 1.01, 1.11))
    original_step = env.step
    hook = OrcaLabTrainingRenderHook(
        env,
        renderer,
        num_envs=2,
        render_fps=10,
        clock=lambda: next(times),
    )
    hook.install()

    assert env.step("one") == "one"
    assert env.step("two") == "two"
    assert env.step("three") == "three"
    assert len(renderer.frames) == 2
    assert renderer.frames[0][0].shape == (2, 8)
    assert renderer.frames[0][1] == pytest.approx(0.02)
    assert renderer.frames[1][1] == pytest.approx(0.06)

    hook.close()
    assert env.step == original_step
    assert renderer.closed is True


def test_training_render_hook_rejects_invalid_limits():
    env = _Env()
    renderer = _Renderer()
    with pytest.raises(ValueError, match="num_envs"):
        OrcaLabTrainingRenderHook(env, renderer, num_envs=5, render_fps=30)
    with pytest.raises(ValueError, match="render_fps"):
        OrcaLabTrainingRenderHook(env, renderer, num_envs=1, render_fps=0)
