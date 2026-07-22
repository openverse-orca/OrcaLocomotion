import pytest

import orcalab_rslrl
from orcalab_rslrl.orca import OrcaRuntimeConfig


def test_root_api_only_exposes_orca_surface():
    assert set(orcalab_rslrl.__all__) == {
        "OrcaRuntimeConfig",
        "list_tasks",
        "make_env",
        "register_task",
    }
    assert not any("warp" in name.lower() or "mujoco" in name.lower() for name in orcalab_rslrl.__all__)


def test_runtime_config_validates_product_inputs():
    cfg = OrcaRuntimeConfig(num_envs=32, device="cuda:1", physics_timestep=0.002)
    assert cfg.task_kwargs()["num_envs"] == 32
    assert cfg.task_kwargs()["physics_timestep"] == 0.002
    with pytest.raises(ValueError, match="num_envs"):
        OrcaRuntimeConfig(num_envs=0)
    with pytest.raises(ValueError, match="physics_timestep"):
        OrcaRuntimeConfig(physics_timestep=0.0)
