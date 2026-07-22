from pathlib import Path

import pytest
import torch


@pytest.mark.orca_gpu
def test_real_orca_runtime_nworld_and_local_reset():
    if not torch.cuda.is_available():
        pytest.skip("Orca GPU runtime requires CUDA")
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    from orcalab_rslrl._internal.runtime import OrcaPhysicsRuntime

    orca = OrcaPhysicsRuntime(
        Path(__file__).parent / "assets/tiny.xml", num_envs=8, device="cuda:0"
    )
    assert orca.state.qpos.shape == (8, 1)
    assert orca.state.ctrl.shape == (8, 1)
    orca.write_action(torch.ones_like(orca.state.ctrl))
    orca.step(2)
    before = orca.state.qpos.clone()
    orca.reset(torch.tensor([1, 5], device=before.device))
    assert orca.state.qpos[[1, 5]].eq(0).all()
    assert torch.equal(orca.state.qpos[[0, 2, 3, 4, 6, 7]], before[[0, 2, 3, 4, 6, 7]])
