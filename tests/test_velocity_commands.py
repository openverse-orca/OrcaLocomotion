from types import SimpleNamespace

import torch

from orcalab_rslrl.mdp.commands import UniformVelocityCommand, VelocityRanges


def test_masked_velocity_sampling_keeps_unexpired_command_state():
    env = SimpleNamespace(num_envs=4, device=torch.device("cpu"))
    command = UniformVelocityCommand(
        VelocityRanges((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), heading=(-3.14, 3.14)),
        rel_standing_envs=0.5,
        heading_command=True,
    )
    initial_ids = torch.arange(env.num_envs)
    command.sample(env, initial_ids)
    heading_before = command.heading_target.clone()
    heading_mask_before = command.is_heading_env.clone()
    standing_before = command.is_standing_env.clone()

    mask = torch.tensor([False, True, False, True])
    sampled = command.sample_masked(env, mask)

    assert sampled.shape == (env.num_envs, 3)
    assert torch.equal(command.heading_target[~mask], heading_before[~mask])
    assert torch.equal(command.is_heading_env[~mask], heading_mask_before[~mask])
    assert torch.equal(command.is_standing_env[~mask], standing_before[~mask])
