from __future__ import annotations

from orca_rl.tasks.velocity.config_types import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RSL-RL PPO runner configuration for Unitree GO2 velocity."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "BetaDistribution",
                "action_range": [-1.0, 1.0],
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3.0e-4),
        run_name="go2_flat_velocity",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=1500,
    )


RL_CONFIG_FACTORY = unitree_go2_ppo_runner_cfg

