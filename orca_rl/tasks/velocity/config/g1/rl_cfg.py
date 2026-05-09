from __future__ import annotations

from orca_rl.tasks.velocity.config_types import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def _unitree_g1_ppo_runner_cfg(run_name: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
    """Create RSL-RL PPO runner configuration for Unitree G1 velocity."""

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
        run_name=run_name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _unitree_g1_ppo_runner_cfg(run_name="g1_flat_velocity", max_iterations=1500)


def unitree_g1_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _unitree_g1_ppo_runner_cfg(run_name="g1_rough_velocity", max_iterations=2500)


RL_CONFIG_FACTORY = unitree_g1_ppo_runner_cfg
