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
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_velocity",
        run_name=run_name,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _unitree_g1_ppo_runner_cfg(run_name="flat", max_iterations=30_000)


def unitree_g1_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return _unitree_g1_ppo_runner_cfg(run_name="rough", max_iterations=30_000)


RL_CONFIG_FACTORY = unitree_g1_ppo_runner_cfg
