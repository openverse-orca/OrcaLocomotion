from orca_rl.tasks.velocity.config.go2.rl_cfg import _unitree_go2_ppo_runner_cfg


def deeprobotics_lite3_flat_ppo_runner_cfg():
    cfg = _unitree_go2_ppo_runner_cfg(run_name="flat", max_iterations=30_000)
    cfg.experiment_name = "deeprobotics_lite3_flat"
    return cfg


RL_CONFIG_FACTORY = deeprobotics_lite3_flat_ppo_runner_cfg
