"""Built-in velocity task configuration factories."""

from __future__ import annotations

from .g1 import (
    unitree_g1_flat_env_cfg,
    unitree_g1_ppo_runner_cfg,
    unitree_g1_rough_env_cfg,
    unitree_g1_rough_ppo_runner_cfg,
)
from .go2 import (
    unitree_go2_flat_env_cfg,
    unitree_go2_ppo_runner_cfg,
    unitree_go2_rough_env_cfg,
    unitree_go2_rough_ppo_runner_cfg,
)

__all__ = [
    "unitree_g1_flat_env_cfg",
    "unitree_g1_ppo_runner_cfg",
    "unitree_g1_rough_env_cfg",
    "unitree_g1_rough_ppo_runner_cfg",
    "unitree_go2_flat_env_cfg",
    "unitree_go2_ppo_runner_cfg",
    "unitree_go2_rough_env_cfg",
    "unitree_go2_rough_ppo_runner_cfg",
]
