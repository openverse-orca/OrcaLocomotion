import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from ..envs import ManagerBasedRLEnv


class RslRlVecEnvAdapter(VecEnv):
    """Sole RSL-RL boundary; the Orca runtime never imports the trainer."""
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.cfg = env.cfg
        self.num_envs, self.device = env.num_envs, env.device
        self.num_actions = env.orca.state.ctrl.shape[-1]
        self.max_episode_length = env.cfg.episode_length_steps
        self.episode_length_buf = env.episode_length_buf

    def _td(self, obs):
        return TensorDict(obs, batch_size=[self.num_envs], device=self.device)

    def get_observations(self): return self._td(self.env.observations())
    def reset(self): return self._td(self.env.reset())

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._td(obs), reward, terminated | truncated, info

    def close(self):
        close = getattr(self.env.orca, "close", None)
        if close: close()
