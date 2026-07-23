import torch

from orcalab_rslrl.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from orcalab_rslrl.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from orcalab_rslrl.orca import OrcaCapabilities, OrcaPhysics, OrcaState


class FakeBackend(OrcaPhysics):
    capabilities = OrcaCapabilities(local_reset=True)

    def __init__(self, n=8, nu=3):
        self.num_envs, self.device = n, torch.device("cpu")
        self._state = OrcaState(
            qpos=torch.zeros(n, nu), qvel=torch.zeros(n, nu), qacc=torch.zeros(n, nu),
            ctrl=torch.zeros(n, nu), actuator_force=torch.zeros(n, nu),
            sensordata=torch.zeros(n, 0), time=torch.zeros(n),
        )

    @property
    def state(self): return self._state
    def write_action(self, action): self.state.ctrl.copy_(action)
    def step(self, nstep=1):
        self.state.qvel.add_(self.state.ctrl * nstep)
        self.state.qpos.add_(self.state.qvel * nstep)
        self.state.time.add_(nstep)
    def forward(self): pass
    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        self.state.qpos[ids] = self.state.qvel[ids] = self.state.ctrl[ids] = 0
        self.state.time[ids] = 0


def make_env(n=8):
    cfg = ManagerBasedRLEnvCfg(
        decimation=2,
        episode_length_steps=3,
        observations={"obs": ObservationGroupCfg({"qpos": lambda e: e.orca.state.qpos})},
        rewards={"alive": RewardTermCfg(lambda e: torch.ones(e.num_envs, device=e.device))},
        terminations={"too_far": TerminationTermCfg(lambda e: e.orca.state.qpos[:, 0] > 100)},
    )
    return ManagerBasedRLEnv(FakeBackend(n), cfg)


def test_batch_shapes_and_step_lifecycle():
    env = make_env(16)
    obs, reward, terminated, truncated, info = env.step(torch.ones(16, 3))
    assert obs["obs"].shape == (16, 3)
    assert reward.shape == terminated.shape == truncated.shape == (16,)
    assert env.orca.state.time.eq(2).all()
    assert "final_observation" in info and info["log"] == {}
    # Episode metrics appear (Orca-style) once episodes finish.
    for _ in range(2):
        _, _, _, truncated, info = env.step(torch.ones(16, 3))
    assert truncated.all()
    assert "Episode_Reward/alive" in info["log"]
    assert "Episode_Termination/too_far" in info["log"]


def test_timeout_performs_local_reset():
    env = make_env(4)
    env.episode_length_buf[:] = torch.tensor([2, 0, 0, 0])
    _, _, _, truncated, _ = env.step(torch.ones(4, 3))
    assert truncated.tolist() == [True, False, False, False]
    assert env.episode_length_buf.tolist() == [0, 1, 1, 1]
    assert env.orca.state.time.tolist() == [0, 2, 2, 2]


def test_action_shape_is_enforced():
    env = make_env(4)
    try:
        env.step(torch.zeros(3, 3))
    except ValueError as exc:
        assert "actions" in str(exc)
    else:
        raise AssertionError("shape mismatch must fail")


def test_shared_observation_term_is_computed_once_per_observation_call():
    calls = 0

    def shared_term(env):
        nonlocal calls
        calls += 1
        return env.orca.state.qpos

    shared = ObservationTermCfg(shared_term)
    cfg = ManagerBasedRLEnvCfg(
        observations={
            "actor": ObservationGroupCfg({"qpos": shared}),
            "critic": ObservationGroupCfg({"qpos": shared}),
        },
        rewards={"alive": RewardTermCfg(lambda e: torch.ones(e.num_envs, device=e.device))},
    )
    env = ManagerBasedRLEnv(FakeBackend(4), cfg)
    calls = 0

    observations = env.observations()

    assert calls == 1
    assert torch.equal(observations["actor"], observations["critic"])


def test_distinct_observation_terms_are_not_cached_together():
    calls = {"actor": 0, "critic": 0}

    def actor_term(env):
        calls["actor"] += 1
        return env.orca.state.qpos

    def critic_term(env):
        calls["critic"] += 1
        return env.orca.state.qvel

    cfg = ManagerBasedRLEnvCfg(
        observations={
            "actor": ObservationGroupCfg({"qpos": ObservationTermCfg(actor_term)}),
            "critic": ObservationGroupCfg({"qvel": ObservationTermCfg(critic_term)}),
        },
        rewards={"alive": RewardTermCfg(lambda e: torch.ones(e.num_envs, device=e.device))},
    )
    env = ManagerBasedRLEnv(FakeBackend(4), cfg)
    calls = {"actor": 0, "critic": 0}

    env.observations()

    assert calls == {"actor": 1, "critic": 1}


def test_shared_observation_term_applies_group_specific_noise_after_caching():
    shared = ObservationTermCfg(lambda env: torch.zeros(env.num_envs, 1, device=env.device), noise=(1.0, 1.0))
    cfg = ManagerBasedRLEnvCfg(
        observations={
            "actor": ObservationGroupCfg({"value": shared}, enable_corruption=True),
            "critic": ObservationGroupCfg({"value": shared}, enable_corruption=False),
        },
        rewards={"alive": RewardTermCfg(lambda e: torch.ones(e.num_envs, device=e.device))},
    )
    env = ManagerBasedRLEnv(FakeBackend(4), cfg)

    observations = env.observations()

    assert torch.equal(observations["actor"], torch.ones(4, 1))
    assert torch.equal(observations["critic"], torch.zeros(4, 1))
