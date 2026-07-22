from dataclasses import dataclass, field

from ..managers.terms import (
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ResetTerm,
    RewardTermCfg,
    TerminationTermCfg,
)


@dataclass
class ManagerBasedRLEnvCfg:
    decimation: int = 4
    episode_length_steps: int = 1000
    # Scalar or per-actuator action scale (matched against the ctrl vector).
    action_scale: "float | tuple[float, ...]" = 1.0
    clip_actions: float | None = None
    # When True the ctrl target is default actuated joint pos + scale * action
    # (Orca JointPositionAction with use_default_offset=True).
    action_offset: bool = False
    # Multiply reward terms by the control step dt (Orca RewardManager default).
    reward_scale_by_dt: bool = False
    gpu_done_reset: bool = False
    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands: dict[str, CommandTermCfg] = field(default_factory=dict)
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    reset_events: list[ResetTerm] = field(default_factory=list)
