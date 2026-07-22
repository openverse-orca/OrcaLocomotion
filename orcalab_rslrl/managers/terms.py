from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch

if TYPE_CHECKING:
    from ..envs.manager_based_rl_env import ManagerBasedRLEnv

Term = Callable[["ManagerBasedRLEnv"], torch.Tensor]
ResetTerm = Callable[["ManagerBasedRLEnv", torch.Tensor], None]
CommandResetTerm = Callable[["ManagerBasedRLEnv", torch.Tensor], torch.Tensor]
CommandUpdateTerm = Callable[["ManagerBasedRLEnv", str], None]
# Reset-mode events receive long env ids; interval-mode events receive a bool mask;
# startup/post_step events receive None.
EventTerm = Callable[["ManagerBasedRLEnv", "torch.Tensor | None"], None]


@dataclass(frozen=True)
class ObservationTermCfg:
    """One observation term with optional uniform noise and scaling."""

    func: Term
    noise: tuple[float, float] | None = None
    scale: float | None = None


@dataclass(frozen=True)
class ObservationGroupCfg:
    terms: dict[str, "Term | ObservationTermCfg"]
    concatenate_terms: bool = True
    enable_corruption: bool = False


@dataclass(frozen=True)
class RewardTermCfg:
    func: Term
    weight: float = 1.0


@dataclass(frozen=True)
class TerminationTermCfg:
    func: Term
    time_out: bool = False


@dataclass(frozen=True)
class CommandTermCfg:
    """Generates a batched command tensor on reset/resample.

    ``func(env, env_ids)`` samples fresh commands for the given envs.
    ``update(env, name)`` is invoked once per control step (e.g. heading control).
    ``resampling_time_range`` enables per-env time based resampling.
    """

    func: CommandResetTerm
    dim: int
    resample_on_done_mask: bool = True
    resampling_time_range: tuple[float, float] | None = None
    update: CommandUpdateTerm | None = None


@dataclass(frozen=True)
class EventTermCfg:
    """Events mirror Orca modes.

    - ``startup``: run once when the env is constructed (func(env, None)).
    - ``reset``: run on every (partial) reset with the env ids being reset.
    - ``interval``: run when the per-env timer expires (func(env, bool_mask)).
    - ``post_step``: run after every physics step (func(env, None)), used for
      stateful trackers such as feet air time.
    """

    func: EventTerm
    mode: Literal["startup", "reset", "interval", "post_step"] = "reset"
    interval_range_s: tuple[float, float] | None = None
