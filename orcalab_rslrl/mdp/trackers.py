from __future__ import annotations

import torch


class FeetContactTracker:
    """Per-foot contact / air-time bookkeeping from `found` contact sensors.

    Registered as a ``post_step`` event; mirrors Orca's ContactSensor
    ``track_air_time`` state (current_air_time / current_contact_time /
    first_contact) at the control-step rate.
    """

    def __init__(self, found_sensors: tuple[str, ...]):
        self.found_sensors = found_sensors
        self.in_contact: torch.Tensor | None = None
        self.air_time: torch.Tensor | None = None
        self.contact_time: torch.Tensor | None = None
        self.first_contact: torch.Tensor | None = None

    def _ensure_state(self, env) -> None:
        if self.in_contact is None:
            shape = (env.num_envs, len(self.found_sensors))
            self.in_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
            self.air_time = torch.zeros(shape, device=env.device)
            self.contact_time = torch.zeros(shape, device=env.device)
            self.first_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)

    def found(self, env) -> torch.Tensor:
        return torch.cat([env.orca.sensor(name) for name in self.found_sensors], dim=-1)

    def force(self, env, force_sensors: tuple[str, ...]) -> torch.Tensor:
        return torch.stack(
            [env.orca.sensor(name) for name in force_sensors], dim=1
        )  # [B, F, 3]

    def __call__(self, env, _arg=None) -> None:
        """post_step update with dt = env.step_dt."""
        self._ensure_state(env)
        contact_now = self.found(env) > 0
        self.first_contact = contact_now & ~self.in_contact
        dt = env.step_dt
        self.air_time = torch.where(contact_now, torch.zeros_like(self.air_time), self.air_time + dt)
        self.contact_time = torch.where(contact_now, self.contact_time + dt, torch.zeros_like(self.contact_time))
        self.in_contact = contact_now

    def reset(self, env, env_ids: torch.Tensor) -> None:
        self._ensure_state(env)
        self.in_contact[env_ids] = False
        self.air_time[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.first_contact[env_ids] = False
