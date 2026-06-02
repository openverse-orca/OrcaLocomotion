from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool = True
    friction_range: tuple[float, float] = (0.7, 1.3)
    base_mass_delta_range: tuple[float, float] = (-0.5, 1.5)
    base_inertia_scale_range: tuple[float, float] = (0.9, 1.1)
    base_com_offset_range: tuple[float, float] = (-0.02, 0.02)
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    torque_scale_range: tuple[float, float] = (0.9, 1.1)
    max_action_delay_steps: int = 2
    push_interval_s: float = 8.0
    push_velocity_range: tuple[float, float] = (-0.4, 0.4)
    push_yaw_velocity_range: tuple[float, float] = (-0.3, 0.3)
    solver_iterations_range: tuple[int, int] = (40, 80)
    solver_tolerance_scale_range: tuple[float, float] = (0.5, 2.0)
    contact_solref_timeconst_scale_range: tuple[float, float] = (0.8, 1.2)
    contact_solref_dampratio_scale_range: tuple[float, float] = (0.8, 1.2)
    contact_solimp_scale_range: tuple[float, float] = (0.9, 1.1)
    contact_margin_scale_range: tuple[float, float] = (0.8, 1.2)
    friction_geom_names: tuple[str, ...] = ()
    friction_geom_patterns: tuple[str, ...] = ("floor", "ground", "terrain", "plane", "hfield")
    base_mass_body_name: str | None = None
    terrain: str = "plane"
    terrain_curriculum: bool = False
    height_scan_dim: int = 0


@dataclass(frozen=True)
class RandomizationState:
    friction_scale: float = 1.0
    base_mass_delta: float = 0.0
    base_inertia_scale: float = 1.0
    base_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    torque_scale: float = 1.0
    action_delay_steps: int = 0
    solver_iterations: int | None = None
    solver_tolerance_scale: float = 1.0
    contact_solref_timeconst_scale: float = 1.0
    contact_solref_dampratio_scale: float = 1.0
    contact_solimp_scale: float = 1.0
    contact_margin_scale: float = 1.0


class DomainRandomizer:
    def __init__(self, cfg: RandomizationConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> RandomizationState:
        if not self.cfg.enabled:
            return RandomizationState()
        friction_scale = float(self.rng.uniform(*self.cfg.friction_range))
        base_mass_delta = float(self.rng.uniform(*self.cfg.base_mass_delta_range))
        base_inertia_scale = float(self.rng.uniform(*self.cfg.base_inertia_scale_range))
        com_range = self.cfg.base_com_offset_range
        base_com_offset = tuple(float(value) for value in self.rng.uniform(com_range[0], com_range[1], size=3))
        max_delay = max(0, int(self.cfg.max_action_delay_steps))
        solver_low, solver_high = self.cfg.solver_iterations_range
        return RandomizationState(
            friction_scale=friction_scale,
            base_mass_delta=base_mass_delta,
            base_inertia_scale=base_inertia_scale,
            base_com_offset=base_com_offset,
            kp_scale=float(self.rng.uniform(*self.cfg.kp_scale_range)),
            kd_scale=float(self.rng.uniform(*self.cfg.kd_scale_range)),
            torque_scale=float(self.rng.uniform(*self.cfg.torque_scale_range)),
            action_delay_steps=int(self.rng.integers(0, max_delay + 1)) if max_delay > 0 else 0,
            solver_iterations=int(self.rng.integers(int(solver_low), int(solver_high) + 1)),
            solver_tolerance_scale=float(self.rng.uniform(*self.cfg.solver_tolerance_scale_range)),
            contact_solref_timeconst_scale=float(
                self.rng.uniform(*self.cfg.contact_solref_timeconst_scale_range)
            ),
            contact_solref_dampratio_scale=float(
                self.rng.uniform(*self.cfg.contact_solref_dampratio_scale_range)
            ),
            contact_solimp_scale=float(self.rng.uniform(*self.cfg.contact_solimp_scale_range)),
            contact_margin_scale=float(self.rng.uniform(*self.cfg.contact_margin_scale_range)),
        )
