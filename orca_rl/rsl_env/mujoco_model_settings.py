from __future__ import annotations

from typing import Any

import mujoco


def apply_unitree_play_global_settings(model: Any) -> None:
    """Apply the MuJoCo global settings used by unitree-orca play."""
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    model.opt.density = 0.0
    model.opt.viscosity = 0.0
    model.opt.wind[:] = 0.0
    model.opt.noslip_iterations = 0
    model.opt.sdf_iterations = 10
