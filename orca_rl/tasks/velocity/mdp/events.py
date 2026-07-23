from __future__ import annotations


def reset_root_state_uniform():
    """Reset root pose with bounded noise."""


def reset_joints_by_offset():
    """Reset controlled joints around the nominal pose."""


def randomize_friction():
    """Sample friction randomization state exposed to privileged observations."""


def randomize_body_mass():
    """Sample and apply reset-time base mass, inertia, and COM randomization."""


def randomize_actuator_properties():
    """Sample PD gain and motor-strength randomization."""


def randomize_action_latency():
    """Sample a bounded action delay in control steps."""


def randomize_solver_params():
    """Sample local MuJoCo solver parameter randomization."""


def randomize_contact_params():
    """Sample local MuJoCo contact parameter randomization."""


def push_robot():
    """Apply periodic base-velocity perturbations during the episode."""
