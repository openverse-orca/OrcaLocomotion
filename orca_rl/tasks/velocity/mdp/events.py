from __future__ import annotations


def reset_root_state_uniform():
    """Reset root pose with bounded noise."""


def reset_joints_by_offset():
    """Reset controlled joints around the nominal pose."""


def randomize_friction():
    """Sample friction randomization state exposed to privileged observations."""


def randomize_body_mass():
    """Sample base-mass randomization state exposed to privileged observations."""


def randomize_terrain():
    """Select or reset a rough-terrain tile."""
