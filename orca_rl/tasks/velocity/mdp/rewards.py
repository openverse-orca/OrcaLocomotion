from __future__ import annotations


def track_linear_velocity():
    """Exponential reward for tracking commanded planar linear velocity."""


def track_angular_velocity():
    """Exponential reward for tracking commanded yaw velocity."""


def z_velocity_l2():
    """Penalty on vertical base velocity."""


def orientation_l2():
    """Penalty on non-upright projected gravity components."""


def base_height_l2():
    """Penalty on base-height error."""


def torques_l2():
    """Torque energy penalty."""


def action_rate_l2():
    """Penalty on action changes."""


def joint_pos_limits():
    """Penalty for joint-limit violation."""


def feet_slip():
    """Penalty for horizontal foot motion while in contact."""


def termination():
    """Terminal penalty."""


def mean_action_acc():
    """Mean action acceleration diagnostic metric."""
