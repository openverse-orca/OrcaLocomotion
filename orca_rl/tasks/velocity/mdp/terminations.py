from __future__ import annotations


def time_out():
    """Episode time limit."""


def bad_orientation():
    """Terminate when the base tilt exceeds the configured limit."""


def base_height():
    """Terminate when base height leaves the valid range."""


def base_contact():
    """Terminate on configured non-foot body contact."""


def illegal_contact():
    """Terminate on non-foot contact above a force/contact threshold."""
