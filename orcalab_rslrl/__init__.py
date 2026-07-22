"""OrcaLab-RSLRL.

The supported API lives in :mod:`orcalab_rslrl.orca`; solver integrations are
private implementation details.
"""

from .orca import OrcaRuntimeConfig, list_tasks, make_env, register_task

__version__ = "0.2.0"

__all__ = ["OrcaRuntimeConfig", "list_tasks", "make_env", "register_task"]
