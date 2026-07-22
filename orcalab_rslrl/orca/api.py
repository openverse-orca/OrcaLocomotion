"""Public Orca task API."""

from __future__ import annotations

from .config import OrcaRuntimeConfig


def make_env(task: str, config: OrcaRuntimeConfig | None = None, **overrides):
    """Create a registered Orca environment.

    ``overrides`` is intended for small interactive experiments. Production
    callers should prefer :class:`OrcaRuntimeConfig` for validation.
    """

    from ..tasks import make_task
    from ..tools.common import call_with_supported_kwargs

    runtime = config or OrcaRuntimeConfig()
    kwargs = runtime.task_kwargs()
    kwargs.update(overrides)
    return call_with_supported_kwargs(lambda **values: make_task(task, **values), **kwargs)
