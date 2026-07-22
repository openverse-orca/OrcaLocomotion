from __future__ import annotations

import importlib
import inspect
from pathlib import Path


def load_symbol(spec: str):
    if ":" not in spec:
        raise ValueError("Expected import spec 'package.module:symbol'")
    module, symbol = spec.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


def load_yaml(path: str | Path) -> dict:
    import yaml
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def call_with_supported_kwargs(func, **kwargs):
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)
    return func(**{key: value for key, value in kwargs.items() if key in signature.parameters})
