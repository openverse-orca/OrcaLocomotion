from __future__ import annotations

import sys
import os
import shutil
import socket
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _split_config_selector(path: str | Path) -> tuple[str | Path, str | None]:
    text = str(path)
    if ".py:" not in text:
        return path, None
    file_text, _, factory_name = text.partition(".py:")
    if not factory_name:
        raise ValueError(f"Config factory selector is empty: {path}")
    return f"{file_text}.py", factory_name


def _resolve_config_path(path: str | Path, base_dir: Path | None = None) -> Path:
    config_path, _ = _split_config_selector(path)
    raw_path = Path(config_path).expanduser()
    if raw_path.is_absolute():
        return raw_path

    search_roots = [base_dir, PROJECT_ROOT, Path.cwd()]
    for root in search_roots:
        if root is None:
            continue
        candidate = (root / raw_path).resolve()
        if candidate.exists():
            return candidate
    return ((base_dir or PROJECT_ROOT) / raw_path).resolve()


def _resolve_referenced_config(path: str | Path, owner_path: str | Path) -> str:
    config_path, factory_name = _split_config_selector(path)
    resolved = _resolve_config_path(config_path, _resolve_config_path(owner_path).parent)
    return f"{resolved}:{factory_name}" if factory_name else str(resolved)


def _config_to_dict(config: Any, path: Path) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        raise ValueError(f"Config factory must return a dict-like object: {path}")
    return config


def _load_python_config(
    path: Path,
    factory_names: tuple[str, ...],
    factory_selector: str | None = None,
) -> dict[str, Any]:
    ensure_project_root_on_path()
    module_name = f"_orca_locomotion_cfg_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Python config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if factory_selector is not None:
        factory = getattr(module, factory_selector, None)
        if not callable(factory):
            raise ValueError(f"Python config factory is not callable: {path}:{factory_selector}")
        return _config_to_dict(factory(), path)

    for factory_name in factory_names:
        factory = getattr(module, factory_name, None)
        if callable(factory):
            return _config_to_dict(factory(), path)

    config = getattr(module, "CONFIG", None)
    if config is not None:
        return _config_to_dict(config, path)

    public_factories = [
        getattr(module, name)
        for name in dir(module)
        if (name.endswith("_env_cfg") or name.endswith("_runner_cfg")) and callable(getattr(module, name))
    ]
    if len(public_factories) == 1:
        return _config_to_dict(public_factories[0](), path)

    expected = ", ".join(factory_names)
    raise ValueError(f"Python config must define one of [{expected}] or CONFIG: {path}")


def load_config(path: str | Path, factory_names: tuple[str, ...] = ("CONFIG_FACTORY",)) -> dict[str, Any]:
    _, factory_selector = _split_config_selector(path)
    config_path = _resolve_config_path(path)
    if config_path.suffix == ".py":
        return _load_python_config(config_path, factory_names, factory_selector)
    raise ValueError(f"Locomotion configs are Python cfg files now. Expected .py, got: {config_path}")


def load_task_and_train_cfg(task_config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_cfg = load_config(task_config_path, ("TASK_CONFIG_FACTORY", "ENV_CONFIG_FACTORY", "CONFIG_FACTORY"))
    train_cfg_path = _resolve_referenced_config(task_cfg["rsl_rl_config"], task_config_path)
    raw_train_cfg = load_config(train_cfg_path, ("RL_CONFIG_FACTORY", "RUNNER_CONFIG_FACTORY", "CONFIG_FACTORY"))
    train_cfg = raw_train_cfg.get("runner", raw_train_cfg)
    return task_cfg, train_cfg


def apply_remote_override(task_cfg: dict[str, Any], remote: str | None) -> None:
    if not remote:
        return
    addresses = [address.strip() for address in remote.split(",") if address.strip()]
    if not addresses:
        raise ValueError("--remote was provided but no usable address was found.")
    task_cfg["orcagym_addresses"] = addresses


def apply_logging_overrides(
    train_cfg: dict[str, Any],
    *,
    logger: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
) -> None:
    if logger:
        train_cfg["logger"] = logger
    if wandb_project:
        train_cfg["wandb_project"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_USERNAME"] = wandb_entity
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode


def check_wandb_dependency(train_cfg: dict[str, Any]) -> None:
    if str(train_cfg.get("logger", "tensorboard")).lower() != "wandb":
        return
    if not train_cfg.get("wandb_project"):
        raise RuntimeError("W&B logging requires `wandb_project` in the runner config or `--wandb-project`.")
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but the `wandb` package is not installed in the OrcaLab Python. "
            "Install it with: pip install wandb"
        ) from exc


def check_orcagym_addresses(task_cfg: dict[str, Any], timeout_s: float = 2.0) -> None:
    addresses = task_cfg.get("orcagym_addresses") or []
    if not addresses:
        raise ValueError("task config must define at least one orcagym address.")
    failures = []
    for address in addresses:
        try:
            host, port = _split_host_port(str(address))
            with socket.create_connection((host, port), timeout=timeout_s):
                pass
        except OSError as exc:
            failures.append(f"{address} ({exc})")
    if failures:
        raise RuntimeError(
            "Cannot connect to OrcaGym gRPC server. "
            "Start OrcaLab/OrcaStudio with the simulation service running, or pass "
            "`--remote host:port` / update `orcagym_addresses` in the locomotion config. "
            f"Failed address(es): {', '.join(failures)}"
        )


def _split_host_port(address: str) -> tuple[str, int]:
    if address.count(":") != 1:
        raise ValueError(f"Expected OrcaGym address in host:port form, got: {address}")
    host, port_text = address.rsplit(":", 1)
    if not host:
        raise ValueError(f"Missing host in OrcaGym address: {address}")
    return host, int(port_text)


def make_log_dir(
    prefix: str | None = None,
    task_name: str = "locomotion",
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment = experiment_name or task_name
    name = prefix or timestamp
    if prefix is None and run_name:
        name = f"{name}_{run_name}"
    path = DEFAULT_LOG_ROOT / experiment / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_checkpoint(root: str | Path | None = None, task_name: str | None = None) -> Path:
    search_roots = [Path(root)] if root is not None else [DEFAULT_LOG_ROOT]
    existing_roots = [path for path in search_roots if path.exists()]
    if not existing_roots:
        raise FileNotFoundError(f"No RSL-RL checkpoint root found: {search_roots[0]}")

    checkpoints = []
    for search_root in existing_roots:
        checkpoints.extend(search_root.rglob("model_*.pt"))
    if task_name:
        task_checkpoints = [
            path for path in checkpoints if any(part.startswith(task_name) for part in path.parts)
        ]
        if task_checkpoints:
            checkpoints = task_checkpoints
    checkpoints = sorted(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No RSL-RL checkpoints found under: {search_root}")
    return checkpoints[0]


def save_checkpoint_aliases(log_dir: Path, iteration: int) -> list[Path]:
    source = log_dir / f"model_{iteration}.pt"
    if not source.exists():
        checkpoints = sorted(
            log_dir.glob("model_*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No RSL-RL checkpoint found under: {log_dir}")
        source = checkpoints[0]

    aliases = [log_dir / "model_last.pt", log_dir / "model_final.pt"]
    for alias in aliases:
        shutil.copy2(source, alias)
    return aliases


def explain_missing_runtime_dependency(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "RSL-RL locomotion runtime dependencies are not installed in this Python environment. "
        "Activate the OrcaLab environment and install orca_rl/requirements.txt. "
        f"Original import error: {exc}"
    )
