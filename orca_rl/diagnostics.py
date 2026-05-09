from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def print_runtime_summary(
    *,
    mode: str,
    env: Any,
    task_cfg: dict[str, Any],
    runner_cfg: dict[str, Any],
    device: str,
    checkpoint: str | None = None,
    log_dir: Any | None = None,
    iterations: int | None = None,
) -> None:
    """Print a compact runtime summary before training/play/eval starts."""

    try:
        import torch
    except ImportError:
        torch = None

    observations = env.get_observations()
    obs_dims = _observation_dims(observations)
    obs_groups = runner_cfg.get("obs_groups", {})
    actor_dim = _group_dim(obs_dims, obs_groups.get("actor", ()))
    critic_dim = _group_dim(obs_dims, obs_groups.get("critic", ()))
    manager_terms = task_cfg.get("manager_terms", {})

    _print_bar()
    print("ORCA_RL RUNTIME SUMMARY")
    _print_bar()
    _print_kv("mode", mode)
    _print_kv("task", task_cfg.get("name", "unknown"))
    _print_kv("robot", task_cfg.get("robot", "unknown"))
    _print_kv("device", device)
    _print_kv("gpu", _gpu_summary(torch, device))
    _print_kv("num_envs", getattr(env, "num_envs", "unknown"))
    _print_kv("num_actions", getattr(env, "num_actions", "unknown"))
    _print_kv("max_episode_length", getattr(env, "max_episode_length", "unknown"))
    _print_kv("control_dt", _control_dt(task_cfg))
    if iterations is not None:
        _print_kv("iterations", iterations)
    if log_dir is not None:
        _print_kv("log_dir", log_dir)
    if checkpoint is not None:
        _print_kv("checkpoint", checkpoint)

    print("\n[Observations]")
    for name, dim in obs_dims.items():
        print(f"  - {name}: {dim}")
    _print_kv("actor_obs", actor_dim, indent=2)
    _print_kv("critic_obs", critic_dim, indent=2)
    _print_kv("obs_groups", obs_groups, indent=2)

    print("\n[Actions]")
    control = task_cfg.get("control", {})
    _print_kv("clip", control.get("action_clip"), indent=2)
    _print_kv("safety_scale", control.get("safety_scale"), indent=2)
    _print_kv("max_delta_dim", len(control.get("max_delta", ())), indent=2)

    print("\n[Rewards]")
    for name, term in _sorted_terms(manager_terms.get("rewards", {})):
        weight = term.get("weight", "-")
        func = _short_func(term.get("func", ""))
        params = _compact_params(term.get("params", {}))
        print(f"  - {name}: weight={weight} func={func}{params}")

    print("\n[Terminations]")
    for name, term in _sorted_terms(manager_terms.get("terminations", {})):
        func = _short_func(term.get("func", ""))
        params = _compact_params(term.get("params", {}))
        timeout = " time_out=True" if term.get("time_out") else ""
        print(f"  - {name}: func={func}{timeout}{params}")

    print("\n[Commands]")
    for name, term in _sorted_terms(manager_terms.get("commands", {})):
        func = _short_func(term.get("func", ""))
        ranges = term.get("ranges", {})
        print(f"  - {name}: func={func} resample_s={term.get('resampling_time_s')} ranges={ranges}")

    print("\n[Domain Randomization]")
    randomization = task_cfg.get("randomization", {})
    _print_kv("enabled", randomization.get("enabled"), indent=2)
    for key, value in randomization.items():
        if key != "enabled":
            _print_kv(key, value, indent=2)
    randomization_events = {
        name: term
        for name, term in manager_terms.get("events", {}).items()
        if "random" in name.lower() or "friction" in name.lower() or "mass" in name.lower()
    }
    for name, term in _sorted_terms(randomization_events):
        print(f"  - event {name}: mode={term.get('mode')} func={_short_func(term.get('func', ''))}")

    print("\n[Terrain]")
    terrain = task_cfg.get("terrain") or {}
    if isinstance(terrain, Mapping):
        _print_kv("type", terrain.get("terrain_type"), indent=2)
        _print_kv("physics_enabled", terrain.get("physics_enabled"), indent=2)
        _print_kv("export_path", terrain.get("export_path"), indent=2)
        generator = terrain.get("terrain_generator")
        if isinstance(generator, Mapping):
            _print_kv("generator", f"{generator.get('num_rows')}x{generator.get('num_cols')}", indent=2)
            _print_kv("curriculum", generator.get("curriculum"), indent=2)
    else:
        _print_kv("type", terrain, indent=2)

    print("\n[Sensors]")
    sensors = task_cfg.get("sensors", {})
    for name, sensor in sorted(sensors.items()):
        if isinstance(sensor, Mapping):
            print(f"  - {name}: {sensor.get('name', name)}")
        else:
            print(f"  - {name}: {sensor}")

    print("\n[Curriculum]")
    for name, term in _sorted_terms(manager_terms.get("curriculum", {})):
        print(f"  - {name}: func={_short_func(term.get('func', ''))}{_compact_params(term.get('params', {}))}")

    print("\n[Scene]")
    scene_binding = task_cfg.get("scene_binding", {})
    _print_kv("resolver", scene_binding.get("resolver"), indent=2)
    _print_kv("addresses", task_cfg.get("orcagym_addresses"), indent=2)
    _print_bar()


def _observation_dims(observations: Any) -> dict[str, int]:
    dims: dict[str, int] = {}
    for key, value in observations.items():
        shape = tuple(value.shape)
        dims[str(key)] = int(shape[-1]) if shape else 1
    return dims


def _group_dim(obs_dims: dict[str, int], group: Any) -> int:
    if not group:
        return 0
    return int(sum(obs_dims.get(str(name), 0) for name in group))


def _gpu_summary(torch: Any, device: str) -> str:
    if torch is None:
        return "torch unavailable"
    if not torch.cuda.is_available():
        return "cuda unavailable"
    try:
        device_obj = torch.device(device)
    except (TypeError, RuntimeError):
        device_obj = torch.device("cuda:0")
    if device_obj.type != "cuda":
        current = torch.cuda.current_device()
    elif device_obj.index is None:
        current = torch.cuda.current_device()
    else:
        current = int(device_obj.index)
    props = torch.cuda.get_device_properties(current)
    total_gb = props.total_memory / (1024**3)
    return (
        f"cuda:{current} {props.name}, capability={props.major}.{props.minor}, "
        f"total_memory={total_gb:.2f} GiB"
    )


def _control_dt(task_cfg: dict[str, Any]) -> float | str:
    sim = task_cfg.get("sim", {})
    try:
        return float(sim["time_step"]) * int(sim["frame_skip"]) * int(sim["decimation"])
    except (KeyError, TypeError, ValueError):
        return "unknown"


def _sorted_terms(terms: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(name, dict(terms[name])) for name in sorted(terms)]


def _short_func(func: Any) -> str:
    text = str(func)
    return text.rsplit(".", 1)[-1]


def _compact_params(params: Any) -> str:
    if not params:
        return ""
    if not isinstance(params, Mapping):
        return f" params={params}"
    compact = {
        key: value
        for key, value in params.items()
        if key not in {"asset_cfg"}
    }
    if not compact:
        return ""
    return f" params={compact}"


def _print_kv(key: str, value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def _print_bar() -> None:
    print("=" * 80)
