from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
    find_latest_checkpoint,
    load_task_and_train_cfg,
)
from orca_rl.rsl_env.scene_binding import G1_AGENT_ASSET_PATH, GO2_AGENT_ASSET_PATH

ensure_project_root_on_path()


DEFAULT_COMMAND_ARROW_ASSET_PATH = "assets/001d46537b9e555b/commandarrow/prefabs/command_arrow_usda"
DEFAULT_HEADING_ARROW_ASSET_PATH = "assets/001d46537b9e555b/heading_arrow/prefabs/heading_arrow_usda"
DEFAULT_ROUGH_TERRAIN_ASSET_PATH = (
    "assets/e071469a36d3c8aa/test_terrain/prefabs/terrain_usda"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMITIVE_TERRAIN_XML = PROJECT_ROOT / "assets/terrain/orca_primitive_terrain_xml/terrain.xml"
DEFAULT_MJLAB_ROUGH_5X5_XML = PROJECT_ROOT / "assets/terrain/mjlab_rough_5x5_xml/terrain.xml"
DEFAULT_PRIMITIVE_TERRAIN_HFIELD = PROJECT_ROOT / "assets/terrain/orca_primitive_terrain_xml/terrain_height_field.npz"
DEFAULT_MULTI_TERRAIN_COLLISION_XML = PROJECT_ROOT / "assets/terrain/multi_terrain_map/multi_terrain_map_collision.xml"
DEFAULT_MULTI_TERRAIN_HFIELD = PROJECT_ROOT / "assets/terrain/multi_terrain_map/multi_terrain_map_hfield.npz"


def _apply_rough_terrain_play_overrides(task_cfg: dict) -> None:
    terrain_cfg = task_cfg.get("terrain")
    if not isinstance(terrain_cfg, dict) or terrain_cfg.get("terrain_type") == "plane":
        return
    generator_cfg = dict(terrain_cfg.get("terrain_generator") or {})
    if generator_cfg:
        generator_cfg["curriculum"] = False
        generator_cfg["num_cols"] = 5
        generator_cfg["num_rows"] = 5
        generator_cfg["border_width"] = 10.0
        terrain_cfg["terrain_generator"] = generator_cfg
    terrain_cfg["physics_enabled"] = True
    task_cfg["curriculum"] = {}
    events = task_cfg.setdefault("events", {})
    events["randomize_terrain"] = {
        "func": "randomize_terrain",
        "mode": "reset",
        "params": {},
    }
    randomization_cfg = task_cfg.setdefault("randomization", {})
    randomization_cfg["terrain"] = "rough"
    randomization_cfg["terrain_curriculum"] = False


def _checkpoint_actor_input_dim(checkpoint: str | Path, torch_module) -> int | None:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        return None
    payload = torch_module.load(path, map_location="cpu")
    actor_state_dict = payload.get("actor_state_dict") if isinstance(payload, dict) else None
    if not isinstance(actor_state_dict, dict):
        return None
    first_weight = actor_state_dict.get("mlp.0.weight")
    if first_weight is None or not hasattr(first_weight, "shape") or len(first_weight.shape) < 2:
        return None
    return int(first_weight.shape[1])


def _actor_observation_dim(env) -> int | None:
    try:
        observations = env.get_observations()
        policy_obs = observations["policy"]
        return int(policy_obs.shape[1])
    except Exception:
        return None


def _looks_like_mjlab_actor_dim(robot_name: str, input_dim: int) -> bool:
    robot = robot_name.strip().lower()
    if robot in {"go2", "unitree_go2"}:
        return int(input_dim) >= 47
    if robot == "g1":
        return int(input_dim) >= 98
    return False


def _apply_play_scene_mode(task_cfg: dict, *, local_mujoco: bool) -> None:
    task_cfg.setdefault("episode", {})["length_s"] = 1.0e9
    task_cfg.setdefault("observations", {})["add_noise"] = False
    task_cfg.setdefault("randomization", {})["enabled"] = False
    task_cfg["curriculum"] = {}
    events = task_cfg.setdefault("events", {})
    for event_name in (
        "push_robot",
        "randomize_friction",
        "randomize_body_mass",
        "randomize_actuator_properties",
        "randomize_action_latency",
        "randomize_solver_params",
        "randomize_contact_params",
    ):
        events.pop(event_name, None)
    randomization_cfg = task_cfg.setdefault("randomization", {})
    randomization_cfg["max_action_delay_steps"] = 0
    _apply_rough_terrain_play_overrides(task_cfg)
    scene_cfg = task_cfg.setdefault("scene_binding", {})
    if local_mujoco:
        if scene_cfg.get("resolver") in {"g1", "go2"}:
            scene_cfg.setdefault("local_xml_path", "auto")
            scene_cfg.setdefault("local_clone_spacing", 2.0)
        return
    if scene_cfg.get("resolver") == "g1":
        scene_cfg["local_xml_path"] = None
        scene_cfg["asset_path"] = G1_AGENT_ASSET_PATH
        scene_cfg["spawn_if_missing"] = True
        scene_cfg["max_auto_spawn_count"] = max(1, int(task_cfg.get("num_envs", 1)))
        terrain_cfg = task_cfg.get("terrain")
        if isinstance(terrain_cfg, dict) and terrain_cfg.get("terrain_type") == "plane":
            terrain_cfg["physics_enabled"] = False
    elif scene_cfg.get("resolver") == "go2":
        scene_cfg["asset_path"] = GO2_AGENT_ASSET_PATH
        scene_cfg["spawn_if_missing"] = True
        scene_cfg["max_auto_spawn_count"] = max(1, int(task_cfg.get("num_envs", 1)))
        terrain_cfg = task_cfg.get("terrain")
        if isinstance(terrain_cfg, dict) and terrain_cfg.get("terrain_type") == "plane":
            terrain_cfg["physics_enabled"] = False


def _apply_fixed_play_command(
    task_cfg: dict,
    *,
    lin_vel_x: float | None,
    lin_vel_y: float | None,
    ang_vel_z: float | None,
) -> None:
    if lin_vel_x is None and lin_vel_y is None and ang_vel_z is None:
        return
    commands = task_cfg.setdefault("commands", {})
    if lin_vel_x is not None:
        commands["lin_vel_x"] = (float(lin_vel_x), float(lin_vel_x))
    if lin_vel_y is not None:
        commands["lin_vel_y"] = (float(lin_vel_y), float(lin_vel_y))
    if ang_vel_z is not None:
        commands["yaw_vel"] = (float(ang_vel_z), float(ang_vel_z))


def _apply_local_terrain_map_override(task_cfg: dict, terrain_xml: str | Path) -> None:
    xml_path = Path(terrain_xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"Local terrain map XML does not exist: {xml_path}")
    hfield_path: Path | None = None
    if xml_path == DEFAULT_PRIMITIVE_TERRAIN_XML:
        hfield_path = DEFAULT_PRIMITIVE_TERRAIN_HFIELD
    elif xml_path == DEFAULT_MULTI_TERRAIN_COLLISION_XML:
        hfield_path = DEFAULT_MULTI_TERRAIN_HFIELD
    else:
        sibling_hfield = xml_path.with_name(f"{xml_path.stem}_height_field.npz")
        fallback_hfield = xml_path.with_name("terrain_height_field.npz")
        if sibling_hfield.exists():
            hfield_path = sibling_hfield
        elif fallback_hfield.exists():
            hfield_path = fallback_hfield
    if hfield_path is not None and not hfield_path.exists():
        raise FileNotFoundError(f"Local terrain height field does not exist: {hfield_path}")
    task_cfg["terrain"] = {
        "terrain_type": "static_xml",
        "physics_enabled": True,
        "static_xml_path": str(xml_path),
        "height_field_path": str(hfield_path) if hfield_path is not None else None,
        "static_friction": 0.9,
        "dynamic_friction": 0.8,
        "restitution": 0.0,
    }
    randomization_cfg = task_cfg.setdefault("randomization", {})
    randomization_cfg["terrain"] = "rough"
    randomization_cfg["terrain_curriculum"] = False
    task_cfg["curriculum"] = {}


def _is_rough_play_task(task_cfg: dict) -> bool:
    terrain_cfg = task_cfg.get("terrain")
    if not isinstance(terrain_cfg, dict):
        return False
    return terrain_cfg.get("terrain_type") not in {None, "plane"}


def _is_go2_rough_play_task(task_cfg: dict) -> bool:
    return str(task_cfg.get("robot", "")).lower() == "go2" and _is_rough_play_task(task_cfg)


def _apply_rough_terrain_visual_config(
    task_cfg: dict,
    *,
    enabled: bool,
    asset_path: str,
    actor_name: str,
    scale: float,
    auto_publish: bool,
) -> None:
    if not enabled:
        return
    task_cfg["rough_terrain_visual"] = {
        "enabled": True,
        "asset_path": asset_path,
        "actor_name": actor_name,
        "position": [0.0, 0.0, 0.05],
        "rotation_euler": [0.0, 0.0, 0.0],
        "scale": float(scale),
        "auto_publish": bool(auto_publish),
    }


def _command_sweep_vector(args: argparse.Namespace, *, sim_time: float) -> np.ndarray:
    target = np.array(
        [
            0.5 if args.lin_vel_x is None else float(args.lin_vel_x),
            0.0 if args.lin_vel_y is None else float(args.lin_vel_y),
            0.0 if args.ang_vel_z is None else float(args.ang_vel_z),
        ],
        dtype=np.float64,
    )
    if not np.any(np.abs(target) > 1e-9):
        target[0] = 0.5
    period = max(0.1, float(args.command_sweep_period))
    magnitude = 0.5 * (1.0 - math.cos(2.0 * math.pi * sim_time / period))
    return target * magnitude


def _set_env_manual_command(env, command: np.ndarray) -> None:
    command = np.asarray(command, dtype=np.float64).reshape(3)
    for task in getattr(env, "tasks", []):
        commands = np.repeat(command.reshape(1, 3), int(task.num_envs), axis=0)
        if hasattr(task, "set_manual_commands"):
            task.set_manual_commands(commands)
        else:
            for agent in getattr(task, "agents", []):
                agent.command = command.copy()


def _fixed_command_vector(args: argparse.Namespace) -> np.ndarray | None:
    if args.lin_vel_x is None and args.lin_vel_y is None and args.ang_vel_z is None:
        return None
    return np.array(
        [
            0.0 if args.lin_vel_x is None else float(args.lin_vel_x),
            0.0 if args.lin_vel_y is None else float(args.lin_vel_y),
            0.0 if args.ang_vel_z is None else float(args.ang_vel_z),
        ],
        dtype=np.float64,
    )


def _apply_command_arrow_debug_config(
    task_cfg: dict,
    *,
    enabled: bool,
    asset_path: str,
    actor_name: str,
    scale: float,
    joint_name: str | None,
    agent_index: int,
    z_offset: float,
    forward_offset: float,
    lateral_offset: float,
    tail_local_x: float,
    yaw_when_standing: bool,
    mode: str = "command",
    key: str = "command_arrow",
) -> None:
    if not enabled:
        return
    debug_cfg = task_cfg.setdefault("debug_visualization", {})
    debug_cfg[key] = {
        "enabled": True,
        "asset_path": asset_path,
        "actor_name": actor_name,
        "scale": float(scale),
        "joint_name": joint_name,
        "mode": mode,
        "agent_index": int(agent_index),
        "z_offset": float(z_offset),
        "forward_offset": float(forward_offset),
        "lateral_offset": float(lateral_offset),
        "tail_local_x": float(tail_local_x),
        "yaw_when_standing": bool(yaw_when_standing),
    }


def _print_debug_visualization_report(env) -> None:
    arrow_reports = []
    for task in getattr(env, "tasks", []):
        if hasattr(task, "debug_visualization_report"):
            report = task.debug_visualization_report()
            if isinstance(report, dict):
                arrow_reports.extend(report.get("debug_arrows", []))
    arrow_reports = [report for report in arrow_reports if isinstance(report, dict) and report.get("enabled")]
    if not arrow_reports:
        return
    available = sum(1 for report in arrow_reports if report.get("available"))
    summary = ", ".join(
        f"{report.get('mode')}:{report.get('actor_name')}->{report.get('joint_name')}" for report in arrow_reports
    )
    print(
        "[orca_rl.play] Debug arrows: "
        f"count={len(arrow_reports)}, available={available}, {summary}"
    )


def _maybe_publish_command_arrow(task_cfg: dict, *, local_mujoco: bool) -> None:
    debug_cfg = task_cfg.get("debug_visualization", {})
    if local_mujoco or not isinstance(debug_cfg, dict):
        return

    try:
        from orca_rl.rsl_env.debug_visualizer import ensure_command_arrow_actor
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    for address in task_cfg.get("orcagym_addresses") or ["localhost:50051"]:
        for key in ("command_arrow", "heading_arrow"):
            arrow_cfg = debug_cfg.get(key)
            if not isinstance(arrow_cfg, dict) or not arrow_cfg.get("enabled"):
                continue
            if not bool(arrow_cfg.get("auto_publish", True)):
                continue
            try:
                published = ensure_command_arrow_actor(
                    orcagym_addr=str(address),
                    actor_name=str(arrow_cfg.get("actor_name") or "cmd_arrow_000"),
                    asset_path=str(arrow_cfg.get("asset_path") or DEFAULT_COMMAND_ARROW_ASSET_PATH),
                )
            except Exception as exc:
                print(f"[orca_rl.play] {key} auto-publish skipped for {address}: {exc}")
                continue
            if published:
                print(
                    "[orca_rl.play] Debug arrow auto-published: "
                    f"mode={arrow_cfg.get('mode')}, address={address}, "
                    f"actor={arrow_cfg.get('actor_name')}, asset={arrow_cfg.get('asset_path')}"
                )


def _maybe_publish_rough_terrain_visual(task_cfg: dict, *, local_mujoco: bool) -> None:
    if local_mujoco:
        return
    terrain_visual_cfg = task_cfg.get("rough_terrain_visual")
    if not isinstance(terrain_visual_cfg, dict) or not terrain_visual_cfg.get("enabled"):
        return
    if not bool(terrain_visual_cfg.get("auto_publish", True)):
        return

    try:
        from orca_rl.rsl_env.debug_visualizer import ensure_scene_actor
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    for address in task_cfg.get("orcagym_addresses") or ["localhost:50051"]:
        try:
            published = ensure_scene_actor(
                orcagym_addr=str(address),
                actor_name=str(terrain_visual_cfg.get("actor_name") or "mjlab_rough_5x5_terrain"),
                asset_path=str(terrain_visual_cfg.get("asset_path") or DEFAULT_ROUGH_TERRAIN_ASSET_PATH),
                position=terrain_visual_cfg.get("position", [0.0, 0.0, 0.0]),
                rotation_euler=terrain_visual_cfg.get("rotation_euler", [0.0, 0.0, 0.0]),
                scale=float(terrain_visual_cfg.get("scale", 1.0)),
            )
        except Exception as exc:
            print(f"[orca_rl.play] Rough terrain visual auto-publish skipped for {address}: {exc}")
            continue
        if published:
            print(
                "[orca_rl.play] Rough terrain visual auto-published: "
                f"address={address}, actor={terrain_visual_cfg.get('actor_name')}, "
                f"asset={terrain_visual_cfg.get('asset_path')}"
            )


def _configure_debug_arrow_scene_actors(task_cfg: dict, *, local_mujoco: bool) -> None:
    if local_mujoco:
        return
    debug_cfg = task_cfg.get("debug_visualization", {})
    if not isinstance(debug_cfg, dict):
        return
    extra_actors = []
    for key, position in (
        ("command_arrow", [0.0, 0.0, 1.1]),
        ("heading_arrow", [0.0, 0.0, 1.25]),
    ):
        arrow_cfg = debug_cfg.get(key)
        if not isinstance(arrow_cfg, dict) or not arrow_cfg.get("enabled"):
            continue
        if not bool(arrow_cfg.get("auto_publish", True)):
            continue
        extra_actors.append(
            {
                "name": str(arrow_cfg.get("actor_name") or "cmd_arrow_000"),
                "asset_path": str(arrow_cfg.get("asset_path") or DEFAULT_COMMAND_ARROW_ASSET_PATH),
                "position": position,
                "scale": float(arrow_cfg.get("scale", 0.6)),
            }
        )
    if extra_actors:
        task_cfg.setdefault("scene_binding", {}).setdefault("extra_actors", []).extend(extra_actors)


def _configure_rough_terrain_scene_actor(task_cfg: dict, *, local_mujoco: bool) -> None:
    if local_mujoco:
        return
    terrain_visual_cfg = task_cfg.get("rough_terrain_visual")
    if not isinstance(terrain_visual_cfg, dict) or not terrain_visual_cfg.get("enabled"):
        return
    if not bool(terrain_visual_cfg.get("auto_publish", True)):
        return
    task_cfg.setdefault("scene_binding", {}).setdefault("extra_actors", []).append(
        {
            "name": str(terrain_visual_cfg.get("actor_name") or "mjlab_rough_5x5_terrain"),
            "asset_path": str(terrain_visual_cfg.get("asset_path") or DEFAULT_ROUGH_TERRAIN_ASSET_PATH),
            "position": terrain_visual_cfg.get("position", [0.0, 0.0, 0.0]),
            "rotation_euler": terrain_visual_cfg.get("rotation_euler", [0.0, 0.0, 0.0]),
            "scale": float(terrain_visual_cfg.get("scale", 1.0)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Orca locomotion RSL-RL policy.")
    parser.add_argument(
        "--config",
        default="Unitree-Go2-Flat",
        help="Registered task name, or Python cfg file. Use file.py:factory_name for a non-default factory.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List registered task names and exit.")
    parser.add_argument(
        "--mjlab",
        action="store_true",
        help="Shortcut for playing the latest Unitree/mjlab velocity checkpoint for the selected config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="RSL-RL checkpoint path, e.g. model_1000.pt.",
    )
    parser.add_argument(
        "--policy-backend",
        choices=("auto", "orca", "mjlab"),
        default="auto",
        help="`auto` selects the loader from checkpoint dimensions; `orca` loads orca_rl checkpoints; "
        "`mjlab` loads Unitree/mjlab velocity checkpoints.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Playback duration in seconds. Overrides --steps when set.",
    )
    parser.add_argument("--lin-vel-x", type=float, default=None, help="Fixed x velocity command for play.")
    parser.add_argument("--lin-vel-y", type=float, default=None, help="Fixed y velocity command for play.")
    parser.add_argument("--ang-vel-z", type=float, default=None, help="Fixed yaw velocity command for play.")
    parser.add_argument(
        "--command-sweep",
        action="store_true",
        help="Sweep command magnitude from 0 to the requested twist and back for visual/debug tests.",
    )
    parser.add_argument(
        "--command-sweep-period",
        type=float,
        default=6.0,
        help="Seconds for one command magnitude sweep cycle.",
    )
    parser.add_argument(
        "--command-arrow",
        action="store_true",
        help="Publish/update an OrcaLab command arrow actor/freejoint during play.",
    )
    parser.add_argument(
        "--command-arrow-asset",
        default=DEFAULT_COMMAND_ARROW_ASSET_PATH,
        help="Documentation/protocol path for the command arrow USD asset.",
    )
    parser.add_argument("--command-arrow-actor", default="cmd_arrow_000", help="Command arrow actor name in OrcaLab.")
    parser.add_argument(
        "--command-arrow-scale",
        type=float,
        default=0.55,
        help="Scene publish scale for the command arrow actor. Existing actors must be republished to change this.",
    )
    parser.add_argument(
        "--heading-arrow",
        action="store_true",
        help="Also show the robot current velocity arrow. Enabled automatically with --command-arrow unless disabled.",
    )
    parser.add_argument(
        "--no-heading-arrow",
        action="store_true",
        help="Do not create the robot current velocity arrow when --command-arrow is used.",
    )
    parser.add_argument(
        "--heading-arrow-asset",
        default=DEFAULT_HEADING_ARROW_ASSET_PATH,
        help="OrcaLab asset path for the robot current velocity arrow.",
    )
    parser.add_argument("--heading-arrow-actor", default="heading_arrow_000", help="Current velocity arrow actor name in OrcaLab.")
    parser.add_argument(
        "--heading-arrow-scale",
        type=float,
        default=0.5,
        help="Scene publish scale for the current velocity arrow actor. Existing actors must be republished to change this.",
    )
    parser.add_argument(
        "--heading-arrow-joint",
        default=None,
        help="Freejoint name for the robot current velocity arrow. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--no-command-arrow-auto-publish",
        action="store_true",
        help="Only update an existing command arrow actor; do not publish the default OrcaLab asset.",
    )
    parser.add_argument(
        "--command-arrow-joint",
        default=None,
        help="Freejoint name for the command arrow. If omitted, common names are auto-detected.",
    )
    parser.add_argument("--command-arrow-agent", type=int, default=0, help="Agent index the command arrow follows.")
    parser.add_argument("--command-arrow-z", type=float, default=0.9, help="Height above robot base for the arrow.")
    parser.add_argument(
        "--command-arrow-forward",
        type=float,
        default=0.0,
        help="Forward offset in the robot base frame.",
    )
    parser.add_argument(
        "--command-arrow-lateral",
        type=float,
        default=0.0,
        help="Lateral offset in the robot base frame.",
    )
    parser.add_argument(
        "--command-arrow-tail-x",
        type=float,
        default=-0.25,
        help="Arrow tail x position in the asset local frame. Legacy OrcaLab arrows use about -0.25.",
    )
    parser.add_argument(
        "--no-command-arrow-yaw",
        action="store_true",
        help="Do not show a side arrow for pure yaw commands.",
    )
    parser.add_argument(
        "--local-mujoco",
        action="store_true",
        help="Play through the generated local MuJoCo MJCF instead of the OrcaLab scene.",
    )
    parser.add_argument(
        "--local-terrain-map",
        nargs="?",
        const="auto",
        default=None,
        help="Use a static MuJoCo terrain collision XML for local-mujoco play. "
        "Omit the value to use the mjlab rough 5x5 terrain for rough configs, "
        "or assets/terrain/orca_primitive_terrain_xml/terrain.xml for flat configs.",
    )
    parser.add_argument(
        "--no-rough-terrain-visual",
        action="store_true",
        help="Do not auto-publish the OrcaLab visual terrain asset for rough play configs.",
    )
    parser.add_argument(
        "--rough-terrain-asset",
        default=DEFAULT_ROUGH_TERRAIN_ASSET_PATH,
        help="OrcaLab spawnable path for the rough terrain visual asset.",
    )
    parser.add_argument(
        "--rough-terrain-actor",
        default="mjlab_rough_5x5_terrain",
        help="Actor name used when auto-publishing the rough terrain visual asset.",
    )
    parser.add_argument(
        "--rough-terrain-scale",
        type=float,
        default=1.0,
        help="Scale for the rough terrain visual actor.",
    )
    parser.add_argument(
        "--remote",
        default=None,
        help="Override OrcaGym address, e.g. localhost:50051.",
    )
    args = parser.parse_args()

    if args.list_tasks:
        from orca_rl.registry import list_tasks

        for spec in list_tasks():
            suffix = f" - {spec.description}" if spec.description else ""
            print(f"{spec.name}{suffix}")
        return

    if args.mjlab:
        args.policy_backend = "mjlab"
        if args.lin_vel_x is None:
            args.lin_vel_x = 0.5
        if args.lin_vel_y is None:
            args.lin_vel_y = 0.0
        if args.ang_vel_z is None:
            args.ang_vel_z = 0.0

    task_cfg, train_cfg = load_task_and_train_cfg(args.config)
    if _is_go2_rough_play_task(task_cfg):
        if args.lin_vel_x is None:
            args.lin_vel_x = 0.5
        if args.lin_vel_y is None:
            args.lin_vel_y = 0.0
        if args.ang_vel_z is None:
            args.ang_vel_z = 0.0
    if args.local_terrain_map is not None:
        args.local_mujoco = True
        terrain_xml = args.local_terrain_map
        if terrain_xml == "auto":
            terrain_xml = DEFAULT_MJLAB_ROUGH_5X5_XML if _is_rough_play_task(task_cfg) else DEFAULT_PRIMITIVE_TERRAIN_XML
        _apply_local_terrain_map_override(task_cfg, terrain_xml)
    _apply_fixed_play_command(
        task_cfg,
        lin_vel_x=args.lin_vel_x,
        lin_vel_y=args.lin_vel_y,
        ang_vel_z=args.ang_vel_z,
    )
    _apply_command_arrow_debug_config(
        task_cfg,
        enabled=bool(args.command_arrow),
        asset_path=args.command_arrow_asset,
        actor_name=args.command_arrow_actor,
        scale=args.command_arrow_scale,
        joint_name=args.command_arrow_joint,
        agent_index=args.command_arrow_agent,
        z_offset=args.command_arrow_z,
        forward_offset=args.command_arrow_forward,
        lateral_offset=args.command_arrow_lateral,
        tail_local_x=args.command_arrow_tail_x,
        yaw_when_standing=not bool(args.no_command_arrow_yaw),
        mode="command",
        key="command_arrow",
    )
    if args.command_arrow:
        task_cfg["debug_visualization"]["command_arrow"]["auto_publish"] = not bool(args.no_command_arrow_auto_publish)
    enable_heading_arrow = bool(args.heading_arrow) or (bool(args.command_arrow) and not bool(args.no_heading_arrow))
    _apply_command_arrow_debug_config(
        task_cfg,
        enabled=enable_heading_arrow,
        asset_path=args.heading_arrow_asset,
        actor_name=args.heading_arrow_actor,
        scale=args.heading_arrow_scale,
        joint_name=args.heading_arrow_joint,
        agent_index=args.command_arrow_agent,
        z_offset=args.command_arrow_z + 0.15,
        forward_offset=args.command_arrow_forward,
        lateral_offset=args.command_arrow_lateral,
        tail_local_x=args.command_arrow_tail_x,
        yaw_when_standing=False,
        mode="velocity",
        key="heading_arrow",
    )
    if enable_heading_arrow:
        task_cfg["debug_visualization"]["heading_arrow"]["auto_publish"] = not bool(args.no_command_arrow_auto_publish)
    _apply_rough_terrain_visual_config(
        task_cfg,
        enabled=_is_rough_play_task(task_cfg) and not bool(args.no_rough_terrain_visual),
        asset_path=args.rough_terrain_asset,
        actor_name=args.rough_terrain_actor,
        scale=args.rough_terrain_scale,
        auto_publish=True,
    )
    _configure_rough_terrain_scene_actor(task_cfg, local_mujoco=bool(args.local_mujoco))
    _configure_debug_arrow_scene_actors(task_cfg, local_mujoco=bool(args.local_mujoco))
    apply_remote_override(task_cfg, args.remote)
    _apply_play_scene_mode(task_cfg, local_mujoco=bool(args.local_mujoco))
    task_cfg.setdefault("sim", {})["render_mode"] = "human"
    task_cfg["sim"]["headless"] = False
    check_orcagym_addresses(task_cfg)
    _maybe_publish_rough_terrain_visual(task_cfg, local_mujoco=bool(args.local_mujoco))
    _maybe_publish_command_arrow(task_cfg, local_mujoco=bool(args.local_mujoco))

    try:
        import torch

        from orca_rl import make_locomotion_vec_env, print_runtime_summary
        from orca_rl.rsl_env.runtime_policy import load_inference_runner
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    device = args.device or task_cfg.get("play", {}).get("device", "cpu")
    robot_name = str(task_cfg.get("robot", "")).strip().lower()
    project_root = Path(__file__).resolve().parents[1]
    terrain_name = "rough" if _is_rough_play_task(task_cfg) else "flat"
    effective_policy_backend = args.policy_backend
    if args.policy_backend == "mjlab":
        from orca_rl.rsl_env.mjlab_policy import find_latest_unitree_mjlab_checkpoint

        checkpoint = str(
            args.checkpoint
            or find_latest_unitree_mjlab_checkpoint(
                project_root,
                robot=robot_name or "g1",
                terrain=terrain_name,
            )
        )
    elif args.policy_backend == "auto" and args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = args.checkpoint or str(
            find_latest_checkpoint(task_name=str(train_cfg.get("experiment_name") or task_cfg.get("name", "")) or None)
        )
    try:
        env = make_locomotion_vec_env(task_cfg, device=device, render_mode="human", headless=False)
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc
    try:
        if effective_policy_backend == "auto":
            checkpoint_actor_dim = _checkpoint_actor_input_dim(checkpoint, torch)
            env_actor_dim = _actor_observation_dim(env)
            if (
                checkpoint_actor_dim is not None
                and env_actor_dim is not None
                and checkpoint_actor_dim != env_actor_dim
                and _looks_like_mjlab_actor_dim(robot_name, checkpoint_actor_dim)
            ):
                effective_policy_backend = "mjlab"
            else:
                effective_policy_backend = "orca"
            print(
                "[orca_rl.play] Auto policy backend: "
                f"{effective_policy_backend} "
                f"(checkpoint_actor_obs={checkpoint_actor_dim}, env_actor_obs={env_actor_dim})"
            )
        print_runtime_summary(
            mode="play",
            env=env,
            task_cfg=task_cfg,
            runner_cfg=train_cfg,
            device=device,
            checkpoint=checkpoint,
        )
        _print_debug_visualization_report(env)
        if effective_policy_backend == "mjlab":
            from orca_rl.rsl_env.mjlab_policy import MjlabRslRlActorPolicy, make_mjlab_orca_play_bridge

            policy = MjlabRslRlActorPolicy.from_checkpoint(checkpoint, device=device)
            if policy.output_dim != env.num_actions:
                raise ValueError(
                    "Unitree/mjlab checkpoint action dimension does not match the OrcaLab scene robot. "
                    f"checkpoint_action_dim={policy.output_dim}, env_num_actions={env.num_actions}, "
                    f"config_robot={robot_name or 'unknown'}"
                )
            bridge = make_mjlab_orca_play_bridge(env, expected_obs_dim=policy.input_dim, robot=robot_name)
            fixed_command = _fixed_command_vector(args)
            if args.command_sweep:
                _set_env_manual_command(env, _command_sweep_vector(args, sim_time=0.0))
            elif fixed_command is not None:
                _set_env_manual_command(env, fixed_command)
            obs = bridge.get_observations()
            alignment = bridge.alignment_report
            print(
                "[orca_rl.play] Loaded Unitree/mjlab policy bridge: "
                f"obs_dim={policy.input_dim}, action_dim={policy.output_dim}, checkpoint={checkpoint}"
            )
            print(
                "[orca_rl.play] Mjlab runtime alignment: "
                f"tasks={alignment['tasks']}, agents={alignment['agents']}, "
                f"joints={alignment['joints']}, actuators={alignment['actuators']}, "
                f"position_actuator_tasks={alignment['position_actuator_tasks']}"
            )
            if "contact_geoms" in alignment and str(robot_name).lower() == "g1":
                print(
                    "[orca_rl.play] G1 mjlab contact alignment: "
                    f"contact_geoms={alignment.get('contact_geoms', 0)}, "
                    f"foot_contact_geoms={alignment.get('foot_contact_geoms', 0)}, "
                    f"nonfoot_contact_geoms={alignment.get('nonfoot_contact_geoms', 0)}, "
                    f"imu_gyro_sensors={alignment.get('imu_gyro_sensors', 0)}"
                )
            elif "imu_gyro_sensors" in alignment and "foot_contact_geoms" not in alignment:
                print(
                    "[orca_rl.play] Mjlab sensor alignment: "
                    f"imu_gyro_sensors={alignment.get('imu_gyro_sensors', 0)}"
                )
            if "foot_contact_geoms" in alignment:
                print(
                    "[orca_rl.play] GO2 mjlab contact alignment: "
                    f"contact_geoms={alignment.get('contact_geoms', 0)}, "
                    f"foot_contact_geoms={alignment.get('foot_contact_geoms', 0)}, "
                    f"nonfoot_contact_geoms={alignment.get('nonfoot_contact_geoms', 0)}, "
                    f"base_height_resets={alignment.get('base_height_resets', 0)}, "
                    f"imu_gyro_sensors={alignment.get('imu_gyro_sensors', 0)}"
                )
            if args.lin_vel_x is not None or args.lin_vel_y is not None or args.ang_vel_z is not None:
                print(
                    "[orca_rl.play] Fixed command: "
                    f"vx={args.lin_vel_x if args.lin_vel_x is not None else 'sampled'}, "
                    f"vy={args.lin_vel_y if args.lin_vel_y is not None else 'sampled'}, "
                    f"wz={args.ang_vel_z if args.ang_vel_z is not None else 'sampled'}"
                )
            if args.command_sweep:
                print(
                    "[orca_rl.play] Command sweep enabled: "
                    f"target_vx={args.lin_vel_x if args.lin_vel_x is not None else 0.5}, "
                    f"target_vy={args.lin_vel_y if args.lin_vel_y is not None else 0.0}, "
                    f"target_wz={args.ang_vel_z if args.ang_vel_z is not None else 0.0}, "
                    f"period={args.command_sweep_period}s"
                )
        else:
            _runner, policy = load_inference_runner(env, train_cfg, checkpoint, log_dir=None, device=device)
            if args.command_sweep:
                _set_env_manual_command(env, _command_sweep_vector(args, sim_time=0.0))
            obs = env.get_observations().to(device)
        step = 0
        dt = float(task_cfg["sim"]["time_step"]) * int(task_cfg["sim"]["frame_skip"]) * int(task_cfg["sim"]["decimation"])
        max_steps = int(round(float(args.seconds) / dt)) if args.seconds is not None else int(args.steps)
        while max_steps <= 0 or step < max_steps:
            start = time.perf_counter()
            if args.command_sweep:
                command = _command_sweep_vector(args, sim_time=step * dt)
                _set_env_manual_command(env, command)
                if effective_policy_backend == "mjlab":
                    obs = bridge.get_observations()
                else:
                    obs = env.get_observations().to(device)
            if effective_policy_backend == "mjlab":
                actions_np = policy.act_numpy(obs, device=device)
                obs = bridge.step(actions_np)
            else:
                with torch.inference_mode():
                    actions = policy(obs, stochastic_output=False)
                obs, _rewards, _dones, _extras = env.step(actions.to(env.device))
                obs = obs.to(device)
            step += 1
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Interrupted RSL-RL playback.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
