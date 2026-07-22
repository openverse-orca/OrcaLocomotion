from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .common import call_with_supported_kwargs, load_yaml
from .torch_backends import configure_torch_backends


def _load_policy(env, args):
    from rsl_rl.runners import OnPolicyRunner

    cfg = load_yaml(args.runner_config)
    cfg.pop("max_iterations", None)
    cfg["logger"] = "tensorboard"  # never spin up W&B during play
    runner = OnPolicyRunner(env, cfg, log_dir=str(Path(args.checkpoint).parent), device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    return runner.get_inference_policy(device=args.device)


def _play_headless(env, policy, steps: int) -> None:
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))


def _make_play_env(args):
    """Build the registered Orca task in play mode."""

    common_kwargs = {
        "num_envs": args.num_envs,
        "device": args.device,
        "headless": True,
        "mjcf_path": args.asset,
        "play": True,
        "physics_timestep": args.physics_timestep,
        "terrain_kind": args.terrain_kind,
    }
    from ..tasks import make_task

    factory = lambda **kwargs: make_task(args.task, **kwargs)
    return call_with_supported_kwargs(factory, **common_kwargs)


def _print_play_runtime(env) -> None:
    inner = env.env
    print(
        "[orca] runtime: "
        f"physics_dt={inner.physics_dt:g}s, decimation={inner.cfg.decimation}, "
        f"control_dt={inner.step_dt:g}s, num_envs={env.num_envs}"
    )


def _play_orcalab(env, policy, args) -> None:
    """Run a batched policy rollout rendered live in OrcaLab.

    The policy keeps stepping all worlds on GPU with the random twist commands
    resampled by the task's UniformVelocityCommand; each control step the
    batched qpos is scattered into the combined OrcaLab scene and streamed via
    UpdateLocalEnv, throttled to ~render_fps.
    """
    from ..orcalab_batch_render import OrcaLabBatchRenderer

    inner = env.env  # ManagerBasedRLEnv under the RSL-RL adapter
    orca = inner.orca
    print(
        "[orca] physics: "
        f"physics_dt={inner.physics_dt:g}s ({1.0 / inner.physics_dt:.0f}Hz), "
        f"decimation={inner.cfg.decimation}, control_dt={inner.step_dt:g}s"
    )
    spawn_center = args.spawn_center
    if args.terrain_align_offset is not None:
        if spawn_center is not None:
            raise ValueError("Use only one of --spawn-center or --terrain-align-offset")
        spawn_center = args.terrain_align_offset
    renderer = OrcaLabBatchRenderer(
        orcagym_addr=args.orca_addr,
        num_envs=env.num_envs,
        joint_qpos_addr=orca.joint_qpos_addresses(),
        agent_prefix=args.agent_prefix,
        asset_path=args.asset_path,
        terrain_asset_path=args.terrain_asset_path,
        terrain_position=args.terrain_position,
        spawn_center=spawn_center,
        spacing=args.spacing,
        spawn_range=args.spawn_range,
        root_xy_scale=args.root_xy_scale,
        render_root_offset=args.render_root_offset,
        scene_timestep=args.orcalab_scene_timestep or args.physics_timestep,
        disable_air_resistance=not args.keep_orcalab_air_resistance,
        publish=not args.no_publish,
    )
    if args.dump_orcalab_geoms:
        geoms = renderer.dump_scene_geoms(
            args.dump_orcalab_geoms,
            collisions_only=not args.dump_orcalab_all_geoms,
            include_robot=args.dump_orcalab_robot_geoms,
        )
        print(f"[orcalab-play] dumped {len(geoms)} OrcaLab geoms to {args.dump_orcalab_geoms}")
        if args.dump_orcalab_geoms_only:
            renderer.close()
            return
    step_dt = inner.step_dt
    render_interval = 1.0 / args.render_fps
    obs = env.get_observations()
    sim_time = 0.0
    last_render = 0.0
    try:
        with torch.inference_mode():
            for _ in range(args.steps):
                start = time.perf_counter()
                obs, _, _, _ = env.step(policy(obs))
                sim_time += step_dt
                now = time.perf_counter()
                if now - last_render >= render_interval:
                    qpos = orca.state.qpos.detach().cpu().numpy()
                    renderer.render(qpos, sim_time)
                    last_render = now
                if args.realtime:
                    remaining = step_dt - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint with Orca")
    parser.add_argument("--task", default="G1-Velocity-Flat", help="Registered Orca task id")
    parser.add_argument("--runner-config", default="configs/train/ppo.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=300)
    parser.add_argument("--asset", help="Optional local robot model override")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument(
        "--physics-timestep",
        type=float,
        help="Override the Orca physics timestep, e.g. 0.001 for 1000Hz.",
    )
    # OrcaLab rendering.
    parser.add_argument("--orcalab", action="store_true", help="Render the batch live in OrcaLab")
    parser.add_argument("--orca-addr", default="localhost:50051", help="OrcaLab bridge address")
    parser.add_argument(
        "--orcalab-scene-timestep",
        type=float,
        help="Override OrcaLab rendered scene timestep; defaults to --physics-timestep when provided.",
    )
    parser.add_argument(
        "--keep-orcalab-air-resistance",
        action="store_true",
        help="Keep OrcaLab Day scene wind/density/viscosity instead of clearing fluid drag.",
    )
    parser.add_argument("--agent-prefix", default="g1")
    parser.add_argument(
        "--terrain-kind",
        default="flat",
        choices=("flat", "stair-mid-flat"),
        help="Orca physics terrain. Use stair-mid-flat with the matching OrcaLab prefab.",
    )
    parser.add_argument(
        "--asset-path",
        default="assets/e071469a36d3c8aa/unitree_robots/prefabs/g1_29dof_usda",
        help="OrcaStudio spawnable asset for one robot actor",
    )
    parser.add_argument(
        "--terrain-asset-path",
        help="Optional OrcaStudio terrain prefab to publish for rendering, e.g. assets/.../terrain_stair_mid_flat_usda.",
    )
    parser.add_argument(
        "--terrain-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Override the OrcaLab terrain actor position. "
            "Defaults to 0 0 0; use --spawn-center to align robots with authored-offset prefabs."
        ),
    )
    parser.add_argument(
        "--spawn-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Center of the OrcaLab robot actor layout. "
            "Defaults to 0 0 0; tune this when a terrain prefab has authored offsets."
        ),
    )
    parser.add_argument(
        "--terrain-align-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Alias for --spawn-center, used to align rendered robots with an OrcaLab terrain prefab.",
    )
    parser.add_argument("--spacing", type=float, default=2.5, help="Grid spacing between actors (m)")
    parser.add_argument(
        "--spawn-range",
        type=float,
        help=(
            "Optional half-width (m) for a dense center-first OrcaLab spawn area. "
            "Default keeps the original unbounded square grid."
        ),
    )
    parser.add_argument(
        "--root-xy-scale",
        type=float,
        default=1.0,
        help=(
            "Visual-only scale for each Orca world's root x/y displacement before adding "
            "the OrcaLab spawn offset. Use <1.0 to keep rendered robots clustered."
        ),
    )
    parser.add_argument(
        "--render-root-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Visual-only xyz offset added to every robot root each render frame. "
            "Use this to tune terrain visual alignment or lift robots out of visual clipping."
        ),
    )
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument(
        "--dump-orcalab-geoms",
        help="Write OrcaLab compiled scene geoms to JSON. Defaults to non-robot collision geoms only.",
    )
    parser.add_argument(
        "--dump-orcalab-all-geoms",
        action="store_true",
        help="With --dump-orcalab-geoms, include visual geoms where contype/conaffinity are both zero.",
    )
    parser.add_argument(
        "--dump-orcalab-robot-geoms",
        action="store_true",
        help="With --dump-orcalab-geoms, include robot actor geoms instead of filtering them out.",
    )
    parser.add_argument(
        "--dump-orcalab-geoms-only",
        action="store_true",
        help="Exit after publishing/loading OrcaLab and dumping geoms.",
    )
    parser.add_argument("--no-publish", action="store_true", help="Reuse the already-published OrcaLab batch scene")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false", help="Run as fast as possible")
    args = parser.parse_args()
    configure_torch_backends()

    env = _make_play_env(args)
    _print_play_runtime(env)
    policy = _load_policy(env, args)
    try:
        if args.orcalab:
            _play_orcalab(env, policy, args)
        else:
            _play_headless(env, policy, args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
