import asyncio
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from orcalab_rslrl.orcalab_batch_render import (  # noqa: E402
    OrcaLabBatchRenderer,
    apply_orcalab_day_options,
    build_batch_layout,
    collect_scene_geoms,
    discover_agent_names,
    resolve_spawn_center,
    resolve_terrain_position,
)
from orcalab_rslrl.scene_options import (  # noqa: E402
    ORCA_TRAIN_SCENE_OPTIONS,
    UNITREE_ORCA_SCENE_OPTIONS,
    assert_flat_ground_options,
    assert_scene_options,
    patch_scene_xml_options,
    scene_xml_contract,
)

_AGENTS = ("g1_000", "g1_001", "g1_002")
_JOINTS = ("left_hip_pitch_joint", "right_knee_joint")


def _combined_model() -> mujoco.MjModel:
    """Mimic the OrcaLab combined scene: prefixed free + hinge joints per actor."""
    bodies = []
    for index, agent in enumerate(_AGENTS):
        bodies.append(f"""
        <body name="{agent}_pelvis" pos="{2.5 * index} 1.0 0.8">
          <freejoint name="{agent}_floating_base_joint"/>
          <geom type="sphere" size="0.05" mass="1"/>
          <body name="{agent}_leg">
            <joint name="{agent}_left_hip_pitch_joint" axis="0 1 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.5"/>
            <body name="{agent}_shin">
              <joint name="{agent}_right_knee_joint" axis="0 1 0"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.5"/>
            </body>
          </body>
        </body>""")
    xml = f"<mujoco><worldbody>{''.join(bodies)}</worldbody></mujoco>"
    return mujoco.MjModel.from_xml_string(xml)


def test_batch_layout_scatter_matches_agents():
    model = _combined_model()
    local_qpos_addr = {"left_hip_pitch_joint": 7, "right_knee_joint": 8}
    layout = build_batch_layout(model, _AGENTS, local_qpos_addr)
    assert layout.nq_combined == model.nq == len(_AGENTS) * 9

    batch = np.zeros((len(_AGENTS), 9))
    for index in range(len(_AGENTS)):
        batch[index, 0:3] = (0.1 * index, 0.0, 0.75)  # local root pos
        batch[index, 3:7] = (1.0, 0.0, 0.0, 0.0)
        batch[index, 7] = 0.5 + index  # hip
        batch[index, 8] = -0.5 - index  # knee

    qpos = np.array(model.qpos0, copy=True)
    values = batch[:, layout.src_index]
    values[:, 0:3] += layout.root_offset
    qpos[layout.dst_index.reshape(-1)] = values.reshape(-1)

    for index, agent in enumerate(_AGENTS):
        root_adr = model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{agent}_floating_base_joint"
            )
        ]
        # x is local x plus the actor's grid offset; z stays local (offset z zeroed).
        assert qpos[root_adr + 0] == pytest.approx(0.1 * index + 2.5 * index)
        assert qpos[root_adr + 1] == pytest.approx(1.0)
        assert qpos[root_adr + 2] == pytest.approx(0.75)
        hip_adr = model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{agent}_left_hip_pitch_joint"
            )
        ]
        knee_adr = model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{agent}_right_knee_joint"
            )
        ]
        assert qpos[hip_adr] == pytest.approx(0.5 + index)
        assert qpos[knee_adr] == pytest.approx(-0.5 - index)


def test_batch_layout_missing_agent_raises():
    model = _combined_model()
    with pytest.raises(RuntimeError, match="free joint"):
        build_batch_layout(model, ("g1_037",), {"left_hip_pitch_joint": 7})


def test_batch_layout_rejects_ambiguous_agent_freejoint():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="g1_000_robot"><freejoint name="g1_000_floating_base_joint"/>
            <geom type="sphere" size="0.1" mass="1"/>
            <body><joint name="g1_000_left_hip_pitch_joint" type="hinge"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
          </body>
          <body name="g1_000_extra"><freejoint name="g1_000_camera_freejoint"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body>
        </worldbody></mujoco>
        """
    )
    with pytest.raises(RuntimeError, match="ambiguous free joints"):
        build_batch_layout(model, ("g1_000",), {"left_hip_pitch_joint": 7})


def test_discover_agent_names_uses_complete_joint_suffix_set():
    model = _combined_model()
    discovered = discover_agent_names(model, _JOINTS, expected_count=3)
    assert discovered == list(_AGENTS)
    with pytest.raises(RuntimeError, match="exactly 1"):
        discover_agent_names(model, _JOINTS, expected_count=1)


def test_bounded_center_positions_stay_inside_range():
    renderer = object.__new__(OrcaLabBatchRenderer)
    renderer.num_envs = 50
    renderer.spacing = 0.8
    renderer.spawn_range = 1.2
    renderer.spawn_height = 0.0
    renderer.spawn_center = np.zeros(3)

    positions = renderer._grid_positions()
    assert positions.shape == (50, 3)
    assert np.abs(positions[:, :2]).max() <= 1.2
    # Center-first ordering: first actor starts at the center.
    assert positions[0, :2].tolist() == pytest.approx([0.0, 0.0])


def test_terrain_prefab_defaults_do_not_guess_authored_offset():
    asset_path = (
        "assets/e071469a36d3c8aa/default_project/prefabs/terrain_stair_mid_flat_usda"
    )
    position = resolve_terrain_position(asset_path, None)
    assert position.tolist() == pytest.approx([0.0, 0.0, 0.0])

    center = resolve_spawn_center(asset_path, None)
    assert center.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert resolve_terrain_position("any", [1.0, 2.0, 3.0]).tolist() == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert resolve_spawn_center("any", [4.0, 5.0, 6.0]).tolist() == pytest.approx(
        [4.0, 5.0, 6.0]
    )


def test_render_root_offset_is_applied_every_frame():
    model = _combined_model()
    local_qpos_addr = {"left_hip_pitch_joint": 7, "right_knee_joint": 8}
    renderer = object.__new__(OrcaLabBatchRenderer)
    renderer.layout = build_batch_layout(model, _AGENTS, local_qpos_addr)
    renderer._qpos = np.array(model.qpos0, copy=True)
    renderer.root_xy_scale = 1.0
    renderer.render_root_offset = np.array([0.2, -0.3, 0.4])

    batch = np.zeros((len(_AGENTS), 9))
    batch[:, 2] = 0.75
    batch[:, 3] = 1.0
    values = batch[:, renderer.layout.src_index]
    values[:, 0:2] *= renderer.root_xy_scale
    values[:, 0:3] += renderer.layout.root_offset
    values[:, 0:3] += renderer.render_root_offset
    renderer._qpos[renderer.layout.dst_index.reshape(-1)] = values.reshape(-1)

    root_adr = model.jnt_qposadr[
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "g1_000_floating_base_joint"
        )
    ]
    assert renderer._qpos[root_adr : root_adr + 3].tolist() == pytest.approx(
        [0.2, 0.7, 1.15]
    )


def test_collect_scene_geoms_filters_non_robot_collisions():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="terrain">
              <geom name="terrain_collision" type="box" pos="1 2 0.1" size="0.5 0.5 0.1" contype="1" conaffinity="1"/>
              <geom name="terrain_visual" type="box" pos="1 2 0.3" size="0.5 0.5 0.1" contype="0" conaffinity="0"/>
            </body>
            <body name="g1_000_pelvis">
              <geom name="g1_000_body_collision" type="sphere" size="0.1" contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    geoms = collect_scene_geoms(model, ("g1_000",))
    assert [geom["name"] for geom in geoms] == ["terrain_collision"]
    assert geoms[0]["type"] == "box"
    assert geoms[0]["world_pos"] == pytest.approx([1.0, 2.0, 0.1])
    assert len(geoms[0]["friction"]) == 3
    assert len(geoms[0]["solref"]) == 2
    assert len(geoms[0]["solimp"]) == 5

    all_non_robot = collect_scene_geoms(model, ("g1_000",), collisions_only=False)
    assert {geom["name"] for geom in all_non_robot} == {
        "terrain_collision",
        "terrain_visual",
    }


def test_orcalab_day_options_disable_air_resistance():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.0166666" density="1.225" viscosity="1.8e-05" wind="1 2 3"/>
          <worldbody><geom type="sphere" size="0.1" mass="1"/></worldbody>
        </mujoco>
        """
    )
    apply_orcalab_day_options(model, timestep=0.001, disable_air_resistance=True)
    assert math.isclose(model.opt.timestep, 0.001)
    assert model.opt.density == 0.0
    assert model.opt.viscosity == 0.0
    assert model.opt.wind.tolist() == [0.0, 0.0, 0.0]
    options = UNITREE_ORCA_SCENE_OPTIONS
    assert int(model.opt.integrator) == options.integrator
    assert model.opt.gravity.tolist() == pytest.approx(options.gravity)
    assert model.opt.iterations == options.iterations
    assert model.opt.ls_iterations == options.ls_iterations
    assert model.opt.noslip_iterations == options.noslip_iterations
    assert model.opt.ccd_iterations == options.ccd_iterations
    assert model.opt.sdf_initpoints == options.sdf_initpoints
    assert model.opt.sdf_iterations == options.sdf_iterations
    assert model.opt.tolerance == pytest.approx(options.tolerance)
    assert model.opt.ls_tolerance == pytest.approx(options.ls_tolerance)
    assert model.opt.noslip_tolerance == pytest.approx(options.noslip_tolerance)
    assert model.opt.ccd_tolerance == pytest.approx(options.ccd_tolerance)


def test_manual_xml_override_applies_Orca_train_profile(tmp_path):
    source = tmp_path / "downloaded.xml"
    source.write_text(
        """
        <mujoco>
          <option timestep="0.001" integrator="Euler" density="1.225" viscosity="1.8e-05"/>
          <worldbody><geom name="ground" type="plane" size="0 0 0.01"/></worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "aligned.xml"
    patch_scene_xml_options(
        source,
        output,
        profile="orca-train",
        align_air_resistance=True,
    )
    source_contract = scene_xml_contract(source)
    output_contract = scene_xml_contract(output)
    assert source_contract["sha256"] != output_contract["sha256"]
    assert source_contract["option_attributes"]["density"] == "1.225"
    assert output_contract["option_attributes"]["density"] == "0"
    model = mujoco.MjModel.from_xml_path(str(output))
    snapshot = assert_scene_options(model, profile="orca-train")
    options = ORCA_TRAIN_SCENE_OPTIONS
    assert snapshot["timestep"] == pytest.approx(options.timestep)
    assert snapshot["integrator"] == options.integrator
    assert snapshot["iterations"] == options.iterations
    assert snapshot["ls_iterations"] == options.ls_iterations
    assert snapshot["ccd_iterations"] == options.ccd_iterations
    assert snapshot["density"] == 0.0
    assert snapshot["viscosity"] == 0.0
    plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert model.geom_friction[plane_id].tolist() == pytest.approx([1.0, 0.005, 0.0001])
    assert model.geom_solref[plane_id].tolist() == pytest.approx([0.02, 1.0])
    assert model.geom_solimp[plane_id].tolist() == pytest.approx(
        [0.9, 0.95, 0.001, 0.5, 2.0]
    )
    assert assert_flat_ground_options(model)[0]["name"] == "ground"


def test_flat_ground_options_accept_orcalab_float32_round_trip():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="0 0 0.01"
                  friction="1 0.005 0.0001" solref="0.02 1"
                  solimp="0.9 0.95 0.001 0.5 2" condim="3"/>
          </worldbody>
        </mujoco>
        """
    )
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    model.geom_friction[ground_id] = np.asarray([1.0, 0.005, 0.0001], dtype=np.float32)
    model.geom_solref[ground_id] = np.asarray([0.02, 1.0], dtype=np.float32)
    model.geom_solimp[ground_id] = np.asarray(
        [0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float32
    )

    assert assert_flat_ground_options(model)[0]["name"] == "ground"


def test_flat_ground_options_still_reject_contact_changes():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="0 0 0.01"
                  friction="1 0.005 0.0001" solref="0.02 1"
                  solimp="0.9 0.95 0.001 0.5 2" condim="3"/>
          </worldbody>
        </mujoco>
        """
    )
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    model.geom_solimp[ground_id, 0] = 0.89

    with pytest.raises(RuntimeError, match="ground.solimp"):
        assert_flat_ground_options(model)


def test_manual_xml_override_keeps_downloaded_asset_paths_resolvable(tmp_path):
    source_dir = tmp_path / "orca_download"
    source_dir.mkdir()
    source = source_dir / "out.xml"
    source.write_text(
        """
        <mujoco>
          <compiler meshdir="meshes"/>
          <include file="parts/extra.xml"/>
          <asset>
            <mesh name="prop" file="prop.obj"/>
            <texture name="albedo" type="2d" file="textures/albedo.png"/>
          </asset>
          <worldbody/>
        </mujoco>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "aligned.xml"
    patch_scene_xml_options(source, output, profile="orca-train")

    import xml.etree.ElementTree as ET

    root = ET.parse(output).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.get("meshdir") == str((source_dir / "meshes").resolve())
    assert root.find("include").get("file") == str(
        (source_dir / "parts/extra.xml").resolve()
    )
    assert root.find("asset/mesh").get("file") == "prop.obj"
    assert root.find("asset/texture").get("file") == str(
        (source_dir / "textures/albedo.png").resolve()
    )


def test_remote_scene_alignment_supports_protobuf_without_apirate(monkeypatch):
    # OrcaGym initializes a rotating file logger during package import. Keep
    # this unit test hermetic when the installed package directory is read-only.
    import logging
    import logging.handlers

    monkeypatch.setattr(
        logging.handlers,
        "RotatingFileHandler",
        lambda *_args, **_kwargs: logging.NullHandler(),
    )
    from orca_gym.protos import mjc_message_pb2

    options = ORCA_TRAIN_SCENE_OPTIONS
    current = mjc_message_pb2.QueryOptConfigResponse(
        timestep=0.001,
        impratio=1.0,
        tolerance=1.0e-8,
        ls_tolerance=1.0e-2,
        noslip_tolerance=1.0e-6,
        ccd_tolerance=1.0e-6,
        gravity=[0.0, 0.0, -9.81],
        wind=[0.0, 0.0, 0.0],
        magnetic=[0.0, -0.5, 0.0],
        density=1.225,
        viscosity=1.8e-5,
        o_margin=0.0,
        o_solref=[0.02, 1.0],
        o_solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
        o_friction=[1.0, 0.005, 0.0001],
        integrator=0,
        cone=0,
        jacobian=0,
        solver=2,
        iterations=100,
        ls_iterations=50,
        noslip_iterations=10,
        ccd_iterations=35,
        disableflags=0,
        enableflags=0,
        disableactuator=0,
        sdf_initpoints=40,
        sdf_iterations=50,
    )
    aligned = mjc_message_pb2.QueryOptConfigResponse(
        timestep=options.timestep,
        integrator=options.integrator,
        gravity=options.gravity,
        iterations=options.iterations,
        ls_iterations=options.ls_iterations,
        noslip_iterations=options.noslip_iterations,
        ccd_iterations=options.ccd_iterations,
        sdf_initpoints=options.sdf_initpoints,
        sdf_iterations=options.sdf_iterations,
        tolerance=options.tolerance,
        ls_tolerance=options.ls_tolerance,
        noslip_tolerance=options.noslip_tolerance,
        ccd_tolerance=options.ccd_tolerance,
        density=options.density,
        viscosity=options.viscosity,
        wind=options.wind,
    )

    class _Stub:
        def __init__(self):
            self.queries = iter((current, aligned))
            self.request = None

        async def QueryOptConfig(self, _request):
            return next(self.queries)

        async def SetOptConfig(self, request):
            self.request = request
            return mjc_message_pb2.SetOptConfigResponse()

    renderer = object.__new__(OrcaLabBatchRenderer)
    renderer.loop = asyncio.new_event_loop()
    renderer.stub = _Stub()
    renderer.scene_options = options
    renderer.scene_timestep = None
    renderer.scene_profile = "orca-train"
    renderer.disable_air_resistance = True
    renderer.strict_scene_options = True
    try:
        renderer._apply_remote_scene_options()
    finally:
        renderer.loop.close()

    assert renderer.remote_scene_options["verified"] is True
    assert renderer.stub.request is not None
    assert "apirate" not in renderer.stub.request.DESCRIPTOR.fields_by_name


def test_scene_anchor_preserves_authored_xy_yaw_and_local_height():
    yaw_90 = math.sqrt(0.5)
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco><worldbody>
          <body name="go2_000_base" pos="2 3 0.45" quat="{yaw_90} 0 0 {yaw_90}">
            <freejoint name="go2_000__joint_0"/>
            <geom type="sphere" size="0.1" mass="1"/>
            <body><joint name="go2_000_FL_hip_joint" type="hinge"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
          </body>
        </worldbody></mujoco>
        """
    )
    renderer = object.__new__(OrcaLabBatchRenderer)
    renderer.num_envs = 1
    renderer.anchor_to_scene = True
    renderer.root_xy_scale = 1.0
    renderer.render_root_offset = np.zeros(3)
    renderer._source_root_reference = None
    renderer.layout = build_batch_layout(model, ("go2_000",), {"FL_hip_joint": 7})

    yaw_45 = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
    first = np.zeros((1, 8))
    first[0, :3] = [0.4, -0.2, 0.32]
    first[0, 3:7] = yaw_45
    first_pos, first_quat = renderer.map_root_pose(first)
    assert first_pos[0].tolist() == pytest.approx([2.0, 3.0, 0.32])
    assert abs(
        float(np.dot(first_quat[0], [yaw_90, 0.0, 0.0, yaw_90]))
    ) == pytest.approx(1.0)

    second = first.copy()
    second[0, 0] += 1.0
    second_pos, _second_quat = renderer.map_root_pose(second)
    # Scene yaw 90 - source yaw 45 rotates the relative +X motion by +45 deg.
    assert second_pos[0, :2].tolist() == pytest.approx(
        [2.0 + math.sqrt(0.5), 3.0 + math.sqrt(0.5)]
    )
    assert second_pos[0, 2] == pytest.approx(0.32)


def test_render_updates_only_robot_qpos_and_preserves_authored_prop_pose():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="go2_000_base" pos="1 2 0.45">
            <freejoint name="go2_000__joint_0"/>
            <geom type="sphere" size="0.1" mass="1"/>
            <body><joint name="go2_000_FL_hip_joint" type="hinge"/>
              <geom type="sphere" size="0.05" mass="0.1"/>
            </body>
          </body>
          <body name="barrel" pos="4 -2 0.6">
            <freejoint name="barrel_free"/>
            <geom type="cylinder" size="0.3 0.6" mass="5"/>
          </body>
        </worldbody></mujoco>
        """
    )
    prop_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "barrel_free")
    prop_qpos_addr = int(model.jnt_qposadr[prop_joint])
    authored_prop_pose = np.array(
        model.qpos0[prop_qpos_addr : prop_qpos_addr + 7], copy=True
    )

    class _ImmediateLoop:
        def run_until_complete(self, value):
            return value

    class _Gym:
        def update_local_env(self, qpos, sim_time):
            self.qpos = np.array(qpos, copy=True)
            self.sim_time = sim_time
            return None

    renderer = object.__new__(OrcaLabBatchRenderer)
    renderer.num_envs = 1
    renderer.layout = build_batch_layout(model, ("go2_000",), {"FL_hip_joint": 7})
    renderer._qpos = np.array(model.qpos0, copy=True)
    renderer.anchor_to_scene = True
    renderer.root_xy_scale = 1.0
    renderer.render_root_offset = np.zeros(3)
    renderer._source_root_reference = None
    renderer.loop = _ImmediateLoop()
    renderer.gym = _Gym()

    local_qpos = np.zeros((1, 8))
    local_qpos[0, :7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0]
    local_qpos[0, 7] = 0.25
    renderer.render(local_qpos, 0.02)

    np.testing.assert_allclose(
        renderer.gym.qpos[prop_qpos_addr : prop_qpos_addr + 7],
        authored_prop_pose,
    )
