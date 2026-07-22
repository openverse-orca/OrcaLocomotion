import math

import mujoco
import pytest

from orcalab_rslrl.robots.g1_flat import (
    TERRAIN_STAIR_MID_FLAT,
    G1FlatRobot,
    build_g1_flat_model,
    g1_action_scale,
    resolve_g1_flat_mjcf,
)
from orcalab_rslrl.tasks import list_tasks
from orcalab_rslrl.tasks.g1_velocity import _root_reset_pose_range, build_g1_flat_env_cfg
from orcalab_rslrl.scene_options import (
    ORCA_TRAIN_SCENE_OPTIONS,
    UNITREE_ORCA_CONTROL_DT,
)


def test_g1_velocity_task_is_registered():
    names = {task.name for task in list_tasks()}
    assert "G1-Velocity-Flat" in names


def test_g1_flat_mjcf_is_bundled():
    path = resolve_g1_flat_mjcf()
    assert path.name == "g1.xml"
    assert path.parent.name == "xmls"
    assert path.parent.joinpath("assets", "pelvis.STL").exists()


@pytest.fixture(scope="module")
def model():
    return build_g1_flat_model(seed=0)


def test_g1_flat_model_actuators_and_sensors(model):
    assert model.nu == 29
    assert model.nkey >= 1  # home keyframe drives resets
    for sensor in (
        "imu_ang_vel", "imu_lin_vel", "root_angmom",
        "left_foot_ground_found", "right_foot_ground_force",
        "left_foot_pos", "right_foot_vel", "self_collision_found",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor) >= 0, sensor
    options = ORCA_TRAIN_SCENE_OPTIONS
    assert math.isclose(model.opt.timestep, options.timestep)
    assert int(model.opt.integrator) == options.integrator
    assert model.opt.gravity.tolist() == pytest.approx(options.gravity)
    assert model.opt.density == options.density
    assert model.opt.viscosity == options.viscosity
    assert model.opt.wind.tolist() == pytest.approx(options.wind)
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


def test_g1_flat_model_timestep_override():
    model = build_g1_flat_model(seed=0, timestep=0.001)
    assert math.isclose(model.opt.timestep, 0.001)


def test_g1_scene_ground_matches_unitree_orca_runtime(model):
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    assert model.geom_contype[floor_id] == 1
    assert model.geom_conaffinity[floor_id] == 1
    assert model.geom_condim[floor_id] == 3
    assert model.geom_priority[floor_id] == 0
    assert model.geom_friction[floor_id].tolist() == pytest.approx([1.0, 0.005, 0.0001])
    assert model.geom_solref[floor_id].tolist() == pytest.approx([0.02, 1.0])
    assert model.geom_solimp[floor_id].tolist() == pytest.approx([0.9, 0.95, 0.001, 0.5, 2.0])


def test_g1_robot_dynamics_remain_Orca_native(model):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll_joint")
    dof_id = int(model.jnt_dofadr[joint_id])
    assert model.dof_armature[dof_id] == pytest.approx(0.025101925)
    assert model.dof_damping[dof_id] == pytest.approx(0.0)
    assert model.dof_frictionloss[dof_id] == pytest.approx(0.0)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_roll")
    assert model.actuator_forcerange[actuator_id].tolist() == pytest.approx([-139.0, 139.0])


def test_g1_train_foot_friction_remains_Orca_randomized(model):
    foot_friction = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(("left_foot", "right_foot")) and name.endswith("_collision"):
            foot_friction.append(float(model.geom_friction[geom_id][0]))

    assert len(foot_friction) == 14
    assert 0.3 <= foot_friction[0] <= 1.6
    assert foot_friction == pytest.approx([foot_friction[0]] * 14)


def test_g1_flat_model_stair_terrain_adds_collision_geoms():
    model = build_g1_flat_model(seed=0, terrain_kind=TERRAIN_STAIR_MID_FLAT)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_stair_00") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_stair_top_flat") >= 0


def test_g1_play_env_disables_startup_domain_randomization():
    cfg = build_g1_flat_env_cfg(G1FlatRobot(), play=True)
    assert "encoder_bias" not in cfg.events

    model = build_g1_flat_model(seed=0, foot_friction_range=None, base_com_offset_range=None)
    foot_friction = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(("left_foot", "right_foot")) and name.endswith("_collision"):
            foot_friction.append(model.geom_friction[geom_id][0])
    assert foot_friction
    assert set(round(value, 6) for value in foot_friction) == {0.6}


def test_g1_action_scale_matches_Orca_form():
    scales = g1_action_scale()
    assert len(scales) == 29
    assert all(scale > 0 for scale in scales)


def test_g1_flat_env_cfg_has_full_mdp():
    cfg = build_g1_flat_env_cfg(G1FlatRobot())
    assert set(cfg.rewards) == {
        "track_linear_velocity", "track_angular_velocity", "body_orientation_l2",
        "pose", "body_ang_vel", "angular_momentum", "is_terminated", "joint_acc_l2",
        "joint_pos_limits", "action_rate_l2", "foot_gait", "foot_clearance",
        "foot_slip", "soft_landing", "stand_still", "self_collisions",
    }
    assert set(cfg.terminations) == {"time_out", "fell_over"}
    assert cfg.terminations["time_out"].time_out
    assert {"reset_base", "reset_robot_joints", "push_robot", "encoder_bias"} <= set(cfg.events)
    assert cfg.commands["twist"].resampling_time_range == (3.0, 8.0)
    assert cfg.reward_scale_by_dt and cfg.action_offset and cfg.clip_actions is None
    assert cfg.observations["actor"].enable_corruption
    assert not cfg.observations["critic"].enable_corruption
    assert cfg.decimation == 4
    assert cfg.decimation * ORCA_TRAIN_SCENE_OPTIONS.timestep == pytest.approx(
        UNITREE_ORCA_CONTROL_DT
    )
    assert cfg.episode_length_steps == 1000
    assert cfg.episode_length_steps * UNITREE_ORCA_CONTROL_DT == pytest.approx(20.0)


def test_g1_timestep_override_preserves_control_period():
    cfg = build_g1_flat_env_cfg(G1FlatRobot(), physics_timestep=0.005)
    assert cfg.decimation == 4
    assert cfg.decimation * 0.005 == pytest.approx(UNITREE_ORCA_CONTROL_DT)


def test_g1_timestep_override_rejects_non_integral_decimation():
    with pytest.raises(ValueError, match="integer decimation"):
        build_g1_flat_env_cfg(G1FlatRobot(), physics_timestep=0.003)


def test_g1_flat_play_cfg_disables_training_only_pieces():
    cfg = build_g1_flat_env_cfg(G1FlatRobot(), play=True)
    assert "push_robot" not in cfg.events
    assert "command_curriculum" not in cfg.events
    assert "encoder_bias" not in cfg.events
    assert not cfg.observations["actor"].enable_corruption
    assert cfg.episode_length_steps > 10**8


def test_g1_play_reset_pose_is_fixed_for_regular_grid_rendering():
    assert _root_reset_pose_range(play=True) == {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    train_range = _root_reset_pose_range(play=False)
    assert train_range["x"] == (-0.5, 0.5)
    assert train_range["y"] == (-0.5, 0.5)
    assert train_range["yaw"] == (-3.14, 3.14)
