from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import mujoco

from orca_gym import OrcaGymLocal
from orca_gym.environment import OrcaGymLocalEnv

from .action_mapper import ActionMapperConfig, ResidualJointTargetActionMapper
from .curriculum import CommandConfig, FlatVelocityCommandSampler
from .debug_visualizer import make_debug_arrow_visualizers
from .math_utils import quat_mul_wxyz, quat_wxyz_to_rotmat, yaw_quat_wxyz
from .obs_builder import LocomotionObservationBuilder, LocomotionTaskState, ObservationConfig
from .randomization import DomainRandomizer, RandomizationConfig, RandomizationState
from .rendering import resolve_rendering
from .reward_manager import FlatVelocityReward, RewardConfig
from .terrain_runtime import TerrainRuntime
from .termination_manager import TerminationConfig, TerminationManager
from .mujoco_model_settings import apply_unitree_play_global_settings


@dataclass
class BatchedLocomotionStepResult:
    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    dones: np.ndarray
    time_outs: np.ndarray
    logs: list[dict[str, float]]
    episode_lengths: np.ndarray


@dataclass
class _AgentRuntime:
    agent_name: str
    base_joint_name: str
    leg_joint_names: list[str]
    actuator_names: list[str]
    contact_site_names: list[str]
    foot_sensor_names: list[str]
    base_contact_body_names: list[str]
    foot_body_names: list[str]
    base_qpos_offset: int
    base_qvel_offset: int
    leg_qpos_indices: np.ndarray
    leg_qvel_indices: np.ndarray
    actuator_ids: np.ndarray
    joint_limits: np.ndarray
    torque_limits: np.ndarray
    nominal_qpos: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    initial_base_qpos: np.ndarray
    action_mapper: ResidualJointTargetActionMapper
    obs_builder: LocomotionObservationBuilder
    reward_manager: FlatVelocityReward
    termination_manager: TerminationManager
    command_sampler: FlatVelocityCommandSampler
    randomizer: DomainRandomizer
    terrain_runtime: TerrainRuntime
    rng: np.random.Generator
    command_resample_steps: int
    last_action: np.ndarray
    previous_action: np.ndarray
    last_torque: np.ndarray
    command: np.ndarray
    randomization_state: RandomizationState
    action_delay_buffer: list[np.ndarray]
    push_interval_steps: int
    last_foot_pos_world: np.ndarray
    feet_air_time: np.ndarray
    robot_body_names: set[str]
    illegal_contact_body_names: set[str]
    baseline_base_mass: float | None = None
    baseline_base_ipos: np.ndarray | None = None
    baseline_base_inertia: np.ndarray | None = None
    randomized_base_body_name: str | None = None


class BatchedOrcaLocomotionTask(OrcaGymLocalEnv):
    """One OrcaGymLocalEnv that advances many robot agents in the same MuJoCo scene."""

    metadata = {"render_modes": ["human", "none"], "version": "0.1.0", "render_fps": 30}

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        orcagym_addr: str,
        agent_names: Sequence[str],
        robot_config: dict[str, Any],
        model_xml_path: str | None = None,
        render_mode: str = "none",
        headless: bool | None = None,
        env_id: str = "BatchedLocomotion-OrcaGym",
    ) -> None:
        if not agent_names:
            raise ValueError("BatchedOrcaLocomotionTask needs at least one agent name.")
        self.cfg = cfg
        self.robot_config = robot_config
        self.agent_names = list(agent_names)
        self.model_xml_path = model_xml_path
        sim_cfg = cfg["sim"]
        self._headless, self._render_mode = resolve_rendering(cfg, render_mode=render_mode, headless=headless)
        self._is_subenv = self._headless
        self._sync_render = bool(sim_cfg.get("sync_render", False))
        self.env_id = env_id
        self.decimation = int(sim_cfg.get("decimation", 4))
        self.control_dt = float(sim_cfg["time_step"]) * int(sim_cfg["frame_skip"]) * self.decimation
        self.max_episode_length = int(round(float(cfg["episode"]["length_s"]) / self.control_dt))
        self._base_seed = int(cfg.get("seed", 1))

        super().__init__(
            frame_skip=int(sim_cfg["frame_skip"]),
            orcagym_addr=orcagym_addr,
            agent_names=self.agent_names,
            time_step=float(sim_cfg["time_step"]),
            render_mode=self._render_mode,
            headless=self._headless,
        )

        if bool(sim_cfg.get("unitree_play_global_settings", False)):
            apply_unitree_play_global_settings(self.gym._mjModel)
            print(
                "[orca_rl.play] Applied unitree-orca MuJoCo global settings: "
                "integrator=Euler gravity=(0, 0, -9.81) density=0 viscosity=0 "
                "wind=(0, 0, 0) noslip_iterations=0 sdf_iterations=10"
            )

        self.nu = self.model.nu
        initial_ctrl = np.asarray(getattr(self.data, "ctrl", np.zeros(self.nu)), dtype=np.float64).reshape(-1)
        self.ctrl = initial_ctrl.copy() if initial_ctrl.size == self.nu else np.zeros(self.nu, dtype=np.float64)
        self._configure_passive_g1_hands()
        self._setup_global_randomization_targets()
        local_terrain_cfg = self.robot_config.get("local_terrain_cfg")
        self._shared_terrain_runtime = (
            TerrainRuntime.from_task_cfg(
                self.cfg,
                np.random.default_rng(self._base_seed),
                terrain_cfg=local_terrain_cfg,
            )
            if local_terrain_cfg is not None
            else None
        )
        self.agents = [self._setup_agent(index, name) for index, name in enumerate(self.agent_names)]
        self.num_envs = len(self.agents)
        self.num_actions = self.agents[0].action_mapper.num_actions
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._build_batch_arrays()
        self._report_control_dimensions()
        self._build_contact_maps()
        debug_cfg = cfg.get("debug_visualization", {})
        self._debug_arrow_visualizers = make_debug_arrow_visualizers(self, debug_cfg)
        self.reset_model()

    async def _load_model_xml(self) -> str:
        if self.model_xml_path:
            return self.model_xml_path
        return await super()._load_model_xml()

    def initialize_grpc(self) -> None:
        if getattr(self, "model_xml_path", None):
            self.channel = None
            self.stub = None
            self.gym = OrcaGymLocal(None)
            return
        super().initialize_grpc()

    def pause_simulation(self) -> None:
        if getattr(self, "model_xml_path", None):
            return
        super().pause_simulation()

    def set_time_step(self, time_step: float) -> None:
        if getattr(self, "model_xml_path", None):
            self.time_step = time_step
            self.realtime_step = time_step * self.frame_skip
            self.gym.set_time_step(time_step)
            return
        super().set_time_step(time_step)

    def close(self) -> None:
        if getattr(self, "model_xml_path", None):
            return
        super().close()

    def render(self) -> None:
        for visualizer in getattr(self, "_debug_arrow_visualizers", []):
            visualizer.update()
        if getattr(self, "model_xml_path", None):
            return
        super().render()

    def debug_visualization_report(self) -> dict[str, Any]:
        visualizers = getattr(self, "_debug_arrow_visualizers", [])
        if not visualizers:
            return {"debug_arrows": []}
        return {"debug_arrows": [visualizer.describe() for visualizer in visualizers]}

    def set_manual_commands(self, commands: np.ndarray) -> None:
        commands = np.asarray(commands, dtype=np.float64).reshape(self.num_envs, 3)
        for index, agent in enumerate(self.agents):
            agent.command = commands[index].copy()
        self._manual_command_override = True

    def clear_manual_commands(self) -> None:
        self._manual_command_override = False

    def get_observations(self, noisy: bool = True) -> dict[str, np.ndarray]:
        foot_positions = self._query_all_foot_positions()
        contact_state = self._query_batched_contacts()
        states = [
            self._read_state(
                index,
                update_air_time=False,
                foot_contacts=contact_state["foot_contacts"][index],
                foot_pos=foot_positions[index],
            )
            for index in range(self.num_envs)
        ]
        return self._stack_observations(states, noisy)

    def step(self, actions: np.ndarray) -> BatchedLocomotionStepResult:
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, self.num_actions)
        self._prepare_actions(actions)
        self._resample_commands()
        torque = self._compute_torques()

        self.prepare_control_buffer()
        self.ctrl[self._flat_actuator_ids] = torque.reshape(-1)
        for _ in range(self.decimation):
            self.set_ctrl(self.ctrl)
            self.mj_step(nstep=self.frame_skip)
        self.update_data()
        if self._render_mode == "human":
            self.render()

        self.episode_lengths += 1
        self._maybe_apply_push_disturbances()
        contact_state = self._query_batched_contacts()
        foot_positions = self._query_all_foot_positions()
        states = [
            self._read_state(
                index,
                update_air_time=True,
                foot_contacts=contact_state["foot_contacts"][index],
                foot_pos=foot_positions[index],
            )
            for index in range(self.num_envs)
        ]

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        logs: list[dict[str, float]] = []
        for index, (agent, state) in enumerate(zip(self.agents, states)):
            terminated, termination_log = agent.termination_manager.check(
                state,
                bool(contact_state["base_contacts"][index]),
                bool(contact_state["illegal_contacts"][index]),
            )
            time_out = self.episode_lengths[index] >= self.max_episode_length
            reward, reward_log = agent.reward_manager.compute(
                state=state,
                action=agent.last_action,
                previous_action=agent.previous_action,
                terminated=terminated,
            )
            done = bool(terminated or time_out)
            rewards[index] = float(reward)
            dones[index] = done
            time_outs[index] = bool(time_out and not terminated)
            log = {**reward_log, **termination_log}
            log["/episode/length"] = float(self.episode_lengths[index])
            log["/episode/reward"] = float(reward)
            logs.append(log)

        done_indices = np.flatnonzero(dones)
        if done_indices.size:
            observations, _ = self.reset_model(done_indices)
        else:
            observations = self._stack_observations(states, noisy=True)
        return BatchedLocomotionStepResult(
            observations=observations,
            rewards=rewards,
            dones=dones,
            time_outs=time_outs,
            logs=logs,
            episode_lengths=self.episode_lengths.copy(),
        )

    def reset_model(
        self,
        env_indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        indices = (
            np.arange(self.num_envs, dtype=np.int64)
            if env_indices is None
            else np.asarray(env_indices, dtype=np.int64)
        )
        if indices.size == 0:
            return self.get_observations(noisy=True), {}

        joint_qpos: dict[str, np.ndarray] = {}
        joint_qvel: dict[str, np.ndarray] = {}
        for index in indices:
            agent = self.agents[int(index)]
            self.episode_lengths[index] = 0
            agent.command = agent.command_sampler.sample()
            agent.randomization_state = agent.randomizer.sample()
            if self.cfg.get("randomization", {}).get("terrain") == "rough" and not agent.terrain_runtime.physics_enabled:
                agent.terrain_runtime.resample()
            self._apply_agent_randomization(agent, int(index))
            self._reset_action_delay_buffer(agent)
            agent.last_action.fill(0.0)
            agent.previous_action.fill(0.0)
            agent.last_torque.fill(0.0)

            base_qpos = self._sample_base_qpos(agent)
            joint_qpos[agent.base_joint_name] = base_qpos
            joint_qvel[agent.base_joint_name] = np.zeros(6, dtype=np.float64)
            q_noise = float(self.cfg.get("reset", {}).get("joint_noise", 0.0))
            for joint_index, joint_name in enumerate(agent.leg_joint_names):
                value = agent.nominal_qpos[joint_index]
                if q_noise > 0.0:
                    value += float(agent.rng.uniform(-q_noise, q_noise))
                joint_qpos[joint_name] = np.array([value], dtype=np.float64)
                joint_qvel[joint_name] = np.zeros(1, dtype=np.float64)

        self.set_joint_qpos(joint_qpos)
        self.set_joint_qvel(joint_qvel)
        self._reset_passive_g1_hands()
        self._apply_global_randomization(self.agents[int(indices[0])].randomization_state)
        self.prepare_control_buffer()
        self.set_ctrl(self.ctrl)
        self.mj_forward()
        self.update_data()
        if self._render_mode == "human":
            self.render()

        foot_positions = self._query_all_foot_positions()
        for index in indices:
            agent = self.agents[int(index)]
            agent.last_foot_pos_world = foot_positions[int(index)]
            agent.feet_air_time.fill(0.0)
        return self.get_observations(noisy=True), {}

    def _setup_agent(self, agent_index: int, agent_name: str) -> _AgentRuntime:
        robot_label = str(self.cfg.get("robot", "robot")).upper()
        base_joint_name = self.joint(self.robot_config["base_joint_name"], agent_index)
        leg_joint_names = [self.joint(name, agent_index) for name in self.robot_config["leg_joint_names"]]
        actuator_names = [self.actuator(name, agent_index) for name in self.robot_config["actuator_names"]]
        requested_contact_sites = [
            self.site(name, agent_index) for name in self.robot_config.get("contact_site_names", [])
        ]
        foot_sensor_names = [
            self.sensor(name, agent_index) for name in self.robot_config.get("sensor_foot_touch_names", [])
        ]
        base_contact_body_names = [
            self.body(name, agent_index) for name in self.robot_config.get("base_contact_body_names", [])
        ]
        foot_body_names = [self.body(name, agent_index) for name in self.robot_config.get("foot_body_names", [])]

        joint_dict = self.model.get_joint_dict()
        actuator_dict = self.model.get_actuator_dict()
        site_dict = self.model.get_site_dict() if hasattr(self.model, "get_site_dict") else {}
        sensor_dict = getattr(self.model, "_sensor_dict", {})
        missing_joints = [name for name in [base_joint_name, *leg_joint_names] if name not in joint_dict]
        missing_actuators = [name for name in actuator_names if name not in actuator_dict]
        if missing_joints or missing_actuators:
            raise ValueError(
                f"{robot_label} binding incomplete for {agent_name}. missing_joints={missing_joints}, "
                f"missing_actuators={missing_actuators}"
            )

        contact_site_names = [name for name in requested_contact_sites if name in site_dict]
        if len(contact_site_names) != len(requested_contact_sites):
            contact_site_names = []
        if not contact_site_names and not foot_body_names:
            raise ValueError(f"{robot_label} binding incomplete for {agent_name}. No foot tracking bodies/sites.")
        foot_sensor_names = [name for name in foot_sensor_names if name in sensor_dict]

        base_qpos_offset, base_qvel_offset, _ = [
            int(values[0]) for values in self.query_joint_offsets([base_joint_name])
        ]
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(leg_joint_names)
        qpos_lengths, qvel_lengths, _ = self.query_joint_lengths(leg_joint_names)
        if any(int(length) != 1 for length in qpos_lengths) or any(int(length) != 1 for length in qvel_lengths):
            raise ValueError(f"{robot_label} controlled joints are expected to be one-DoF joints.")

        leg_qpos_indices = np.array([int(offset) for offset in qpos_offsets], dtype=np.int64)
        leg_qvel_indices = np.array([int(offset) for offset in qvel_offsets], dtype=np.int64)
        actuator_ids = np.array([int(actuator_dict[name]["ActuatorId"]) for name in actuator_names], dtype=np.int64)
        configured_joint_limits = self.robot_config.get("joint_limits")
        if configured_joint_limits is not None:
            joint_limits = np.asarray(configured_joint_limits, dtype=np.float64).reshape(-1, 2)
        else:
            joint_limits = np.array([joint_dict[name]["Range"] for name in leg_joint_names], dtype=np.float64)
        effort_limits = self.robot_config.get("motor_effort_limit_list")
        if effort_limits is not None:
            effort = np.asarray(effort_limits, dtype=np.float64).reshape(-1)
            torque_limits = np.stack([-effort, effort], axis=1)
        else:
            torque_limits = np.array(
                [actuator_dict[name].get("CtrlRange", [-1.0, 1.0]) for name in actuator_names],
                dtype=np.float64,
            )
        nominal_qpos = np.array(
            [self.robot_config["neutral_joint_angles"][name] for name in self.robot_config["leg_joint_names"]],
            dtype=np.float64,
        )
        kp = np.array(self.robot_config["kps"], dtype=np.float64)
        kd = np.array(self.robot_config["kds"], dtype=np.float64)
        control_cfg = self.cfg.get("control", {})
        action_mapper = ResidualJointTargetActionMapper(
            nominal_qpos=nominal_qpos,
            joint_limits=joint_limits,
            torque_limits=torque_limits,
            kp=kp,
            kd=kd,
            cfg=ActionMapperConfig(
                safety_scale=float(control_cfg.get("safety_scale", 0.85)),
                max_delta=control_cfg.get("max_delta"),
                action_clip=float(control_cfg.get("action_clip", 1.0)),
            ),
        )
        rng = np.random.default_rng(self._base_seed + agent_index)
        obs_builder = LocomotionObservationBuilder(
            nominal_qpos=nominal_qpos,
            cfg=ObservationConfig(**self.cfg.get("observations", {})),
            rng=rng,
        )
        reward_manager = FlatVelocityReward(
            joint_limits=joint_limits,
            nominal_qpos=nominal_qpos,
            cfg=RewardConfig(**self.cfg.get("rewards", {})),
        )
        command_sampler = FlatVelocityCommandSampler(CommandConfig(**self.cfg.get("commands", {})), rng)
        randomizer = DomainRandomizer(RandomizationConfig(**self.cfg.get("randomization", {})), rng)
        terrain_runtime = self._shared_terrain_runtime or TerrainRuntime.from_task_cfg(self.cfg, rng)
        command_resample_steps = max(
            1,
            int(round(float(self.cfg.get("commands", {}).get("resample_time_s", 4.0)) / self.control_dt)),
        )
        num_feet = len(contact_site_names) if contact_site_names else len(foot_body_names)
        agent = _AgentRuntime(
            agent_name=agent_name,
            base_joint_name=base_joint_name,
            leg_joint_names=leg_joint_names,
            actuator_names=actuator_names,
            contact_site_names=contact_site_names,
            foot_sensor_names=foot_sensor_names,
            base_contact_body_names=base_contact_body_names,
            foot_body_names=foot_body_names,
            base_qpos_offset=base_qpos_offset,
            base_qvel_offset=base_qvel_offset,
            leg_qpos_indices=leg_qpos_indices,
            leg_qvel_indices=leg_qvel_indices,
            actuator_ids=actuator_ids,
            joint_limits=joint_limits,
            torque_limits=torque_limits,
            nominal_qpos=nominal_qpos,
            kp=kp,
            kd=kd,
            initial_base_qpos=self.data.qpos[base_qpos_offset : base_qpos_offset + 7].copy(),
            action_mapper=action_mapper,
            obs_builder=obs_builder,
            reward_manager=reward_manager,
            termination_manager=TerminationManager(TerminationConfig(**self.cfg.get("termination", {}))),
            command_sampler=command_sampler,
            randomizer=randomizer,
            terrain_runtime=terrain_runtime,
            rng=rng,
            command_resample_steps=command_resample_steps,
            last_action=np.zeros(action_mapper.num_actions, dtype=np.float64),
            previous_action=np.zeros(action_mapper.num_actions, dtype=np.float64),
            last_torque=np.zeros(action_mapper.num_actions, dtype=np.float64),
            command=command_sampler.sample(),
            randomization_state=RandomizationState(),
            action_delay_buffer=[np.zeros(action_mapper.num_actions, dtype=np.float64)],
            push_interval_steps=self._push_interval_steps(),
            last_foot_pos_world=np.zeros((num_feet, 3), dtype=np.float64),
            feet_air_time=np.zeros(num_feet, dtype=np.float64),
            robot_body_names=set(),
            illegal_contact_body_names=set(),
        )
        self._setup_agent_randomization_targets(agent)
        self._setup_contact_classification(agent)
        return agent

    def _build_batch_arrays(self) -> None:
        self._base_qpos_offsets = np.array([agent.base_qpos_offset for agent in self.agents], dtype=np.int64)
        self._base_qvel_offsets = np.array([agent.base_qvel_offset for agent in self.agents], dtype=np.int64)
        self._base_qpos_indices = self._base_qpos_offsets[:, None] + np.arange(7, dtype=np.int64)
        self._base_qvel_indices = self._base_qvel_offsets[:, None] + np.arange(6, dtype=np.int64)
        self._leg_qpos_indices = np.stack([agent.leg_qpos_indices for agent in self.agents], axis=0)
        self._leg_qvel_indices = np.stack([agent.leg_qvel_indices for agent in self.agents], axis=0)
        self._actuator_ids = np.stack([agent.actuator_ids for agent in self.agents], axis=0)
        self._flat_actuator_ids = self._actuator_ids.reshape(-1)
        controlled_mask = np.zeros(self.nu, dtype=bool)
        controlled_mask[self._flat_actuator_ids] = True
        self._uncontrolled_actuator_ids = np.flatnonzero(~controlled_mask)
        self._nominal_qpos = np.stack([agent.nominal_qpos for agent in self.agents], axis=0)
        self._joint_limit_low = np.stack([agent.joint_limits[:, 0] for agent in self.agents], axis=0)
        self._joint_limit_high = np.stack([agent.joint_limits[:, 1] for agent in self.agents], axis=0)
        self._delta_low = np.stack([agent.action_mapper.delta_low for agent in self.agents], axis=0)
        self._delta_high = np.stack([agent.action_mapper.delta_high for agent in self.agents], axis=0)
        self._kp = np.stack([agent.kp for agent in self.agents], axis=0)
        self._kd = np.stack([agent.kd for agent in self.agents], axis=0)
        self._torque_low = np.stack([agent.torque_limits[:, 0] for agent in self.agents], axis=0)
        self._torque_high = np.stack([agent.torque_limits[:, 1] for agent in self.agents], axis=0)

    def prepare_control_buffer(self) -> np.ndarray:
        """Clear policy-owned channels while preserving all other actuator commands.

        A G1 + Dex3-1 model has 43 actuators, but locomotion owns only the 29
        body actuators.  Copying the live ctrl buffer first prevents reset and
        policy steps from overwriting the 14 hand channels.
        """

        live_ctrl = np.asarray(getattr(self.data, "ctrl", self.ctrl), dtype=np.float64).reshape(-1)
        if live_ctrl.size == self.nu:
            self.ctrl[:] = live_ctrl
        self.ctrl[self._flat_actuator_ids] = 0.0
        return self.ctrl

    def _configure_passive_g1_hands(self) -> None:
        """Disable Dex3 drives and stabilize its otherwise passive light links."""

        self._passive_hand_actuator_ids = np.zeros(0, dtype=np.int64)
        self._passive_hand_dof_ids = np.zeros(0, dtype=np.int64)
        self._passive_hand_qpos_ids = np.zeros(0, dtype=np.int64)
        self._passive_hand_target_qpos = np.zeros(0, dtype=np.float64)
        if str(self.robot_config.get("model_name", "")).lower() != "g1":
            return

        model = self.gym._mjModel
        data = self.gym._mjData
        agent_prefixes = tuple(f"{name}_" for name in self.agent_names)
        actuator_ids: list[int] = []
        dof_ids: list[int] = []
        qpos_ids: list[int] = []
        open_qpos: list[float] = []
        for actuator_id in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            if "_hand_" not in actuator_name or not actuator_name.startswith(agent_prefixes):
                continue
            if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            dof_id = int(model.jnt_dofadr[joint_id])
            qpos_id = int(model.jnt_qposadr[joint_id])
            actuator_ids.append(actuator_id)
            dof_ids.append(dof_id)
            qpos_ids.append(qpos_id)
            open_qpos.append(self._dex3_open_joint_position(model, joint_id))
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_biasprm[actuator_id, :] = 0.0
            open_target = open_qpos[-1]
            # Keep Dex3 fully open without adding 14 hand actions to the
            # locomotion ABI or leaving light finger links free to shake the
            # wrists.
            model.jnt_stiffness[joint_id] = max(float(model.jnt_stiffness[joint_id]), 10.0)
            if hasattr(model, "qpos_spring"):
                model.qpos_spring[qpos_id] = open_target
            model.dof_damping[dof_id] = max(float(model.dof_damping[dof_id]), 0.3)
            model.dof_armature[dof_id] = max(float(model.dof_armature[dof_id]), 0.002)
            model.dof_frictionloss[dof_id] = max(float(model.dof_frictionloss[dof_id]), 0.05)
            data.ctrl[actuator_id] = 0.0
            self.ctrl[actuator_id] = 0.0

        if not actuator_ids:
            return
        self._passive_hand_actuator_ids = np.asarray(actuator_ids, dtype=np.int64)
        self._passive_hand_dof_ids = np.asarray(dof_ids, dtype=np.int64)
        self._passive_hand_qpos_ids = np.asarray(qpos_ids, dtype=np.int64)
        self._passive_hand_target_qpos = np.asarray(open_qpos, dtype=np.float64)
        enabled_hand_contact_geoms = 0
        for geom_id in range(model.ngeom):
            body_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            ) or ""
            if "_hand_" not in body_name or not body_name.startswith(agent_prefixes):
                continue
            if int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
                enabled_hand_contact_geoms += 1
        self._reset_passive_g1_hands()
        mujoco.mj_setConst(model, data)
        print(
            "[orca_rl.control] G1 Dex3 passive mode: "
            f"disabled_actuators={len(actuator_ids)}, default_pose=fully_open, "
            f"contact_geoms={enabled_hand_contact_geoms}, spring>=10.0, "
            "damping>=0.3, armature>=0.002, frictionloss>=0.05"
        )

    @staticmethod
    def _dex3_open_joint_position(model: mujoco.MjModel, joint_id: int) -> float:
        low, high = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        return float(np.clip(0.0, low, high))

    def _reset_passive_g1_hands(self) -> None:
        if not getattr(self, "_passive_hand_qpos_ids", np.empty(0)).size:
            return
        data = self.gym._mjData
        data.qpos[self._passive_hand_qpos_ids] = self._passive_hand_target_qpos
        data.qvel[self._passive_hand_dof_ids] = 0.0
        data.ctrl[self._passive_hand_actuator_ids] = 0.0
        self.ctrl[self._passive_hand_actuator_ids] = 0.0

    def _report_control_dimensions(self) -> None:
        model_name = str(self.robot_config.get("model_name", "robot"))
        if model_name.lower() == "g1" and self.num_actions != 29:
            raise ValueError(f"G1 locomotion ABI requires 29 actions, got {self.num_actions}.")
        if self._uncontrolled_actuator_ids.size:
            print(
                f"[orca_rl.control] {model_name}: action_dim={self.num_actions}, model_nu={self.nu}, "
                f"preserved_uncontrolled_actuators={self._uncontrolled_actuator_ids.size}"
            )

    def _build_contact_maps(self) -> None:
        self._body_to_foot_entries: dict[str, list[tuple[int, int]]] = {}
        self._body_to_base_envs: dict[str, list[int]] = {}
        self._body_to_robot_envs: dict[str, set[int]] = {}
        self._body_to_illegal_envs: dict[str, list[int]] = {}
        for env_index, agent in enumerate(self.agents):
            for foot_index, body_name in enumerate(agent.foot_body_names):
                self._body_to_foot_entries.setdefault(body_name, []).append((env_index, foot_index))
            for body_name in agent.base_contact_body_names:
                self._body_to_base_envs.setdefault(body_name, []).append(env_index)
            for body_name in agent.robot_body_names:
                self._body_to_robot_envs.setdefault(body_name, set()).add(env_index)
            for body_name in agent.illegal_contact_body_names:
                self._body_to_illegal_envs.setdefault(body_name, []).append(env_index)

    def _prepare_actions(self, actions: np.ndarray) -> None:
        for index, agent in enumerate(self.agents):
            clip = float(agent.action_mapper.cfg.action_clip)
            clipped = np.clip(actions[index], -clip, clip)
            agent.previous_action = agent.last_action.copy()
            agent.last_action = self._delayed_action(agent, clipped)

    def _resample_commands(self) -> None:
        if getattr(self, "_manual_command_override", False):
            return
        for index, agent in enumerate(self.agents):
            if self.episode_lengths[index] % agent.command_resample_steps == 0:
                agent.command = agent.command_sampler.sample()

    def _compute_torques(self) -> np.ndarray:
        actions = np.stack([agent.last_action for agent in self.agents], axis=0)
        alpha = 0.5 * (actions + 1.0)
        target_qpos = self._nominal_qpos + self._delta_low + alpha * (self._delta_high - self._delta_low)
        target_qpos = np.clip(target_qpos, self._joint_limit_low, self._joint_limit_high)
        qpos = self.data.qpos[self._leg_qpos_indices]
        qvel = self.data.qvel[self._leg_qvel_indices]
        torque = self._kp * (target_qpos - qpos) - self._kd * qvel
        torque = np.clip(torque, self._torque_low, self._torque_high)
        for index, agent in enumerate(self.agents):
            agent.last_torque = torque[index].copy()
        return torque

    def _read_state(
        self,
        index: int,
        *,
        update_air_time: bool,
        foot_contacts: np.ndarray | None = None,
        foot_pos: np.ndarray | None = None,
    ) -> LocomotionTaskState:
        agent = self.agents[index]
        base_qpos = self.data.qpos[self._base_qpos_indices[index]].copy()
        base_qvel = self.data.qvel[self._base_qvel_indices[index]].copy()
        qpos = self.data.qpos[agent.leg_qpos_indices].copy()
        qvel = self.data.qvel[agent.leg_qvel_indices].copy()
        if foot_pos is None:
            foot_pos = self._query_foot_positions(index)
        foot_vel = (foot_pos - agent.last_foot_pos_world) / max(self.control_dt, 1e-6)
        agent.last_foot_pos_world = foot_pos.copy()
        if foot_contacts is None:
            foot_contacts = self._query_batched_contacts()["foot_contacts"][index]
        if foot_contacts.shape != agent.feet_air_time.shape:
            agent.feet_air_time = np.zeros_like(foot_contacts, dtype=np.float64)
        previous_air_time = agent.feet_air_time.copy()
        if update_air_time:
            agent.feet_air_time = np.where(foot_contacts > 0.5, 0.0, agent.feet_air_time + self.control_dt)
        first_foot_contact = np.logical_and(foot_contacts > 0.5, previous_air_time > 0.0).astype(np.float64)
        foot_ground_heights = np.array(
            [agent.terrain_runtime.height_at(float(pos[0]), float(pos[1])) for pos in foot_pos],
            dtype=np.float64,
        )
        return LocomotionTaskState(
            base_pos=base_qpos[:3],
            base_quat=base_qpos[3:7],
            base_lin_vel_world=base_qvel[:3],
            base_ang_vel_world=base_qvel[3:6],
            qpos=qpos,
            qvel=qvel,
            command=agent.command.copy(),
            last_action=agent.last_action.copy(),
            last_torque=agent.last_torque.copy(),
            foot_pos_world=foot_pos,
            foot_vel_world=foot_vel,
            foot_contacts=foot_contacts,
            foot_air_time=previous_air_time,
            first_foot_contact=first_foot_contact,
            foot_ground_heights=foot_ground_heights,
            friction_scale=agent.randomization_state.friction_scale,
            base_mass_delta=agent.randomization_state.base_mass_delta,
            domain_randomization=self._randomization_observation(agent),
            height_scan=self._query_height_scan(agent, base_qpos[:3], base_qpos[3:7]),
        )

    def _query_height_scan(self, agent: _AgentRuntime, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """Return a vertical height scan from the live MuJoCo scene."""

        scan_cfg = agent.terrain_runtime.scan_cfg
        if not scan_cfg.enabled:
            return np.zeros(0, dtype=np.float64)
        scale = float(scan_cfg.scale)
        fallback_scaled = agent.terrain_runtime.scan(base_pos, base_quat)
        model = getattr(getattr(self, "gym", None), "_mjModel", None)
        data = getattr(getattr(self, "gym", None), "_mjData", None)
        if model is None or data is None:
            return fallback_scaled
        fallback = fallback_scaled / scale if abs(scale) > 1.0e-12 else fallback_scaled

        samples = self._raycast_ground_truth_height_scan(
            model=model,
            data=data,
            agent=agent,
            base_pos=np.asarray(base_pos, dtype=np.float64),
            base_quat=np.asarray(base_quat, dtype=np.float64),
            fallback=fallback,
        )
        return samples * scale

    def _raycast_ground_truth_height_scan(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        agent: _AgentRuntime,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        scan_cfg = agent.terrain_runtime.scan_cfg
        xs = np.arange(-scan_cfg.size[0] / 2.0, scan_cfg.size[0] / 2.0 + 0.5 * scan_cfg.resolution, scan_cfg.resolution)
        ys = np.arange(-scan_cfg.size[1] / 2.0, scan_cfg.size[1] / 2.0 + 0.5 * scan_cfg.resolution, scan_cfg.resolution)
        if fallback.size != xs.size * ys.size:
            fallback = np.resize(fallback, xs.size * ys.size).astype(np.float64)

        rot = quat_wxyz_to_rotmat(base_quat)
        yaw = float(np.arctan2(float(rot[1, 0]), float(rot[0, 0])))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        geomgroup = _height_scan_geomgroup(model, self.cfg)
        bodyexclude = _agent_root_body_id(model, self.model, agent)
        robot_geom_ids = _robot_geom_ids(model, agent)
        saved_robot_groups = model.geom_group[robot_geom_ids].copy() if robot_geom_ids.size else None
        if robot_geom_ids.size:
            model.geom_group[robot_geom_ids] = 5
            geomgroup[5] = 0
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        heights = np.empty(xs.size * ys.size, dtype=np.float64)

        try:
            index = 0
            for y in ys:
                for x in xs:
                    offset = np.array(
                        [
                            cos_yaw * float(x) - sin_yaw * float(y),
                            sin_yaw * float(x) + cos_yaw * float(y),
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                    ray_start = np.array([base_pos[0] + offset[0], base_pos[1] + offset[1], base_pos[2]], dtype=np.float64)
                    geomid[0] = -1
                    distance = float(
                        mujoco.mj_ray(
                            model,
                            data,
                            ray_start,
                            ray_dir,
                            geomgroup,
                            1,
                            bodyexclude,
                            geomid,
                        )
                    )
                    if distance >= 0.0 and geomid[0] >= 0:
                        heights[index] = float(base_pos[2]) - float(ray_start[2] - distance)
                    else:
                        heights[index] = float(fallback[index])
                    index += 1
        finally:
            if robot_geom_ids.size and saved_robot_groups is not None:
                model.geom_group[robot_geom_ids] = saved_robot_groups
        return heights

    def _stack_observations(self, states: list[LocomotionTaskState], noisy: bool) -> dict[str, np.ndarray]:
        observations = [agent.obs_builder.build(state, noisy=noisy) for agent, state in zip(self.agents, states)]
        return {
            key: np.stack([obs[key] for obs in observations], axis=0).astype(np.float32)
            for key in observations[0].keys()
        }

    def _query_foot_positions(self, index: int) -> np.ndarray:
        return self._query_all_foot_positions()[index]

    def _query_all_foot_positions(self) -> list[np.ndarray]:
        if all(agent.contact_site_names for agent in self.agents):
            all_sites = [site for agent in self.agents for site in agent.contact_site_names]
            site_data = self.query_site_pos_and_quat(all_sites)
            return [
                np.array([site_data[name]["xpos"] for name in agent.contact_site_names], dtype=np.float64)
                for agent in self.agents
            ]

        unique_body_names = list(dict.fromkeys(body for agent in self.agents for body in agent.foot_body_names))
        xpos, _, _ = self.get_body_xpos_xmat_xquat(unique_body_names)
        body_positions = {
            body_name: np.asarray(xpos[index * 3 : (index + 1) * 3], dtype=np.float64)
            for index, body_name in enumerate(unique_body_names)
        }
        return [
            np.array([body_positions[name] for name in agent.foot_body_names], dtype=np.float64)
            for agent in self.agents
        ]

    def _query_batched_contacts(self) -> dict[str, np.ndarray]:
        foot_counts = [
            len(agent.contact_site_names) if agent.contact_site_names else len(agent.foot_body_names)
            for agent in self.agents
        ]
        max_feet = max(foot_counts) if foot_counts else 0
        foot_contacts = np.zeros((self.num_envs, max_feet), dtype=np.float64)
        base_contacts = np.zeros(self.num_envs, dtype=bool)
        illegal_contacts = np.zeros(self.num_envs, dtype=bool)

        if all(agent.foot_sensor_names for agent in self.agents):
            try:
                all_sensors = [sensor for agent in self.agents for sensor in agent.foot_sensor_names]
                sensor_data = self.query_sensor_data(all_sensors)
                threshold = float(self.cfg.get("contacts", {}).get("touch_threshold", 1.0))
                for env_index, agent in enumerate(self.agents):
                    values = [
                        float(np.asarray(sensor_data[name]).reshape(-1)[0])
                        for name in agent.foot_sensor_names
                    ]
                    foot_contacts[env_index, : len(values)] = (
                        np.asarray(values, dtype=np.float64) > threshold
                    ).astype(np.float64)
            except Exception:
                foot_contacts.fill(0.0)

        contacts = self.query_contact_simple()
        contact_force_threshold = float(self.cfg.get("contacts", {}).get("illegal_force_threshold", 0.0))
        for contact in contacts:
            body1 = self._geom_body_name(contact["Geom1"])
            body2 = self._geom_body_name(contact["Geom2"])
            self._mark_contact_body(foot_contacts, body1)
            self._mark_contact_body(foot_contacts, body2)
            for env_index in self._body_to_base_envs.get(body1, ()):
                base_contacts[env_index] = True
            for env_index in self._body_to_base_envs.get(body2, ()):
                base_contacts[env_index] = True
            if self._contact_force_exceeds(contact, contact_force_threshold):
                self._mark_illegal_contacts(illegal_contacts, body1, body2)
                self._mark_illegal_contacts(illegal_contacts, body2, body1)

        return {
            "foot_contacts": foot_contacts,
            "base_contacts": base_contacts,
            "illegal_contacts": illegal_contacts,
        }

    def _geom_body_name(self, geom_id: int) -> str:
        try:
            return self.model.get_geom_body_name(int(geom_id))
        except (KeyError, ValueError, TypeError):
            model = self._local_mujoco_model()
            if model is None:
                return ""
            body_id = int(model.geom_bodyid[int(geom_id)])
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

    def _mark_contact_body(self, foot_contacts: np.ndarray, body_name: str) -> None:
        for env_index, foot_index in self._body_to_foot_entries.get(body_name, ()):
            foot_contacts[env_index, foot_index] = 1.0

    def _mark_illegal_contacts(self, illegal_contacts: np.ndarray, illegal_body: str, other_body: str) -> None:
        other_robot_envs = self._body_to_robot_envs.get(other_body, set())
        for env_index in self._body_to_illegal_envs.get(illegal_body, ()):
            if env_index not in other_robot_envs:
                illegal_contacts[env_index] = True

    def _contact_force_exceeds(self, contact: dict[str, Any], threshold: float) -> bool:
        if threshold <= 0.0:
            return True
        contact_id = int(contact.get("ID", -1))
        if contact_id < 0:
            return True
        try:
            forces = self.query_contact_force([contact_id])
            force = np.asarray(forces[contact_id], dtype=np.float64).reshape(-1)
        except Exception:
            return True
        return bool(np.linalg.norm(force[:3]) >= threshold)

    def _sample_base_qpos(self, agent: _AgentRuntime) -> np.ndarray:
        reset_cfg = self.cfg.get("reset", {})
        base_qpos = agent.initial_base_qpos.copy()
        xy_noise = float(reset_cfg.get("xy_noise", 0.0))
        yaw_noise = float(reset_cfg.get("yaw_noise", 0.0))
        if xy_noise > 0.0:
            base_qpos[:2] += agent.rng.uniform(-xy_noise, xy_noise, size=2)
        if "base_height" in reset_cfg:
            base_qpos[2] = float(reset_cfg["base_height"])
            if agent.terrain_runtime.physics_enabled:
                base_qpos[2] += agent.terrain_runtime.height_at(float(base_qpos[0]), float(base_qpos[1]))
        if yaw_noise > 0.0:
            yaw_delta = agent.rng.uniform(-yaw_noise, yaw_noise)
            base_qpos[3:7] = quat_mul_wxyz(base_qpos[3:7], yaw_quat_wxyz(yaw_delta))
            quat_norm = np.linalg.norm(base_qpos[3:7])
            if quat_norm > 1e-12:
                base_qpos[3:7] /= quat_norm
        return base_qpos

    def _setup_contact_classification(self, agent: _AgentRuntime) -> None:
        body_names = set(self.model.get_body_names()) if hasattr(self.model, "get_body_names") else set()
        prefixed = {name for name in body_names if name.startswith(f"{agent.agent_name}_")}
        configured = set(agent.base_contact_body_names) | set(agent.foot_body_names)
        agent.robot_body_names = prefixed | configured
        agent.illegal_contact_body_names = agent.robot_body_names - set(agent.foot_body_names)

    def _setup_global_randomization_targets(self) -> None:
        self._baseline_geom_friction: dict[str, np.ndarray] = {}
        self._baseline_geom_solref: dict[str, np.ndarray] = {}
        self._baseline_geom_solimp: dict[str, np.ndarray] = {}
        self._baseline_geom_margin: dict[str, float] = {}
        self._baseline_solver_iterations: int | None = None
        self._baseline_solver_tolerance: float | None = None
        model = self._local_mujoco_model()
        if model is None:
            return
        self._baseline_solver_iterations = int(getattr(model.opt, "iterations", 0))
        self._baseline_solver_tolerance = float(getattr(model.opt, "tolerance", 0.0))
        for geom_name in self._find_friction_geom_names():
            try:
                geom = model.geom(geom_name)
                self._baseline_geom_friction[geom_name] = np.asarray(geom.friction, dtype=np.float64).copy()
                self._baseline_geom_solref[geom_name] = np.asarray(geom.solref, dtype=np.float64).copy()
                self._baseline_geom_solimp[geom_name] = np.asarray(geom.solimp, dtype=np.float64).copy()
                self._baseline_geom_margin[geom_name] = float(np.asarray(geom.margin).reshape(-1)[0])
            except Exception:
                continue

    def _setup_agent_randomization_targets(self, agent: _AgentRuntime) -> None:
        model = self._local_mujoco_model()
        if model is None:
            return
        randomization_cfg = self.cfg.get("randomization", {})
        base_body_name = randomization_cfg.get("base_mass_body_name")
        if base_body_name is None and agent.base_contact_body_names:
            base_body_name = agent.base_contact_body_names[0]
        if base_body_name is None:
            return
        try:
            body = model.body(base_body_name)
            agent.randomized_base_body_name = str(base_body_name)
            agent.baseline_base_mass = float(np.asarray(body.mass, dtype=np.float64).reshape(-1)[0])
            agent.baseline_base_ipos = np.asarray(body.ipos, dtype=np.float64).copy()
            agent.baseline_base_inertia = np.asarray(body.inertia, dtype=np.float64).copy()
        except Exception:
            agent.randomized_base_body_name = None

    def _apply_agent_randomization(self, agent: _AgentRuntime, index: int) -> None:
        state = agent.randomization_state
        agent.action_mapper.kp = agent.kp * state.kp_scale
        agent.action_mapper.kd = agent.kd * state.kd_scale
        agent.action_mapper.torque_limits = agent.torque_limits * state.torque_scale
        if hasattr(self, "_kp"):
            self._kp[index] = agent.action_mapper.kp
            self._kd[index] = agent.action_mapper.kd
            self._torque_low[index] = agent.action_mapper.torque_limits[:, 0]
            self._torque_high[index] = agent.action_mapper.torque_limits[:, 1]

        model = self._local_mujoco_model()
        if model is None:
            return
        if agent.randomized_base_body_name is not None and agent.baseline_base_mass is not None:
            body = model.body(agent.randomized_base_body_name)
            body.mass = [max(1e-3, agent.baseline_base_mass + state.base_mass_delta)]
            if agent.baseline_base_ipos is not None:
                body.ipos = agent.baseline_base_ipos + np.asarray(state.base_com_offset, dtype=np.float64)
            if agent.baseline_base_inertia is not None:
                body.inertia = agent.baseline_base_inertia * state.base_inertia_scale

    def _apply_global_randomization(self, state: RandomizationState) -> None:
        model = self._local_mujoco_model()
        if model is None:
            return
        if self._baseline_solver_iterations is not None and state.solver_iterations is not None:
            try:
                model.opt.iterations = int(state.solver_iterations)
            except Exception:
                pass
        if self._baseline_solver_tolerance is not None:
            try:
                model.opt.tolerance = max(0.0, self._baseline_solver_tolerance * state.solver_tolerance_scale)
            except Exception:
                pass
        if self._baseline_geom_friction:
            friction_dict = {
                name: (base_friction * state.friction_scale).astype(np.float64)
                for name, base_friction in self._baseline_geom_friction.items()
            }
            self.set_geom_friction(friction_dict)
            for geom_name, solref in self._baseline_geom_solref.items():
                try:
                    geom = model.geom(geom_name)
                    randomized_solref = solref.copy()
                    if randomized_solref.size >= 1:
                        randomized_solref[0] = max(1e-5, randomized_solref[0] * state.contact_solref_timeconst_scale)
                    if randomized_solref.size >= 2:
                        randomized_solref[1] = max(1e-5, randomized_solref[1] * state.contact_solref_dampratio_scale)
                    geom.solref = randomized_solref
                    geom.solimp = self._randomized_solimp(self._baseline_geom_solimp[geom_name], state)
                    geom.margin = max(0.0, self._baseline_geom_margin[geom_name] * state.contact_margin_scale)
                except Exception:
                    continue

    def _find_friction_geom_names(self) -> list[str]:
        geom_dict = self.model.get_geom_dict() if hasattr(self.model, "get_geom_dict") else {}
        randomization_cfg = self.cfg.get("randomization", {})
        explicit_names = [str(name) for name in randomization_cfg.get("friction_geom_names", ())]
        if explicit_names:
            return [name for name in explicit_names if name in geom_dict]
        patterns = tuple(
            str(pattern).lower()
            for pattern in randomization_cfg.get(
                "friction_geom_patterns",
                ("floor", "ground", "terrain", "plane", "hfield"),
            )
        )
        matches: list[str] = []
        for geom_name, geom_info in geom_dict.items():
            body_name = str(geom_info.get("BodyName", ""))
            haystack = f"{geom_name} {body_name}".lower()
            if any(pattern in haystack for pattern in patterns):
                matches.append(str(geom_name))
        return matches

    def _randomized_solimp(self, solimp: np.ndarray, state: RandomizationState) -> np.ndarray:
        randomized = solimp.copy()
        scale = state.contact_solimp_scale
        if randomized.size >= 1:
            randomized[0] = float(np.clip(randomized[0] * scale, 1e-4, 0.999))
        if randomized.size >= 2:
            randomized[1] = float(np.clip(randomized[1] * scale, randomized[0] + 1e-4, 0.9999))
        if randomized.size >= 3:
            randomized[2] = max(1e-6, randomized[2] * scale)
        if randomized.size >= 4:
            randomized[3] = float(np.clip(randomized[3], 1e-4, 0.9999))
        if randomized.size >= 5:
            randomized[4] = max(1e-3, randomized[4])
        return randomized

    def _delayed_action(self, agent: _AgentRuntime, action: np.ndarray) -> np.ndarray:
        delay_steps = max(0, int(agent.randomization_state.action_delay_steps))
        max_len = delay_steps + 1
        agent.action_delay_buffer.append(action.copy())
        if len(agent.action_delay_buffer) > max_len:
            agent.action_delay_buffer = agent.action_delay_buffer[-max_len:]
        if len(agent.action_delay_buffer) <= delay_steps:
            return agent.action_delay_buffer[0].copy()
        return agent.action_delay_buffer[-delay_steps - 1].copy()

    def _reset_action_delay_buffer(self, agent: _AgentRuntime) -> None:
        delay_steps = max(0, int(agent.randomization_state.action_delay_steps))
        zero_action = np.zeros(self.num_actions, dtype=np.float64)
        agent.action_delay_buffer = [zero_action.copy() for _ in range(delay_steps + 1)]

    def _maybe_apply_push_disturbances(self) -> None:
        data = getattr(getattr(self, "gym", None), "_mjData", None) or self.data
        any_push = False
        for index, agent in enumerate(self.agents):
            if agent.push_interval_steps <= 0 or self.episode_lengths[index] <= 0:
                continue
            if self.episode_lengths[index] % agent.push_interval_steps != 0:
                continue
            low, high = agent.randomizer.cfg.push_velocity_range
            yaw_low, yaw_high = agent.randomizer.cfg.push_yaw_velocity_range
            data.qvel[agent.base_qvel_offset : agent.base_qvel_offset + 2] += agent.rng.uniform(low, high, size=2)
            data.qvel[agent.base_qvel_offset + 5] += agent.rng.uniform(yaw_low, yaw_high)
            any_push = True
        if any_push:
            self.mj_forward()
            self.update_data()

    def _push_interval_steps(self) -> int:
        interval_s = float(self.cfg.get("randomization", {}).get("push_interval_s", 0.0))
        if interval_s <= 0.0:
            return 0
        return max(1, int(round(interval_s / self.control_dt)))

    def _randomization_observation(self, agent: _AgentRuntime) -> np.ndarray:
        state = agent.randomization_state
        solver_iterations = 0.0 if state.solver_iterations is None else float(state.solver_iterations) / 100.0
        return np.array(
            [
                state.friction_scale,
                state.base_mass_delta,
                state.base_inertia_scale,
                *state.base_com_offset,
                state.kp_scale,
                state.kd_scale,
                state.torque_scale,
                float(state.action_delay_steps),
                solver_iterations,
                state.solver_tolerance_scale,
                state.contact_solref_timeconst_scale,
                state.contact_solref_dampratio_scale,
                state.contact_solimp_scale,
                state.contact_margin_scale,
            ],
            dtype=np.float64,
        )

    def _local_mujoco_model(self) -> Any | None:
        gym = getattr(self, "gym", None)
        return getattr(gym, "_mjModel", None)


def _height_scan_geomgroup(model: mujoco.MjModel, cfg: dict[str, Any]) -> np.ndarray:
    sensors = cfg.get("sensors", {})
    sensor_cfg = sensors.get("terrain_scan") if isinstance(sensors, dict) else None
    terrain_cfg = cfg.get("terrain", {})
    groups = None
    if isinstance(sensor_cfg, dict):
        groups = sensor_cfg.get("include_geom_groups")
    if groups is None and isinstance(terrain_cfg, dict):
        groups = terrain_cfg.get("raycast_geom_groups")
    if groups is None:
        groups = (0,)
    if isinstance(groups, int):
        groups = (groups,)
    geomgroup = np.zeros(6, dtype=np.uint8)
    for group in groups:
        group_id = int(group)
        if 0 <= group_id < geomgroup.size:
            geomgroup[group_id] = 1
    if not np.any(geomgroup):
        geomgroup[:] = 1
    # If the loaded model has no geoms in the requested groups, fall back to all
    # groups so converted OrcaLab assets with unknown group IDs can still be hit.
    model_groups = np.asarray(model.geom_group[: model.ngeom], dtype=np.int32)
    if not np.any(geomgroup[np.clip(model_groups, 0, geomgroup.size - 1)]):
        geomgroup[:] = 1
    return geomgroup


def _agent_root_body_id(model: mujoco.MjModel, model_wrapper: Any, agent: _AgentRuntime) -> int:
    try:
        joint_dict = model_wrapper.get_joint_dict()
        joint_info = joint_dict.get(agent.base_joint_name)
        if joint_info is not None:
            joint_id = int(joint_info["JointId"])
        else:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, agent.base_joint_name)
        if joint_id >= 0:
            return int(model.jnt_bodyid[joint_id])
    except Exception:
        pass
    return -1


def _robot_geom_ids(model: mujoco.MjModel, agent: _AgentRuntime) -> np.ndarray:
    robot_body_names = set(getattr(agent, "robot_body_names", set()))
    if not robot_body_names:
        return np.zeros(0, dtype=np.int32)
    body_ids = set()
    for body_name in robot_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
        if body_id >= 0:
            body_ids.add(int(body_id))
    if not body_ids:
        return np.zeros(0, dtype=np.int32)
    geom_ids = [geom_id for geom_id in range(int(model.ngeom)) if int(model.geom_bodyid[geom_id]) in body_ids]
    return np.asarray(geom_ids, dtype=np.int32)
