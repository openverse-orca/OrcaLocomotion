from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orca_gym.environment import OrcaGymLocalEnv

from .action_mapper import ActionMapperConfig, ResidualJointTargetActionMapper
from .curriculum import CommandConfig, FlatVelocityCommandSampler
from .math_utils import quat_mul_wxyz, yaw_quat_wxyz
from .obs_builder import LocomotionObservationBuilder, LocomotionTaskState, ObservationConfig
from .randomization import DomainRandomizer, RandomizationConfig, RandomizationState
from .reward_manager import FlatVelocityReward, RewardConfig
from .terrain_runtime import TerrainRuntime
from .termination_manager import TerminationManager, TerminationConfig


@dataclass
class LocomotionStepResult:
    observations: dict[str, np.ndarray]
    reward: float
    done: bool
    time_out: bool
    log: dict[str, float]


class OrcaLocomotionTask(OrcaGymLocalEnv):
    """Single-robot flat velocity task for the clean RSL-RL locomotion stack."""

    metadata = {"render_modes": ["human", "none"], "version": "0.1.0", "render_fps": 30}

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        orcagym_addr: str,
        agent_name: str,
        robot_config: dict[str, Any],
        render_mode: str = "none",
        env_id: str = "Locomotion-OrcaGym",
    ) -> None:
        self.cfg = cfg
        self.robot_config = robot_config
        self.agent_name = agent_name
        sim_cfg = cfg["sim"]
        self._render_mode = render_mode
        self.env_id = env_id
        self.decimation = int(sim_cfg.get("decimation", 4))
        self.control_dt = float(sim_cfg["time_step"]) * int(sim_cfg["frame_skip"]) * self.decimation
        self.max_episode_length = int(round(float(cfg["episode"]["length_s"]) / self.control_dt))
        self.episode_length = 0
        self.rng = np.random.default_rng(int(cfg.get("seed", 1)))

        super().__init__(
            frame_skip=int(sim_cfg["frame_skip"]),
            orcagym_addr=orcagym_addr,
            agent_names=[agent_name],
            time_step=float(sim_cfg["time_step"]),
        )

        self.nu = self.model.nu
        self.ctrl = np.zeros(self.nu, dtype=np.float64)
        self._setup_robot_handles()
        self._setup_managers()
        self.reset_model()

    @property
    def num_actions(self) -> int:
        return self.action_mapper.num_actions

    def _setup_robot_handles(self) -> None:
        robot_label = str(self.cfg.get("robot", "robot")).upper()
        self.base_joint_name = self.joint(self.robot_config["base_joint_name"])
        self.leg_joint_names = [self.joint(name) for name in self.robot_config["leg_joint_names"]]
        self.actuator_names = [self.actuator(name) for name in self.robot_config["actuator_names"]]
        requested_contact_sites = [self.site(name) for name in self.robot_config.get("contact_site_names", [])]
        self.foot_sensor_names = [self.sensor(name) for name in self.robot_config.get("sensor_foot_touch_names", [])]
        self.base_contact_body_names = [
            self.body(name) for name in self.robot_config.get("base_contact_body_names", [])
        ]
        self.foot_body_names = [self.body(name) for name in self.robot_config.get("foot_body_names", [])]

        joint_dict = self.model.get_joint_dict()
        actuator_dict = self.model.get_actuator_dict()
        site_dict = self.model.get_site_dict() if hasattr(self.model, "get_site_dict") else {}
        sensor_dict = getattr(self.model, "_sensor_dict", {})
        missing_joints = [name for name in [self.base_joint_name, *self.leg_joint_names] if name not in joint_dict]
        missing_actuators = [name for name in self.actuator_names if name not in actuator_dict]
        if missing_joints or missing_actuators:
            raise ValueError(
                f"{robot_label} binding incomplete. missing_joints={missing_joints}, "
                f"missing_actuators={missing_actuators}"
            )
        available_contact_sites = [name for name in requested_contact_sites if name in site_dict]
        self.contact_site_names = (
            available_contact_sites if len(available_contact_sites) == len(requested_contact_sites) else []
        )
        if not self.contact_site_names and not self.foot_body_names:
            raise ValueError(
                f"{robot_label} binding incomplete. No contact sites or foot bodies are available for foot tracking."
            )
        self.foot_sensor_names = [name for name in self.foot_sensor_names if name in sensor_dict]

        self.base_qpos_offset, self.base_qvel_offset, _ = [
            int(values[0]) for values in self.query_joint_offsets([self.base_joint_name])
        ]
        qpos_offsets, qvel_offsets, _ = self.query_joint_offsets(self.leg_joint_names)
        qpos_lengths, qvel_lengths, _ = self.query_joint_lengths(self.leg_joint_names)
        self.leg_qpos_indices = np.array([int(offset) for offset in qpos_offsets], dtype=np.int64)
        self.leg_qvel_indices = np.array([int(offset) for offset in qvel_offsets], dtype=np.int64)
        if any(int(length) != 1 for length in qpos_lengths) or any(int(length) != 1 for length in qvel_lengths):
            raise ValueError(f"{robot_label} controlled joints are expected to be one-DoF joints.")

        self.actuator_ids = np.array(
            [int(actuator_dict[name]["ActuatorId"]) for name in self.actuator_names],
            dtype=np.int64,
        )
        configured_joint_limits = self.robot_config.get("joint_limits")
        if configured_joint_limits is not None:
            self.joint_limits = np.asarray(configured_joint_limits, dtype=np.float64).reshape(-1, 2)
        else:
            self.joint_limits = np.array([joint_dict[name]["Range"] for name in self.leg_joint_names], dtype=np.float64)

        effort_limits = self.robot_config.get("motor_effort_limit_list")
        if effort_limits is not None:
            effort = np.asarray(effort_limits, dtype=np.float64).reshape(-1)
            self.torque_limits = np.stack([-effort, effort], axis=1)
        else:
            self.torque_limits = np.array(
                [actuator_dict[name].get("CtrlRange", [-1.0, 1.0]) for name in self.actuator_names],
                dtype=np.float64,
            )
        self.nominal_qpos = np.array(
            [self.robot_config["neutral_joint_angles"][name] for name in self.robot_config["leg_joint_names"]],
            dtype=np.float64,
        )
        self.kp = np.array(self.robot_config["kps"], dtype=np.float64)
        self.kd = np.array(self.robot_config["kds"], dtype=np.float64)
        self.initial_base_qpos = self.data.qpos[self.base_qpos_offset : self.base_qpos_offset + 7].copy()

    def _setup_managers(self) -> None:
        control_cfg = self.cfg.get("control", {})
        self.action_mapper = ResidualJointTargetActionMapper(
            nominal_qpos=self.nominal_qpos,
            joint_limits=self.joint_limits,
            torque_limits=self.torque_limits,
            kp=self.kp,
            kd=self.kd,
            cfg=ActionMapperConfig(
                safety_scale=float(control_cfg.get("safety_scale", 0.85)),
                max_delta=control_cfg.get("max_delta"),
                action_clip=float(control_cfg.get("action_clip", 1.0)),
            ),
        )
        self.obs_builder = LocomotionObservationBuilder(
            nominal_qpos=self.nominal_qpos,
            cfg=ObservationConfig(**self.cfg.get("observations", {})),
            rng=self.rng,
        )
        self.reward_manager = FlatVelocityReward(
            joint_limits=self.joint_limits,
            nominal_qpos=self.nominal_qpos,
            cfg=RewardConfig(**self.cfg.get("rewards", {})),
        )
        self.termination_manager = TerminationManager(TerminationConfig(**self.cfg.get("termination", {})))
        self.command_sampler = FlatVelocityCommandSampler(CommandConfig(**self.cfg.get("commands", {})), self.rng)
        self.randomizer = DomainRandomizer(RandomizationConfig(**self.cfg.get("randomization", {})), self.rng)
        self.terrain_runtime = TerrainRuntime.from_task_cfg(self.cfg, self.rng)
        self._setup_domain_randomization_targets()
        self._setup_contact_classification()
        self.command_resample_steps = max(
            1,
            int(round(float(self.cfg.get("commands", {}).get("resample_time_s", 4.0)) / self.control_dt)),
        )
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)
        self.previous_action = np.zeros(self.num_actions, dtype=np.float64)
        self.last_torque = np.zeros(self.num_actions, dtype=np.float64)
        self.command = self.command_sampler.sample()
        self.randomization_state = RandomizationState()
        self.action_delay_buffer = [np.zeros(self.num_actions, dtype=np.float64)]
        self.push_interval_steps = self._push_interval_steps()
        num_feet = len(self.contact_site_names) if self.contact_site_names else len(self.foot_body_names)
        self.last_foot_pos_world = np.zeros((num_feet, 3), dtype=np.float64)
        self.feet_air_time = np.zeros(num_feet, dtype=np.float64)

    def get_observations(self, noisy: bool = True) -> dict[str, np.ndarray]:
        return self.obs_builder.build(self._read_state(update_air_time=False), noisy=noisy)

    def step(self, action: np.ndarray) -> LocomotionStepResult:
        action = np.asarray(action, dtype=np.float64).reshape(self.num_actions)
        clipped_action = np.clip(action, -1.0, 1.0)
        self.previous_action = self.last_action.copy()
        self.last_action = self._delayed_action(clipped_action)

        if self.episode_length % self.command_resample_steps == 0:
            self.command = self.command_sampler.sample()

        qpos = self.data.qpos[self.leg_qpos_indices].copy()
        qvel = self.data.qvel[self.leg_qvel_indices].copy()
        target_qpos = self.action_mapper.action_to_target_qpos(self.last_action)
        torque = self.action_mapper.compute_torque(target_qpos, qpos, qvel)
        self.last_torque = torque.copy()

        for _ in range(self.decimation):
            self.ctrl[:] = 0.0
            self.ctrl[self.actuator_ids] = torque
            self.do_simulation(self.ctrl, self.frame_skip)
        if self._render_mode == "human":
            self.render()

        self.episode_length += 1
        self._maybe_apply_push_disturbance()
        state = self._read_state(update_air_time=True)
        base_contact = self._has_base_contact()
        illegal_contact = self._has_illegal_contact()
        terminated, termination_log = self.termination_manager.check(state, base_contact, illegal_contact)
        time_out = self.episode_length >= self.max_episode_length
        reward, reward_log = self.reward_manager.compute(
            state=state,
            action=self.last_action,
            previous_action=self.previous_action,
            terminated=terminated,
        )
        done = bool(terminated or time_out)
        log = {**reward_log, **termination_log}
        log["/episode/length"] = float(self.episode_length)
        log["/episode/reward"] = float(reward)

        if done:
            observations, _ = self.reset_model()
        else:
            observations = self.get_observations(noisy=True)

        return LocomotionStepResult(
            observations=observations,
            reward=reward,
            done=done,
            time_out=bool(time_out and not terminated),
            log=log,
        )

    def reset_model(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.episode_length = 0
        self.command = self.command_sampler.sample()
        self.randomization_state = self.randomizer.sample()
        if self.cfg.get("randomization", {}).get("terrain") == "rough":
            self.terrain_runtime.resample()
        self._apply_domain_randomization()
        self._reset_action_delay_buffer()
        self.last_action.fill(0.0)
        self.previous_action.fill(0.0)
        self.last_torque.fill(0.0)

        reset_cfg = self.cfg.get("reset", {})
        base_qpos = self.initial_base_qpos.copy()
        xy_noise = float(reset_cfg.get("xy_noise", 0.0))
        yaw_noise = float(reset_cfg.get("yaw_noise", 0.0))
        if xy_noise > 0.0:
            base_qpos[:2] += self.rng.uniform(-xy_noise, xy_noise, size=2)
        if "base_height" in reset_cfg:
            base_qpos[2] = float(reset_cfg["base_height"])
            if self.terrain_runtime.physics_enabled:
                base_qpos[2] += self.terrain_runtime.height_at(float(base_qpos[0]), float(base_qpos[1]))
        if yaw_noise > 0.0:
            yaw_delta = self.rng.uniform(-yaw_noise, yaw_noise)
            base_qpos[3:7] = quat_mul_wxyz(base_qpos[3:7], yaw_quat_wxyz(yaw_delta))
            quat_norm = np.linalg.norm(base_qpos[3:7])
            if quat_norm > 1e-12:
                base_qpos[3:7] /= quat_norm

        joint_qpos = {self.base_joint_name: base_qpos}
        joint_qvel = {self.base_joint_name: np.zeros(6, dtype=np.float64)}
        q_noise = float(reset_cfg.get("joint_noise", 0.0))
        for index, joint_name in enumerate(self.leg_joint_names):
            value = self.nominal_qpos[index]
            if q_noise > 0.0:
                value += float(self.rng.uniform(-q_noise, q_noise))
            joint_qpos[joint_name] = np.array([value], dtype=np.float64)
            joint_qvel[joint_name] = np.zeros(1, dtype=np.float64)

        self.set_joint_qpos(joint_qpos)
        self.set_joint_qvel(joint_qvel)
        self.ctrl[:] = 0.0
        self.set_ctrl(self.ctrl)
        self.mj_forward()
        self.update_data()
        if self._render_mode == "human":
            self.render()
        self.last_foot_pos_world = self._query_foot_positions()
        self.feet_air_time.fill(0.0)
        return self.get_observations(noisy=True), {"command": self.command.copy()}

    def _read_state(self, update_air_time: bool = False) -> LocomotionTaskState:
        base_qpos = self.data.qpos[self.base_qpos_offset : self.base_qpos_offset + 7].copy()
        base_qvel = self.data.qvel[self.base_qvel_offset : self.base_qvel_offset + 6].copy()
        qpos = self.data.qpos[self.leg_qpos_indices].copy()
        qvel = self.data.qvel[self.leg_qvel_indices].copy()
        foot_pos = self._query_foot_positions()
        foot_vel = (foot_pos - self.last_foot_pos_world) / max(self.control_dt, 1e-6)
        self.last_foot_pos_world = foot_pos.copy()
        foot_contacts = self._query_foot_contacts()
        if foot_contacts.shape != self.feet_air_time.shape:
            self.feet_air_time = np.zeros_like(foot_contacts, dtype=np.float64)
        previous_air_time = self.feet_air_time.copy()
        if update_air_time:
            self.feet_air_time = np.where(foot_contacts > 0.5, 0.0, self.feet_air_time + self.control_dt)
        first_foot_contact = np.logical_and(foot_contacts > 0.5, previous_air_time > 0.0).astype(np.float64)
        foot_ground_heights = np.array(
            [self.terrain_runtime.height_at(float(pos[0]), float(pos[1])) for pos in foot_pos],
            dtype=np.float64,
        )
        return LocomotionTaskState(
            base_pos=base_qpos[:3],
            base_quat=base_qpos[3:7],
            base_lin_vel_world=base_qvel[:3],
            base_ang_vel_world=base_qvel[3:6],
            qpos=qpos,
            qvel=qvel,
            command=self.command.copy(),
            last_action=self.last_action.copy(),
            last_torque=self.last_torque.copy(),
            foot_pos_world=foot_pos,
            foot_vel_world=foot_vel,
            foot_contacts=foot_contacts,
            foot_air_time=previous_air_time,
            first_foot_contact=first_foot_contact,
            foot_ground_heights=foot_ground_heights,
            friction_scale=self.randomization_state.friction_scale,
            base_mass_delta=self.randomization_state.base_mass_delta,
            domain_randomization=self._randomization_observation(),
            height_scan=self.terrain_runtime.scan(base_qpos[:3], base_qpos[3:7]),
        )

    def _query_foot_positions(self) -> np.ndarray:
        if self.contact_site_names:
            site_data = self.query_site_pos_and_quat(self.contact_site_names)
            return np.array([site_data[name]["xpos"] for name in self.contact_site_names], dtype=np.float64)

        unique_body_names = list(dict.fromkeys(self.foot_body_names))
        xpos, _, _ = self.get_body_xpos_xmat_xquat(unique_body_names)
        body_positions = {
            body_name: np.asarray(xpos[index * 3 : (index + 1) * 3], dtype=np.float64)
            for index, body_name in enumerate(unique_body_names)
        }
        return np.array([body_positions[name] for name in self.foot_body_names], dtype=np.float64)

    def _query_foot_contacts(self) -> np.ndarray:
        if self.foot_sensor_names:
            try:
                sensor_data = self.query_sensor_data(self.foot_sensor_names)
                values = [float(np.asarray(sensor_data[name]).reshape(-1)[0]) for name in self.foot_sensor_names]
                threshold = float(self.cfg.get("contacts", {}).get("touch_threshold", 1.0))
                return (np.asarray(values, dtype=np.float64) > threshold).astype(np.float64)
            except Exception:
                pass
        contacts = self.query_contact_simple()
        foot_contacts = np.zeros(len(self.contact_site_names), dtype=np.float64)
        foot_bodies = self.foot_body_names
        if len(foot_contacts) != len(foot_bodies):
            foot_contacts = np.zeros(len(foot_bodies), dtype=np.float64)
        for contact in contacts:
            body1 = self.model.get_geom_body_name(contact["Geom1"])
            body2 = self.model.get_geom_body_name(contact["Geom2"])
            for index, body_name in enumerate(foot_bodies):
                if body_name in (body1, body2):
                    foot_contacts[index] = 1.0
        return foot_contacts

    def _has_base_contact(self) -> bool:
        contacts = self.query_contact_simple()
        for contact in contacts:
            body1 = self.model.get_geom_body_name(contact["Geom1"])
            body2 = self.model.get_geom_body_name(contact["Geom2"])
            if body1 in self.base_contact_body_names or body2 in self.base_contact_body_names:
                return True
        return False

    def _has_illegal_contact(self) -> bool:
        if not self.termination_manager.cfg.terminate_on_illegal_contact:
            return False
        contacts = self.query_contact_simple()
        for contact in contacts:
            body1 = self.model.get_geom_body_name(contact["Geom1"])
            body2 = self.model.get_geom_body_name(contact["Geom2"])
            if self._is_illegal_robot_world_contact(body1, body2):
                return True
        return False

    def _is_illegal_robot_world_contact(self, body1: str, body2: str) -> bool:
        body1_is_robot = body1 in self.robot_body_names
        body2_is_robot = body2 in self.robot_body_names
        if body1_is_robot and body2_is_robot:
            return False
        if body1 in self.illegal_contact_body_names and body2 not in self.robot_body_names:
            return True
        if body2 in self.illegal_contact_body_names and body1 not in self.robot_body_names:
            return True
        return False

    def _setup_contact_classification(self) -> None:
        body_names = set(self.model.get_body_names()) if hasattr(self.model, "get_body_names") else set()
        prefixed = {name for name in body_names if name.startswith(f"{self.agent_name}_")}
        configured = set(self.base_contact_body_names) | set(self.foot_body_names)
        self.robot_body_names = prefixed | configured
        self.illegal_contact_body_names = self.robot_body_names - set(self.foot_body_names)

    def _setup_domain_randomization_targets(self) -> None:
        self._baseline_geom_friction: dict[str, np.ndarray] = {}
        self._baseline_geom_solref: dict[str, np.ndarray] = {}
        self._baseline_geom_solimp: dict[str, np.ndarray] = {}
        self._baseline_geom_margin: dict[str, float] = {}
        self._baseline_base_mass: float | None = None
        self._baseline_base_ipos: np.ndarray | None = None
        self._baseline_base_inertia: np.ndarray | None = None
        self._randomized_base_body_name: str | None = None
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

        randomization_cfg = self.cfg.get("randomization", {})
        base_body_name = randomization_cfg.get("base_mass_body_name")
        if base_body_name is None and self.base_contact_body_names:
            base_body_name = self.base_contact_body_names[0]
        if base_body_name is None:
            return
        try:
            body = model.body(base_body_name)
            self._randomized_base_body_name = str(base_body_name)
            self._baseline_base_mass = float(np.asarray(body.mass, dtype=np.float64).reshape(-1)[0])
            self._baseline_base_ipos = np.asarray(body.ipos, dtype=np.float64).copy()
            self._baseline_base_inertia = np.asarray(body.inertia, dtype=np.float64).copy()
        except Exception:
            self._randomized_base_body_name = None

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

    def _apply_domain_randomization(self) -> None:
        model = self._local_mujoco_model()
        if model is None:
            return

        self._apply_actuator_randomization()
        self._apply_solver_randomization(model)

        if self._baseline_geom_friction:
            friction_dict = {
                name: (base_friction * self.randomization_state.friction_scale).astype(np.float64)
                for name, base_friction in self._baseline_geom_friction.items()
            }
            self.set_geom_friction(friction_dict)
            self._apply_contact_randomization(model)

        if self._randomized_base_body_name is not None and self._baseline_base_mass is not None:
            body = model.body(self._randomized_base_body_name)
            randomized_mass = max(1e-3, self._baseline_base_mass + self.randomization_state.base_mass_delta)
            body.mass = [randomized_mass]
            if self._baseline_base_ipos is not None:
                body.ipos = self._baseline_base_ipos + np.asarray(
                    self.randomization_state.base_com_offset,
                    dtype=np.float64,
                )
            if self._baseline_base_inertia is not None:
                body.inertia = self._baseline_base_inertia * self.randomization_state.base_inertia_scale

    def _apply_actuator_randomization(self) -> None:
        state = self.randomization_state
        self.action_mapper.kp = self.kp * state.kp_scale
        self.action_mapper.kd = self.kd * state.kd_scale
        self.action_mapper.torque_limits = self.torque_limits * state.torque_scale

    def _apply_solver_randomization(self, model: Any) -> None:
        state = self.randomization_state
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

    def _apply_contact_randomization(self, model: Any) -> None:
        state = self.randomization_state
        for geom_name, solref in self._baseline_geom_solref.items():
            try:
                geom = model.geom(geom_name)
                randomized_solref = solref.copy()
                if randomized_solref.size >= 1:
                    randomized_solref[0] = max(1e-5, randomized_solref[0] * state.contact_solref_timeconst_scale)
                if randomized_solref.size >= 2:
                    randomized_solref[1] = max(
                        1e-5,
                        randomized_solref[1] * state.contact_solref_dampratio_scale,
                    )
                geom.solref = randomized_solref
                geom.solimp = self._randomized_solimp(self._baseline_geom_solimp[geom_name])
                geom.margin = max(0.0, self._baseline_geom_margin[geom_name] * state.contact_margin_scale)
            except Exception:
                continue

    def _randomized_solimp(self, solimp: np.ndarray) -> np.ndarray:
        randomized = solimp.copy()
        scale = self.randomization_state.contact_solimp_scale
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

    def _delayed_action(self, action: np.ndarray) -> np.ndarray:
        delay_steps = max(0, int(self.randomization_state.action_delay_steps))
        max_len = delay_steps + 1
        self.action_delay_buffer.append(action.copy())
        if len(self.action_delay_buffer) > max_len:
            self.action_delay_buffer = self.action_delay_buffer[-max_len:]
        if len(self.action_delay_buffer) <= delay_steps:
            return self.action_delay_buffer[0].copy()
        return self.action_delay_buffer[-delay_steps - 1].copy()

    def _reset_action_delay_buffer(self) -> None:
        delay_steps = max(0, int(self.randomization_state.action_delay_steps))
        zero_action = np.zeros(self.num_actions, dtype=np.float64)
        self.action_delay_buffer = [zero_action.copy() for _ in range(delay_steps + 1)]

    def _maybe_apply_push_disturbance(self) -> None:
        if self.push_interval_steps <= 0 or self.episode_length <= 0:
            return
        if self.episode_length % self.push_interval_steps != 0:
            return
        low, high = self.randomizer.cfg.push_velocity_range
        yaw_low, yaw_high = self.randomizer.cfg.push_yaw_velocity_range
        data = getattr(getattr(self, "gym", None), "_mjData", None) or self.data
        data.qvel[self.base_qvel_offset : self.base_qvel_offset + 2] += self.rng.uniform(low, high, size=2)
        data.qvel[self.base_qvel_offset + 5] += self.rng.uniform(yaw_low, yaw_high)
        self.mj_forward()
        self.update_data()

    def _push_interval_steps(self) -> int:
        interval_s = float(self.cfg.get("randomization", {}).get("push_interval_s", 0.0))
        if interval_s <= 0.0:
            return 0
        return max(1, int(round(interval_s / self.control_dt)))

    def _randomization_observation(self) -> np.ndarray:
        state = self.randomization_state
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
