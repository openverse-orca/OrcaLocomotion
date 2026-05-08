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
        self.base_contact_body_names = [self.body(name) for name in self.robot_config.get("base_contact_body_names", [])]
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
            cfg=RewardConfig(**self.cfg.get("rewards", {})),
        )
        self.termination_manager = TerminationManager(TerminationConfig(**self.cfg.get("termination", {})))
        self.command_sampler = FlatVelocityCommandSampler(CommandConfig(**self.cfg.get("commands", {})), self.rng)
        self.randomizer = DomainRandomizer(RandomizationConfig(**self.cfg.get("randomization", {})), self.rng)
        self.command_resample_steps = max(
            1,
            int(round(float(self.cfg.get("commands", {}).get("resample_time_s", 4.0)) / self.control_dt)),
        )
        self.last_action = np.zeros(self.num_actions, dtype=np.float64)
        self.previous_action = np.zeros(self.num_actions, dtype=np.float64)
        self.last_torque = np.zeros(self.num_actions, dtype=np.float64)
        self.command = self.command_sampler.sample()
        self.randomization_state = RandomizationState()
        num_feet = len(self.contact_site_names) if self.contact_site_names else len(self.foot_body_names)
        self.last_foot_pos_world = np.zeros((num_feet, 3), dtype=np.float64)

    def get_observations(self, noisy: bool = True) -> dict[str, np.ndarray]:
        return self.obs_builder.build(self._read_state(), noisy=noisy)

    def step(self, action: np.ndarray) -> LocomotionStepResult:
        action = np.asarray(action, dtype=np.float64).reshape(self.num_actions)
        self.previous_action = self.last_action.copy()
        self.last_action = np.clip(action, -1.0, 1.0)

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
        state = self._read_state()
        base_contact = self._has_base_contact()
        terminated, termination_log = self.termination_manager.check(state, base_contact)
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
        return self.get_observations(noisy=True), {"command": self.command.copy()}

    def _read_state(self) -> LocomotionTaskState:
        base_qpos = self.data.qpos[self.base_qpos_offset : self.base_qpos_offset + 7].copy()
        base_qvel = self.data.qvel[self.base_qvel_offset : self.base_qvel_offset + 6].copy()
        qpos = self.data.qpos[self.leg_qpos_indices].copy()
        qvel = self.data.qvel[self.leg_qvel_indices].copy()
        foot_pos = self._query_foot_positions()
        foot_vel = (foot_pos - self.last_foot_pos_world) / max(self.control_dt, 1e-6)
        self.last_foot_pos_world = foot_pos.copy()
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
            foot_contacts=self._query_foot_contacts(),
            friction_scale=self.randomization_state.friction_scale,
            base_mass_delta=self.randomization_state.base_mass_delta,
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
