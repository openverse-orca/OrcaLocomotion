from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MdpFunc = Callable[..., Any] | str


def _func_name(func: MdpFunc) -> str:
    if isinstance(func, str):
        return func
    return f"{func.__module__}.{func.__name__}"


@dataclass
class SceneEntityCfg:
    name: str = "robot"
    body_names: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    site_names: tuple[str, ...] = ()
    geom_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body_names": self.body_names,
            "joint_names": self.joint_names,
            "site_names": self.site_names,
            "geom_names": self.geom_names,
        }


@dataclass
class TermCfg:
    func: MdpFunc
    weight: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    time_out: bool = False

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "func": _func_name(self.func),
            "params": _metadata_value(self.params),
        }
        if self.weight is not None:
            data["weight"] = self.weight
        if self.time_out:
            data["time_out"] = True
        return data


@dataclass
class EventTermCfg(TermCfg):
    mode: str = "reset"
    interval_range_s: tuple[float, float] | None = None

    def metadata(self) -> dict[str, Any]:
        data = super().metadata()
        data["mode"] = self.mode
        if self.interval_range_s is not None:
            data["interval_range_s"] = self.interval_range_s
        return data


@dataclass
class ObservationTermCfg(TermCfg):
    scale: float | None = None
    noise: tuple[float, float] | None = None

    def metadata(self) -> dict[str, Any]:
        data = super().metadata()
        if self.scale is not None:
            data["scale"] = self.scale
        if self.noise is not None:
            data["noise"] = self.noise
        return data


@dataclass
class ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg]
    concatenate_terms: bool = True
    enable_corruption: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "terms": {name: term.metadata() for name, term in self.terms.items()},
            "concatenate_terms": self.concatenate_terms,
            "enable_corruption": self.enable_corruption,
        }


@dataclass
class UniformVelocityCommandCfg(TermCfg):
    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-0.5, 0.5)

    entity_name: str = "robot"
    resampling_time_s: float = 4.0
    ranges: Ranges = field(default_factory=Ranges)
    heading_command: bool = False

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "lin_vel_x": self.ranges.lin_vel_x,
            "lin_vel_y": self.ranges.lin_vel_y,
            "yaw_vel": self.ranges.ang_vel_z,
            "resample_time_s": self.resampling_time_s,
        }

    def metadata(self) -> dict[str, Any]:
        data = super().metadata()
        data.update(
            {
                "entity_name": self.entity_name,
                "resampling_time_s": self.resampling_time_s,
                "heading_command": self.heading_command,
                "ranges": {
                    "lin_vel_x": self.ranges.lin_vel_x,
                    "lin_vel_y": self.ranges.lin_vel_y,
                    "ang_vel_z": self.ranges.ang_vel_z,
                },
            }
        )
        return data


@dataclass
class JointPositionActionCfg(TermCfg):
    scale: float | tuple[float, ...] = 0.5
    use_default_offset: bool = True
    clip: float = 1.0
    safety_scale: float = 0.85
    max_delta: tuple[float, ...] = ()

    def to_legacy_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action_clip": self.clip,
            "safety_scale": self.safety_scale,
        }
        if self.max_delta:
            data["max_delta"] = self.max_delta
        return data

    def metadata(self) -> dict[str, Any]:
        data = super().metadata()
        data.update(
            {
                "scale": self.scale,
                "use_default_offset": self.use_default_offset,
                "clip": self.clip,
                "safety_scale": self.safety_scale,
                "max_delta": self.max_delta,
            }
        )
        return data


@dataclass
class LocomotionEnvCfg:
    name: str
    robot: str
    rsl_rl_config: str
    num_envs: int = 1
    device: str = "cuda:0"
    orcagym_addresses: tuple[str, ...] = ("localhost:50051",)
    seed: int = 1
    sim: dict[str, Any] = field(default_factory=dict)
    terrain: Any | None = None
    sensors: dict[str, Any] = field(default_factory=dict)
    scene_binding: dict[str, Any] = field(default_factory=dict)
    episode: dict[str, Any] = field(default_factory=dict)
    actions: dict[str, JointPositionActionCfg] = field(default_factory=dict)
    commands: dict[str, UniformVelocityCommandCfg] = field(default_factory=dict)
    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    observation_scales: dict[str, Any] = field(default_factory=dict)
    rewards: dict[str, TermCfg] = field(default_factory=dict)
    terminations: dict[str, TermCfg] = field(default_factory=dict)
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    reset: dict[str, Any] = field(default_factory=dict)
    contacts: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)
    curriculum: dict[str, TermCfg] = field(default_factory=dict)
    metrics: dict[str, TermCfg] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    play: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "robot": self.robot,
            "num_envs": self.num_envs,
            "device": self.device,
            "orcagym_addresses": list(self.orcagym_addresses),
            "rsl_rl_config": self.rsl_rl_config,
            "seed": self.seed,
            "sim": self.sim,
            "terrain": _metadata_value(self.terrain),
            "sensors": _metadata_value(self.sensors),
            "scene_binding": self.scene_binding,
            "episode": self.episode,
            "control": self._legacy_control(),
            "commands": self._legacy_commands(),
            "observations": self.observation_scales,
            "rewards": self._legacy_rewards(),
            "termination": self._legacy_termination(),
            "reset": self.reset,
            "contacts": self.contacts,
            "randomization": self.randomization,
            "curriculum": {name: term.metadata() for name, term in self.curriculum.items()},
            "metrics": {name: term.metadata() for name, term in self.metrics.items()},
            "train": self.train,
            "play": self.play,
            "eval": self.eval,
            "export": self.export,
            "manager_terms": self._manager_metadata(),
        }

    def _legacy_control(self) -> dict[str, Any]:
        action = self.actions.get("joint_pos")
        return action.to_legacy_dict() if action is not None else {}

    def _legacy_commands(self) -> dict[str, Any]:
        command = self.commands.get("twist")
        return command.to_legacy_dict() if command is not None else {}

    def _legacy_rewards(self) -> dict[str, Any]:
        rewards = self.rewards
        foot_clearance = rewards.get("foot_clearance")
        data = {
            "tracking_lin_vel": float(rewards["track_linear_velocity"].weight or 0.0),
            "tracking_ang_vel": float(rewards["track_angular_velocity"].weight or 0.0),
            "lin_vel_sigma": float(rewards["track_linear_velocity"].params.get("sigma", 0.25)),
            "ang_vel_sigma": float(rewards["track_angular_velocity"].params.get("sigma", 0.25)),
            "z_vel": float(rewards["z_velocity_l2"].weight or 0.0),
            "orientation": float(rewards["orientation_l2"].weight or 0.0),
            "height": float(rewards["base_height_l2"].weight or 0.0),
            "torque": float(rewards["torques_l2"].weight or 0.0),
            "action_rate": float(rewards["action_rate_l2"].weight or 0.0),
            "joint_limit": float(rewards["joint_pos_limits"].weight or 0.0),
            "foot_slip": float(rewards["feet_slip"].weight or 0.0),
            "termination": float(rewards["termination"].weight or 0.0),
            "target_height": float(rewards["base_height_l2"].params.get("target_height", 0.34)),
        }
        if "feet_air_time" in rewards:
            data["feet_air_time"] = float(rewards["feet_air_time"].weight or 0.0)
        if foot_clearance is not None:
            data["foot_clearance"] = float(foot_clearance.weight or 0.0)
            data["target_foot_clearance"] = float(foot_clearance.params.get("target_height", 0.08))
        if "body_ang_vel_l2" in rewards:
            data["body_ang_vel"] = float(rewards["body_ang_vel_l2"].weight or 0.0)
        if "stand_still" in rewards:
            data["stand_still"] = float(rewards["stand_still"].weight or 0.0)
            data["command_deadzone"] = float(rewards["stand_still"].params.get("command_deadzone", 0.1))
        if "joint_deviation_l1" in rewards:
            data["joint_deviation"] = float(rewards["joint_deviation_l1"].weight or 0.0)
        return data

    def _legacy_termination(self) -> dict[str, Any]:
        fell_over = self.terminations["fell_over"]
        base_height = self.terminations["base_height"]
        return {
            "min_base_height": float(base_height.params["min_base_height"]),
            "max_base_height": float(base_height.params["max_base_height"]),
            "max_tilt_rad": float(fell_over.params["max_tilt_rad"]),
            "terminate_on_base_contact": bool(self.terminations["base_contact"].params["enabled"]),
            "terminate_on_illegal_contact": "illegal_contact" in self.terminations,
        }

    def _manager_metadata(self) -> dict[str, Any]:
        return {
            "actions": {name: term.metadata() for name, term in self.actions.items()},
            "commands": {name: term.metadata() for name, term in self.commands.items()},
            "observations": {name: group.metadata() for name, group in self.observations.items()},
            "events": {name: term.metadata() for name, term in self.events.items()},
            "rewards": {name: term.metadata() for name, term in self.rewards.items()},
            "terminations": {name: term.metadata() for name, term in self.terminations.items()},
            "curriculum": {name: term.metadata() for name, term in self.curriculum.items()},
            "metrics": {name: term.metadata() for name, term in self.metrics.items()},
        }


@dataclass
class RslRlModelCfg:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict[str, Any] | None = None
    class_name: str = "MLPModel"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "class_name": self.class_name,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "obs_normalization": self.obs_normalization,
        }
        if self.distribution_cfg is not None:
            data["distribution_cfg"] = self.distribution_cfg
        return data


@dataclass
class RslRlPpoAlgorithmCfg:
    learning_rate: float = 3.0e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    value_loss_coef: float = 1.0
    clip_param: float = 0.2
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    class_name: str = "PPO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "learning_rate": self.learning_rate,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "schedule": self.schedule,
            "value_loss_coef": self.value_loss_coef,
            "clip_param": self.clip_param,
            "use_clipped_value_loss": self.use_clipped_value_loss,
            "desired_kl": self.desired_kl,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "lam": self.lam,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage_per_mini_batch": self.normalize_advantage_per_mini_batch,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }


@dataclass
class RslRlOnPolicyRunnerCfg:
    actor: RslRlModelCfg
    critic: RslRlModelCfg
    algorithm: RslRlPpoAlgorithmCfg
    experiment_name: str
    run_name: str
    save_interval: int = 100
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    logger: str = "tensorboard"
    wandb_project: str = "orca_locomotion"
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"actor": ["policy"], "critic": ["policy", "privileged"]}
    )
    check_for_nan: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": {
                "num_steps_per_env": self.num_steps_per_env,
                "max_iterations": self.max_iterations,
                "obs_groups": self.obs_groups,
                "save_interval": self.save_interval,
                "logger": self.logger,
                "wandb_project": self.wandb_project,
                "experiment_name": self.experiment_name,
                "run_name": self.run_name,
                "check_for_nan": self.check_for_nan,
                "algorithm": self.algorithm.to_dict(),
                "actor": self.actor.to_dict(),
                "critic": self.critic.to_dict(),
            }
        }


def _metadata_value(value: Any) -> Any:
    if isinstance(value, SceneEntityCfg):
        return value.to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_value(item) for item in value)
    return value
