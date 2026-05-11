from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class MjWarpUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MjWarpRuntimeInfo:
    device: str
    nworld: int
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    ctrl_shape: tuple[int, ...]


class MjWarpRuntime:
    """Small MJWarp bridge for an existing OrcaGymLocalEnv MuJoCo model.

    This keeps OrcaGymLocalEnv as the source of model metadata and lets the
    physics step run through mujoco_warp. CPU MjData is synchronized after each
    control step so the existing observation/reward/contact code remains usable.
    """

    def __init__(
        self,
        *,
        model: Any,
        data: Any,
        device: str = "cuda:0",
        nworld: int = 1,
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        try:
            import mujoco
            import mujoco_warp as mjwarp
            import torch
            import warp as wp
        except ImportError as exc:
            raise MjWarpUnavailableError(
                "MJWarp backend requires MuJoCo >= 3.8, mujoco-warp, and warp-lang. "
                "Install the GPU requirements from README.md."
            ) from exc

        self.mujoco = mujoco
        self.mjwarp = mjwarp
        self.torch = torch
        self.wp = wp
        self.model = model
        self.cpu_data = data
        self.device = str(device)
        self.nworld = int(nworld)
        if self.nworld != 1:
            raise ValueError("OrcaGymLocalEnv integration currently expects nworld=1.")
        self.nconmax = int(nconmax or max(256, model.nbody * 8))
        self.njmax = int(njmax or max(512, model.nv * 8))

        with self.wp.ScopedDevice(self.device):
            self.wp_model = self.mjwarp.put_model(model)
            self.wp_data = self.mjwarp.put_data(
                model,
                data,
                nworld=self.nworld,
                nconmax=self.nconmax,
                njmax=self.njmax,
            )

        self.qpos = self.wp.to_torch(self.wp_data.qpos)
        self.qvel = self.wp.to_torch(self.wp_data.qvel)
        self.qacc = self.wp.to_torch(self.wp_data.qacc)
        self.ctrl = self.wp.to_torch(self.wp_data.ctrl)
        self.actuator_force = self.wp.to_torch(self.wp_data.actuator_force)
        self.sensordata = self.wp.to_torch(self.wp_data.sensordata)
        self.time = self.wp.to_torch(self.wp_data.time)
        self.sync_from_cpu(data)

    @property
    def info(self) -> MjWarpRuntimeInfo:
        return MjWarpRuntimeInfo(
            device=self.device,
            nworld=self.nworld,
            qpos_shape=tuple(self.qpos.shape),
            qvel_shape=tuple(self.qvel.shape),
            ctrl_shape=tuple(self.ctrl.shape),
        )

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        value = self.torch.as_tensor(ctrl, dtype=self.ctrl.dtype, device=self.ctrl.device).reshape(-1)
        self.ctrl[0, : value.numel()] = value

    def step(self, nstep: int = 1) -> None:
        with self.wp.ScopedDevice(self.device):
            for _ in range(int(nstep)):
                self.mjwarp.step(self.wp_model, self.wp_data)

    def forward(self) -> None:
        with self.wp.ScopedDevice(self.device):
            self.mjwarp.forward(self.wp_model, self.wp_data)

    def sync_from_cpu(self, data: Any) -> None:
        self.qpos[0] = self.torch.as_tensor(np.asarray(data.qpos), dtype=self.qpos.dtype, device=self.qpos.device)
        self.qvel[0] = self.torch.as_tensor(np.asarray(data.qvel), dtype=self.qvel.dtype, device=self.qvel.device)
        if self.ctrl.shape[-1]:
            self.ctrl[0] = self.torch.as_tensor(np.asarray(data.ctrl), dtype=self.ctrl.dtype, device=self.ctrl.device)
        self.time[0] = float(data.time)

    def sync_to_cpu(self, data: Any, *, recompute_contacts: bool = True) -> None:
        data.qpos[:] = self.qpos[0].detach().cpu().numpy().astype(np.float64, copy=False)
        data.qvel[:] = self.qvel[0].detach().cpu().numpy().astype(np.float64, copy=False)
        if data.ctrl.size:
            data.ctrl[:] = self.ctrl[0].detach().cpu().numpy().astype(np.float64, copy=False)
        data.time = float(self.time[0].detach().cpu().item())
        if hasattr(data, "qacc") and self.qacc.numel():
            data.qacc[:] = self.qacc[0].detach().cpu().numpy().astype(np.float64, copy=False)
        if hasattr(data, "actuator_force") and self.actuator_force.numel():
            data.actuator_force[:] = self.actuator_force[0].detach().cpu().numpy().astype(np.float64, copy=False)
        if hasattr(data, "sensordata") and self.sensordata.numel():
            data.sensordata[:] = self.sensordata[0].detach().cpu().numpy().astype(np.float64, copy=False)
        if recompute_contacts:
            self.mujoco.mj_forward(self.model, data)
