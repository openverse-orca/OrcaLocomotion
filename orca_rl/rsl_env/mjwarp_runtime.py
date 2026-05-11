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
        self._step_graphs: dict[int, Any] = {}
        self._forward_graph: Any | None = None
        self.sync_from_cpu(data)
        self._capture_forward_graph()

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
        nstep = int(nstep)
        with self.wp.ScopedDevice(self.device):
            graph = self._step_graphs.get(nstep)
            if graph is None:
                self._capture_step_graph(nstep)
                graph = self._step_graphs.get(nstep)
            if graph is not None:
                self.wp.capture_launch(graph)
            else:
                for _ in range(nstep):
                    self.mjwarp.step(self.wp_model, self.wp_data)

    def forward(self) -> None:
        with self.wp.ScopedDevice(self.device):
            if self._forward_graph is not None:
                self.wp.capture_launch(self._forward_graph)
            else:
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

    def _capture_step_graph(self, nstep: int) -> None:
        if nstep <= 0 or not self._can_capture_graphs():
            return
        try:
            with self.wp.ScopedDevice(self.device):
                with self.wp.ScopedCapture() as capture:
                    for _ in range(nstep):
                        self.mjwarp.step(self.wp_model, self.wp_data)
                self._step_graphs[nstep] = capture.graph
        except Exception as exc:
            print(f"[orca_rl.mjwarp] CUDA graph capture disabled for nstep={nstep}: {exc}")

    def _capture_forward_graph(self) -> None:
        if not self._can_capture_graphs():
            return
        try:
            with self.wp.ScopedDevice(self.device):
                with self.wp.ScopedCapture() as capture:
                    self.mjwarp.forward(self.wp_model, self.wp_data)
                self._forward_graph = capture.graph
        except Exception as exc:
            print(f"[orca_rl.mjwarp] forward CUDA graph capture disabled: {exc}")

    def _can_capture_graphs(self) -> bool:
        try:
            device = self.wp.get_device(self.device)
            return bool(device.is_cuda and self.wp.is_mempool_enabled(device))
        except Exception:
            return False
