from __future__ import annotations

from pathlib import Path
import socket
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def apply_remote_override(task_cfg: dict[str, Any], remote: str | None) -> None:
    if not remote:
        return
    addresses = [address.strip() for address in remote.split(",") if address.strip()]
    if len(addresses) != 1:
        raise ValueError("HEFT playback expects one --remote host:port address.")
    task_cfg["orcagym_addresses"] = addresses


def check_orcagym_addresses(task_cfg: dict[str, Any], timeout_s: float = 2.0) -> None:
    addresses = task_cfg.get("orcagym_addresses") or []
    if len(addresses) != 1:
        raise ValueError("HEFT playback requires exactly one OrcaGym address.")
    address = str(addresses[0])
    host, port = _split_host_port(address)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Cannot connect to OrcaGym at {address}. Start OrcaLab/OrcaStudio or pass --remote host:port. "
            f"Original error: {exc}"
        ) from exc


def _split_host_port(address: str) -> tuple[str, int]:
    if address.count(":") != 1:
        raise ValueError(f"Expected host:port, got {address!r}.")
    host, port_text = address.rsplit(":", 1)
    if not host:
        raise ValueError(f"Missing host in OrcaGym address {address!r}.")
    return host, int(port_text)


def explain_missing_runtime_dependency(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "HEFT runtime dependencies are missing. Run ./install_heft.sh first. "
        f"Original import error: {exc}"
    )
