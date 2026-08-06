#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="${XROBOT_SERVICE_DIR:-/opt/apps/roboticsservice}"
SERVICE_SCRIPT="${SERVICE_DIR}/runService.sh"

if [[ ! -f "${SERVICE_SCRIPT}" ]]; then
  echo "[PICO] XRoboToolkit PC service was not found: ${SERVICE_SCRIPT}" >&2
  echo "Install XRoboToolkit-PC-Service first, or set XROBOT_SERVICE_DIR=/path/to/roboticsservice." >&2
  exit 1
fi

if pgrep -f "RoboticsServiceProcess" >/dev/null 2>&1; then
  echo "[PICO] XRoboToolkit PC service is already running."
  exit 0
fi

echo "[PICO] Starting XRoboToolkit PC service from ${SERVICE_DIR}"
echo "[PICO] In the PICO XRoboToolkit app, connect to this host's LAN IPv4 address and enable upper-body/controller streaming."
cd "${SERVICE_DIR}"
exec bash "${SERVICE_SCRIPT}"
