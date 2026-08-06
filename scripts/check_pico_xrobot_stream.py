#!/usr/bin/env python3
"""Report whether the local XRoboToolkit service is receiving PICO data."""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PICO data reaching the local XRoboToolkit PC service.")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    import xrobotoolkit_sdk as xrt

    xrt.init()
    seen_headset = False
    seen_body = False
    seen_trackers = False
    try:
        deadline = time.monotonic() + max(float(args.seconds), 0.1)
        while time.monotonic() < deadline:
            headset = np.asarray(xrt.get_headset_pose(), dtype=np.float64).reshape(-1)
            body = bool(xrt.is_body_data_available())
            tracker_count = int(xrt.num_motion_data_available())
            headset_valid = bool(headset.size >= 7 and np.all(np.isfinite(headset)) and np.linalg.norm(headset) > 1e-6)
            seen_headset |= headset_valid
            seen_body |= body
            seen_trackers |= tracker_count > 0
            print(
                f"headset={'yes' if headset_valid else 'no'} "
                f"body={'yes' if body else 'no'} trackers={tracker_count}",
                flush=True,
            )
            time.sleep(1.0)
    finally:
        xrt.close()

    if not seen_headset and not seen_body and not seen_trackers:
        raise SystemExit(
            "No PICO data reached XRoboToolkit PC service. Check the PICO App host LAN IP, "
            "connection state, and Upper body/Controller tracking toggles."
        )
    print(
        "PICO stream detected: "
        f"headset={seen_headset} body={seen_body} trackers={seen_trackers}"
    )


if __name__ == "__main__":
    main()
