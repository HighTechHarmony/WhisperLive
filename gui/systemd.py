"""Systemd helpers used by the GUI."""

from __future__ import annotations

import subprocess


def service_is_active(unit: str, runner=None) -> bool:
    runner = runner or subprocess.run
    try:
        result = runner(
            ["systemctl", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "active"


def start_service(unit: str, runner=None) -> tuple[bool, str]:
    runner = runner or subprocess.run
    try:
        result = runner(
            ["systemctl", "start", unit],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()
    return True, ""