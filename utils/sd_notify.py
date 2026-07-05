"""Minimal sd_notify — no-op when not running under systemd (NOTIFY_SOCKET unset).

Used for Type=notify readiness (READY=1), watchdog pings (WATCHDOG=1), and
shutdown signaling (STOPPING=1). Zero dependencies by design.
"""
import os
import socket


def sd_notify(state: str) -> None:
    """Send a state string to the systemd notification socket.

    Silently does nothing outside systemd (dev runs) or on socket errors —
    notification failure must never break the robot.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return  # dev run, not under systemd
    if addr.startswith("@"):
        addr = "\0" + addr[1:]  # abstract socket namespace
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(state.encode())
    except OSError:
        pass
