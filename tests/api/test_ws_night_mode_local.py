#!/usr/bin/env python3
"""Night-vision setting must work over /ws/local (AP mode) — 2026-10-24.

Morgan: settings -> night vision -> "Daytime" stayed stuck on night vision and
showed "waiting for the robot" while the robot was clearly online (AP mode).
Cause: /ws/local had no set_night_mode_override command ("Unknown command") and
night_mode_state was only ever pushed over the relay, so a local-mode app never
heard the robot's state. Now the local socket sends the state on connect,
answers set/pull commands, and the controller pushes to both transports.

No camera: the controller's detector handle is None, so transitions log
"camera not available" and only flip state. State file redirected to a temp dir.

Run: /home/morgan/dogbot/env_new/bin/python tests/api/test_ws_night_mode_local.py
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import modes.night_mode_controller as nmc  # noqa: E402
import api.ws as ws  # noqa: E402

_tmp = Path(tempfile.mkdtemp(prefix="nm_test_"))
nmc.STATE_FILE = _tmp / "night_mode.json"
nmc.LOG_FILE = _tmp / "night_mode.log"

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class FakeWS:
    def __init__(self):
        self.sent = []
        self.url = type("U", (), {"path": "/ws/local"})()

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    def events(self, name):
        return [m for m in self.sent if m.get("event") == name]

    def responses(self):
        return [m["data"] for m in self.sent if m.get("type") == "command_response"]


def test_normalize():
    n = nmc.normalize_override
    check(n("Daytime") == "force_day" and n("day") == "force_day", "normalize: day/Daytime -> force_day")
    check(n("night vision") == "force_night" and n("force_night") == "force_night", "normalize: night spellings -> force_night")
    check(n("AUTO") == "auto" and n("bogus") is None and n(None) is None, "normalize: auto ok, junk -> None")


async def test_local_ws():
    srv = ws.get_websocket_server()   # the singleton the controller pushes to
    srv._main_loop = asyncio.get_running_loop()
    sock = FakeWS()
    srv.manager.active_connections.add(sock)

    nm = nmc.get_night_mode_controller()
    nm._detector = None
    nm.override_mode = "auto"
    nm.current_mode = "night"            # the stuck state Morgan saw

    # 1. connect sync: initial_status carries night_mode and a night_mode_state event follows
    await srv._send_initial_status(sock)
    init = [m for m in sock.sent if m.get("type") == "initial_status"]
    check(init and init[0]["data"].get("night_mode", {}).get("mode") == "night",
          "initial_status includes night_mode")
    ev = sock.events("night_mode_state")
    check(ev and ev[-1]["reason"] == "connect sync" and ev[-1]["mode"] == "night",
          "night_mode_state pushed on connect")

    # 2. the app's "Daytime" choice over the local socket
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "set_night_mode_override", "override": "force_day"})
    resp = sock.responses()
    check(resp and resp[-1]["success"] and resp[-1]["override"] == "force_day",
          "set_night_mode_override force_day -> success response")
    # transition runs on a worker thread; its _push_state must land on this socket
    for _ in range(50):
        if nm.current_mode == "day" and sock.events("night_mode_state"):
            break
        await asyncio.sleep(0.1)
    check(nm.current_mode == "day" and nm.override_mode == "force_day", "controller flipped to day/force_day")
    ev = sock.events("night_mode_state")
    check(ev and ev[-1]["mode"] == "day" and ev[-1]["override"] == "force_day",
          "night_mode_state(day) broadcast to local WS from the worker thread")

    # 3. UI spelling alias + unchanged override still echoes state (no 'waiting' hang)
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "set_night_mode_override", "mode": "daytime"})
    await asyncio.sleep(0.2)
    resp = sock.responses()
    check(resp and resp[-1]["success"] and resp[-1]["override"] == "force_day", "alias 'daytime' accepted")
    check(bool(sock.events("night_mode_state")), "unchanged override still echoes night_mode_state")

    # 4. pull
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "night_mode_status"})
    resp = sock.responses()
    check(resp and resp[-1]["night_mode"]["mode"] == "day", "night_mode_status response carries state")
    check(sock.events("night_mode_state") and sock.events("night_mode_state")[-1]["reason"] == "status_pull",
          "night_mode_status also emits night_mode_state event")

    # 5. junk rejected with the current state attached
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "set_night_mode_override", "override": "sepia"})
    resp = sock.responses()
    check(resp and not resp[-1]["success"] and "night_mode" in resp[-1], "invalid override -> success:false + state")

    # 6. back to auto
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "set_night_mode_override", "override": "auto"})
    await asyncio.sleep(0.2)
    check(nm.override_mode == "auto" and sock.events("night_mode_state")[-1]["override"] == "auto",
          "auto restores and echoes")
    check(json.loads(nmc.STATE_FILE.read_text())["override_mode"] == "auto", "override persisted to state file")


if __name__ == "__main__":
    test_normalize()
    asyncio.run(test_local_ws())
    print()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
