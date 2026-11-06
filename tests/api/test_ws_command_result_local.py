#!/usr/bin/env python3
"""/ws/local parity for CONTRACT v2 commands + settings (2026-11-06).

A command frame with command_id on the local socket is run by the
ToolDispatcher and answered with command_result on the same socket (plus
the usual command_response); frames without command_id keep the legacy
path. settings_pull returns the remote settings and pushes settings_state;
initial_status carries remote_settings; mark_departure emits departure
(method=app). Same lesson as the 2026-10-24 night-mode bug: anything added
on the relay path must exist locally too.

Run: /home/morgan/dogbot/env_new/bin/python tests/api/test_ws_command_result_local.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='wscmd_')

import api.ws as ws  # noqa: E402
from services.cloud.tool_dispatcher import get_tool_dispatcher, ToolSpec  # noqa: E402
from services.cloud.remote_settings import get_remote_settings  # noqa: E402

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

    def of_type(self, t):
        return [m for m in self.sent if m.get("type") == t]

    def events(self, name):
        return [m for m in self.sent if m.get("event") == name]


class FakeStore:
    kv = {}
    def set_setting(self, k, v): self.kv[k] = v
    def get_setting(self, k, default=None): return self.kv.get(k, default)


async def main():
    srv = ws.get_websocket_server()
    srv._main_loop = asyncio.get_running_loop()
    sock = FakeWS()
    srv.manager.active_connections.add(sock)

    rs = get_remote_settings()
    rs._store = FakeStore()
    from services.reward.treat_policy import get_treat_policy
    get_treat_policy()._get_dispenser = lambda: None
    rs.apply({'treat_schedule': {'daily_cap': 9}}, source='test')

    disp = get_tool_dispatcher()
    disp.register(ToolSpec('echo_test', lambda a, c: {'ok': True, 'got': a.get('x')}, 5.0))

    # initial_status carries remote_settings + settings_state event
    await srv._send_initial_status(sock)
    init = sock.of_type("initial_status")
    check(init and init[0]["data"].get("remote_settings", {}).get("treat_schedule", {}).get("daily_cap") == 9,
          "initial_status includes remote_settings")
    check(sock.events("settings_state") and sock.events("settings_state")[-1]["reason"] == "connect sync",
          "settings_state pushed on connect")

    # command with command_id -> command_result
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "echo_test", "command_id": "c-77", "x": 5})
    for _ in range(40):
        if sock.of_type("command_result"):
            break
        await asyncio.sleep(0.05)
    cr = sock.of_type("command_result")
    check(cr and cr[0]["command_id"] == "c-77" and cr[0]["ok"] is True and cr[0]["result"]["got"] == 5,
          f"command_id frame -> command_result on the local socket ({cr[:1]})")
    resp = sock.of_type("command_response")
    check(resp and resp[0]["data"].get("accepted") is True, "command_response acknowledges acceptance")

    # unknown command with command_id but not registered -> legacy Unknown command
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "no_such_cmd", "command_id": "c-78"})
    resp = sock.of_type("command_response")
    check(resp and resp[0]["data"].get("success") is False and "Unknown command" in resp[0]["data"].get("error", ""),
          "unregistered command stays on the legacy path")

    # settings_pull
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "settings_pull"})
    resp = sock.of_type("command_response")
    check(resp and resp[0]["data"]["settings"]["treat_schedule"]["daily_cap"] == 9
          and sock.events("settings_state") and sock.events("settings_state")[-1]["reason"] == "status_pull",
          "settings_pull answers + pushes settings_state")

    # settings apply over local WS
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "settings", "settings": {"treat_schedule": {"daily_cap": 4}}})
    resp = sock.of_type("command_response")
    check(resp and resp[0]["data"]["success"] and rs.get()["treat_schedule"]["daily_cap"] == 4, "settings applied over /ws/local")

    # mark_departure -> departure event broadcast locally (method app)
    sock.sent.clear()
    await srv._execute_contract_command(sock, {"command": "mark_departure"})
    await asyncio.sleep(0.2)
    resp = sock.of_type("command_response")
    check(resp and resp[0]["data"]["success"] and resp[0]["data"]["event"]["data"]["method"] == "app",
          "mark_departure emits departure method=app")
    check(sock.events("departure"), "departure broadcast to the local socket")


if __name__ == "__main__":
    asyncio.run(main())
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
