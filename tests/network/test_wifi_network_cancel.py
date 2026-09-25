"""Network cancel (power-button long press) — 2026-09-25.

A captive-portal WiFi (associated + DHCP, no usable internet) reads as
"connected" to the WiFi monitor, so no AP fallback ever fired and the robot
was unreachable (no relay, no local AP). The escape hatch is a 5 s hold on
the soft power button -> POST /system/network-cancel (or, if the API is
down, /run/wimz/net-cancel consumed by the WiFi monitor) -> the ONE AP,
raised sticky exactly like app Local Mode. Nothing about the saved WiFi
profile is modified — we often don't know WHY a network is failing.

Run: env_new/bin/python -m pytest tests/network/test_wifi_network_cancel.py -v
"""
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.network.wifi_manager import WiFiManager  # noqa: E402


def _quiet_manager(connected_ssid="BNYDC Guest", ap_up=False):
    wifi = WiFiManager()
    wifi._run_cmd = lambda *a, **k: (True, "")
    wifi._run_nmcli = lambda *a, **k: (True, "")
    wifi._sudo_rm = lambda *a, **k: True
    wifi.is_ap_mode = lambda: ap_up
    wifi.get_connection_status = lambda: {
        'connected': connected_ssid is not None, 'ssid': connected_ssid,
        'ip_address': '10.0.0.5', 'signal': 60, 'state': 'connected'}
    wifi.get_device_serial = lambda: "ab12"
    return wifi


def test_raise_local_ap_is_sticky_and_records_breadcrumb():
    wifi = _quiet_manager()
    started = []
    wifi.start_demo_hotspot = lambda ssid, password: started.append((ssid, password)) or True

    assert wifi.raise_local_ap(reason="network_cancel") is True
    assert started == [("WIMZ-ab12", wifi.AP_PASSWORD)]
    assert wifi.ap_deliberate is True
    assert wifi.cancelled_ssid == "BNYDC Guest"
    assert wifi.cancel_reason == "network_cancel"
    assert wifi.cancelled_at and time.time() - wifi.cancelled_at < 5


def test_raise_local_ap_never_modifies_saved_profiles():
    """No `nmcli connection modify/delete` — the cancel must not 'fix' WiFi."""
    wifi = _quiet_manager()
    nmcli_calls = []
    wifi._run_nmcli = lambda args, **k: nmcli_calls.append(args) or (True, "")
    wifi.start_demo_hotspot = lambda ssid, password: True

    wifi.raise_local_ap(reason="network_cancel")

    for args in nmcli_calls:
        assert not ({"modify", "delete"} & set(args)), args


def test_raise_local_ap_idempotent_when_ap_already_up():
    wifi = _quiet_manager(ap_up=True)
    wifi.start_demo_hotspot = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rebuild a live AP"))
    wifi.ap_deliberate = False

    assert wifi.raise_local_ap(reason="network_cancel") is True
    assert wifi.ap_deliberate is True


def test_raise_local_ap_failure_clears_stickiness():
    wifi = _quiet_manager()
    wifi.start_demo_hotspot = lambda *a, **k: False

    assert wifi.raise_local_ap(reason="network_cancel") is False
    assert wifi.ap_deliberate is False
    assert wifi.cancelled_ssid is None


def test_clear_cancel_breadcrumb():
    wifi = _quiet_manager()
    wifi.start_demo_hotspot = lambda *a, **k: True
    wifi.raise_local_ap()
    wifi.clear_cancel_breadcrumb()
    assert (wifi.cancelled_ssid, wifi.cancelled_at, wifi.cancel_reason) == (None, None, None)


# ── WiFi monitor flag consumption (main_treatbot._consume_net_cancel_flag) ──

class _Bot:
    """Just enough of TreatBotMain for the flag consumer."""

    def __init__(self):
        import logging
        self.logger = logging.getLogger("test")
        self._wifi_ap_active = False
        self._wifi_disconnected_since = None
        self.played = []
        self.usb_audio = type("A", (), {"play_file": lambda s, f: self.played.append(f)})()
        self.events = 0

    def _send_network_state_event(self, wifi):
        self.events += 1


def _consumer():
    """Bind the real method to the stub without importing hardware."""
    import ast
    src = (Path(__file__).resolve().parents[2] / "main_treatbot.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TreatBotMain")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_consume_net_cancel_flag")
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id.startswith("NET_CANCEL")}
    ns = {"os": os, "time": time}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "main_treatbot.py", "exec"), ns)
    return ns["_consume_net_cancel_flag"], consts


def test_flag_consumed_once_and_raises_sticky_ap(tmp_path):
    consume, consts = _consumer()
    flag = tmp_path / "net-cancel"
    flag.write_text("1\n")
    bot = _Bot()
    for k, v in consts.items():
        setattr(bot, k, v)
    bot.NET_CANCEL_FLAG = str(flag)
    wifi = _quiet_manager()
    wifi.start_demo_hotspot = lambda *a, **k: True

    assert consume(bot, wifi) is True
    assert wifi.ap_deliberate is True
    assert bot._wifi_ap_active is True
    assert bot.played == ["/wimz/ap_mode.mp3"]
    assert bot.events == 1
    # Consumed: unlinked (we own it here), and never acted on twice.
    assert not flag.exists()
    assert consume(bot, wifi) is False


def test_flag_consumed_once_even_if_unlink_fails(tmp_path):
    consume, consts = _consumer()
    flag = tmp_path / "net-cancel"
    flag.write_text("1\n")
    bot = _Bot()
    for k, v in consts.items():
        setattr(bot, k, v)
    bot.NET_CANCEL_FLAG = str(flag)
    wifi = _quiet_manager()
    wifi.start_demo_hotspot = lambda *a, **k: True

    with patch("os.unlink", side_effect=PermissionError):
        assert consume(bot, wifi) is True
        assert flag.exists()  # root-owned in production
        assert consume(bot, wifi) is False  # mtime memory


def test_stale_flag_ignored(tmp_path):
    consume, consts = _consumer()
    flag = tmp_path / "net-cancel"
    flag.write_text("1\n")
    old = time.time() - consts["NET_CANCEL_MAX_AGE"] - 60
    os.utime(flag, (old, old))
    bot = _Bot()
    for k, v in consts.items():
        setattr(bot, k, v)
    bot.NET_CANCEL_FLAG = str(flag)
    wifi = _quiet_manager()
    wifi.start_demo_hotspot = lambda *a, **k: (_ for _ in ()).throw(AssertionError("stale flag must not raise AP"))

    assert consume(bot, wifi) is False
    assert wifi.ap_deliberate is False


def test_no_flag_is_noop(tmp_path):
    consume, consts = _consumer()
    bot = _Bot()
    for k, v in consts.items():
        setattr(bot, k, v)
    bot.NET_CANCEL_FLAG = str(tmp_path / "absent")
    assert consume(bot, _quiet_manager()) is False


if __name__ == "__main__":
    import tempfile
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    fn(Path(tempfile.mkdtemp()))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
