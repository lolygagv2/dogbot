"""Power-button hold classification — long-press "network cancel", 2026-09-25.

The soft power button watcher (services/power/wimz_power_button.py, installed
by hand at /usr/local/bin/) used to shut down ~100 ms into any press. It now
classifies by hold time:
    release < 0.1 s  -> noise
    release < 5 s    -> graceful shutdown (fires on release)
    hold reaches 5 s -> NETWORK CANCEL (raise the AP via treatbot, no shutdown)
    hold reaches 15 s-> stuck button, ignored

No GPIO is touched: the module's gpiozero import lives inside main(), and
the press tracker takes injectable is_active/now/sleep.

Run: env_new/bin/python -m pytest tests/hardware/test_power_button_hold.py -v
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "wimz_power_button", ROOT / "services" / "power" / "wimz_power_button.py")
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)


class FakeClock:
    """Deterministic press: button is HIGH for `hold` seconds, then LOW."""

    def __init__(self, hold: float, step: float = 0.05):
        self.t = 0.0
        self.hold = hold
        self.step = step

    def now(self):
        return self.t

    def sleep(self, _):
        self.t += self.step

    def is_active(self):
        return self.t < self.hold


def _run_press(hold):
    clk = FakeClock(hold)
    calls = {"shutdown": 0, "cancel": 0}
    with patch.object(pb, "shutdown", lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)), \
         patch.object(pb, "network_cancel", lambda: calls.__setitem__("cancel", calls["cancel"] + 1) or True):
        action = pb.track_press(clk.is_active, now=clk.now, sleep=clk.sleep)
    return action, calls


def test_thresholds_are_what_the_docs_say():
    assert pb.DEBOUNCE_S == 0.1
    assert pb.CANCEL_HOLD_S == 5.0
    assert pb.STUCK_HOLD_S == 15.0


def test_noise_is_ignored():
    action, calls = _run_press(hold=0.04)
    assert action == "noise"
    assert calls == {"shutdown": 0, "cancel": 0}


def test_short_press_shuts_down_on_release():
    action, calls = _run_press(hold=0.8)
    assert action == "shutdown"
    assert calls == {"shutdown": 1, "cancel": 0}


def test_just_under_cancel_threshold_still_shuts_down():
    action, calls = _run_press(hold=4.9)
    assert action == "shutdown"
    assert calls["shutdown"] == 1 and calls["cancel"] == 0


def test_five_second_hold_is_network_cancel_not_shutdown():
    action, calls = _run_press(hold=7.0)
    assert action == "cancel"
    assert calls == {"shutdown": 0, "cancel": 1}


def test_cancel_fires_once_per_hold():
    """Holding well past 5 s must not re-trigger the cancel every poll."""
    action, calls = _run_press(hold=12.0)
    assert action == "cancel"
    assert calls["cancel"] == 1


def test_stuck_button_does_nothing():
    action, calls = _run_press(hold=20.0)
    # cancel fires at 5 s (the user may be holding deliberately); at 15 s we
    # only log. Crucially: never a shutdown, never a second cancel.
    assert action == "cancel"
    assert calls == {"shutdown": 0, "cancel": 1}


def test_classify_release_after_cancel_is_noop():
    assert pb.classify_hold(6.0, released=True) == "wait"
    assert pb.classify_hold(16.0, released=False) == "stuck"


def test_network_cancel_prefers_api(tmp_path):
    with patch.object(pb, "CANCEL_FLAG_DIR", str(tmp_path)), \
         patch.object(pb, "CANCEL_FLAG", str(tmp_path / "net-cancel")), \
         patch.object(pb.subprocess, "run",
                      lambda *a, **k: SimpleNamespace(returncode=0, stdout="202", stderr="")):
        assert pb.network_cancel() is True
    assert not (tmp_path / "net-cancel").exists()


def test_network_cancel_falls_back_to_flag_when_api_down(tmp_path):
    with patch.object(pb, "CANCEL_FLAG_DIR", str(tmp_path)), \
         patch.object(pb, "CANCEL_FLAG", str(tmp_path / "net-cancel")), \
         patch.object(pb.subprocess, "run",
                      lambda *a, **k: SimpleNamespace(returncode=7, stdout="000", stderr="")):
        assert pb.network_cancel() is False
    flag = tmp_path / "net-cancel"
    assert flag.exists()
    assert flag.read_text().strip().isdigit()


def test_network_cancel_never_touches_networkmanager(tmp_path):
    """The cancel is an escape hatch, not a diagnosis: no nmcli, ever."""
    seen = []

    def fake_run(cmd, *a, **k):
        seen.append(cmd[0])
        return SimpleNamespace(returncode=7, stdout="", stderr="")

    with patch.object(pb, "CANCEL_FLAG_DIR", str(tmp_path)), \
         patch.object(pb, "CANCEL_FLAG", str(tmp_path / "net-cancel")), \
         patch.object(pb.subprocess, "run", fake_run):
        pb.network_cancel()
    assert "nmcli" not in seen
    assert seen == ["curl"]


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                import tempfile
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
