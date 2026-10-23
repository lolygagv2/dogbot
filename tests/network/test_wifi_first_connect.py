#!/usr/bin/env python3
"""First-time WiFi connect at a new location — 2026-10-23.

Journal of the 12:01 boot at a new location: wifi-provision tried all 10 saved
profiles blind (`nmcli connection up`, 10 s timeout each = 110 s of "Command
timed out" ERRORs) before raising the setup AP, then the portal's /connect ran
save_credentials() inline — which tears the AP down first, so the phone never
got the response (page: "Connection failed" with a correct password).

Checks (no nmcli, no hotspot — everything is patched):
  - try_connect_known scans first and only brings up saved profiles that are
    in range (hidden profiles always), returning False fast when none are.
  - an empty scan falls back to trying every profile (NM-not-ready safety).
  - connect_in_background returns at once, records the outcome, calls
    on_success / restart_ap, and refuses a second concurrent connect.
  - save_credentials no longer deletes a profile that does not exist.

Run: /home/morgan/dogbot/env_new/bin/python tests/network/test_wifi_first_connect.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.network.wifi_manager import WiFiManager  # noqa: E402

SAVED = [{'name': n, 'type': 'wifi', 'device': ''}
         for n in ('Allforthewag', 'Estrella', 'HomeNet', 'Secret')]
PROFILE_SSID = {'Allforthewag': 'Allforthewag', 'Estrella': 'Estrella',
                'HomeNet': 'Home Net 5G', 'Secret': 'Secret'}
HIDDEN = {'Secret'}


def _manager(scan, connected_after_up=None):
    wifi = WiFiManager()
    calls = []

    def run_nmcli(args, timeout=30):
        calls.append(list(args))
        if args[:1] == ["-g"]:
            name = args[-1]
            return True, f"{PROFILE_SSID[name]}\n{'yes' if name in HIDDEN else 'no'}\n"
        if args[:2] == ["connection", "up"]:
            if connected_after_up and args[2] == connected_after_up:
                wifi._now_connected = True
                return True, ""
            return False, "Command timed out"
        if args[:2] == ["device", "connect"]:
            return False, "Error: No network with SSID could be found."
        return True, ""

    wifi._now_connected = False
    wifi._run_nmcli = run_nmcli
    wifi._run_cmd = lambda *a, **k: (True, "")
    wifi._do_scan = lambda: scan
    wifi.get_saved_connections = lambda: SAVED
    wifi.get_connection_status = lambda: {
        'connected': wifi._now_connected, 'ssid': 'x' if wifi._now_connected else None,
        'ip_address': '10.0.0.5' if wifi._now_connected else None,
        'signal': 60, 'state': 'connected' if wifi._now_connected else 'disconnected'}
    return wifi, calls


def _ups(calls):
    return [c[2] for c in calls if c[:2] == ["connection", "up"]]


def test_none_in_range_skips_every_visible_profile():
    # Only the hidden profile is worth a blind attempt; the three broadcast
    # SSIDs that the scan did not see are never brought up.
    wifi, calls = _manager(scan=[{'ssid': 'Neighbour', 'signal': 70}])
    assert wifi.try_connect_known(timeout=15) is False
    assert _ups(calls) == ['Secret'], _ups(calls)


def test_none_in_range_and_no_hidden_returns_fast_without_connection_up():
    wifi, calls = _manager(scan=[{'ssid': 'Neighbour', 'signal': 70}])
    wifi.get_saved_connections = lambda: [c for c in SAVED if c['name'] != 'Secret']
    t0 = time.monotonic()
    assert wifi.try_connect_known(timeout=15) is False
    assert time.monotonic() - t0 < 2.0
    assert _ups(calls) == [], _ups(calls)
    assert not any(c[:2] == ["device", "connect"] for c in calls)


def test_only_visible_and_hidden_profiles_are_tried_strongest_first():
    wifi, calls = _manager(
        scan=[{'ssid': 'Estrella', 'signal': 40}, {'ssid': 'Home Net 5G', 'signal': 80}],
        connected_after_up='HomeNet')
    assert wifi.try_connect_known(timeout=15) is True
    # HomeNet (ssid != profile name, strongest) first; Allforthewag never tried.
    assert _ups(calls) == ['HomeNet'], _ups(calls)


def test_hidden_profile_is_attempted_even_when_not_in_scan():
    wifi, calls = _manager(scan=[{'ssid': 'Estrella', 'signal': 40}])
    assert wifi.try_connect_known(timeout=15) is False
    ups = _ups(calls)
    assert 'Secret' in ups and 'Estrella' in ups and 'Allforthewag' not in ups, ups


def test_empty_scan_falls_back_to_every_profile():
    wifi, calls = _manager(scan=[])
    wifi.try_connect_known(timeout=15)
    assert set(_ups(calls)) == set(PROFILE_SSID), _ups(calls)


def test_connect_in_background_failure_restarts_ap_and_records_job():
    wifi = WiFiManager()
    wifi.CONNECT_RESPONSE_GRACE_S = 0
    wifi.save_credentials = lambda ssid, pw: {
        "success": False, "message": "Wrong password. Please try again.",
        "has_internet": False, "should_restart_ap": True}
    restarted, ok_calls = [], []
    wifi.start_hotspot = lambda ssid, pw: restarted.append(ssid) or True
    wifi.get_ap_ssid = lambda: "WIMZ-TEST"

    started, job = wifi.connect_in_background("HomeNet", "hunter2",
                                              on_success=lambda s, r: ok_calls.append(s))
    assert started and job['state'] == 'connecting'
    for _ in range(100):
        if wifi.get_connect_job()['state'] == 'done':
            break
        time.sleep(0.02)
    job = wifi.get_connect_job()
    assert job['state'] == 'done' and job['success'] is False
    assert job['message'].startswith("Wrong password"), job
    assert job['ap_restarted'] is True and restarted == ["WIMZ-TEST"]
    assert ok_calls == []


def test_connect_in_background_success_calls_back_and_rejects_concurrent():
    wifi = WiFiManager()
    wifi.CONNECT_RESPONSE_GRACE_S = 0.2
    wifi.save_credentials = lambda ssid, pw: {
        "success": True, "message": f"Connected to {ssid} with internet access!",
        "has_internet": True, "should_restart_ap": False}
    got = []
    started, _ = wifi.connect_in_background("HomeNet", "pw", on_success=lambda s, r: got.append((s, r)))
    assert started
    again, job = wifi.connect_in_background("Other", "pw")
    assert again is False and job['ssid'] == "HomeNet"
    for _ in range(100):
        if wifi.get_connect_job()['state'] == 'done':
            break
        time.sleep(0.02)
    job = wifi.get_connect_job()
    assert job['success'] is True and job['has_internet'] is True
    assert got and got[0][0] == "HomeNet"


def test_save_credentials_skips_delete_when_profile_absent():
    wifi = WiFiManager()
    calls = []
    wifi.stop_hotspot = lambda: True
    wifi._wait_for_ssid_in_scan = lambda ssid, timeout=30: True
    wifi.get_saved_connections = lambda: [{'name': 'Other'}]
    wifi._run_nmcli = lambda args, timeout=30: (calls.append(list(args)) or (True, ""))
    wifi.get_connection_status = lambda: {'connected': True, 'ssid': 'New', 'ip_address': '1.2.3.4'}
    wifi._ssid_has_5ghz = lambda ssid: False
    wifi.check_internet = lambda timeout=5: True
    r = wifi.save_credentials("New", "pw")
    assert r["success"] is True
    assert not any(c[:2] == ["connection", "delete"] for c in calls), calls
    # ...and still deletes when the profile exists
    calls.clear()
    wifi.get_saved_connections = lambda: [{'name': 'New'}]
    wifi.save_credentials("New", "pw")
    assert any(c[:2] == ["connection", "delete"] for c in calls), calls


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    sys.exit(1 if failed else 0)
