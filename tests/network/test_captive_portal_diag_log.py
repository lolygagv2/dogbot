#!/usr/bin/env python3
"""
Regression test: captive-portal diagnostic log + password-reveal markup.

Runs the portal app on a localhost port with a mocked WiFiManager (no hotspot,
no nmcli). Checks:
  - setup.html serves the password-reveal toggle and the diagnostic log panel
  - a failed /connect lands the reason (wifi_manager's raw error + the portal's
    summary) in GET /log
  - GET /log?since=<seq> returns only newer records

Run: /home/morgan/dogbot/env_new/bin/python tests/network/test_captive_portal_diag_log.py
"""
import sys, logging, threading, time, json, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO)

import uvicorn
from services.network.captive_portal import CaptivePortal

PORT = 18099
BASE = f"http://127.0.0.1:{PORT}"


class FakeWM:
    AP_PASSWORD = "x"

    def save_credentials(self, ssid, pw):
        logging.getLogger("services.network.wifi_manager").error(
            "Failed to connect: Error: Connection activation failed: (7) Secrets were required, but not provided.")
        return {"success": False, "message": "Wrong password. Please try again.",
                "has_internet": False, "should_restart_ap": False}

    def get_connection_status(self):
        return {}


def get(path):
    return urllib.request.urlopen(BASE + path).read().decode()


def main():
    cp = CaptivePortal(FakeWM())
    srv = uvicorn.Server(uvicorn.Config(cp.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    time.sleep(1.5)
    try:
        html = get("/")
        assert 'id="toggle-password"' in html, "password reveal toggle missing"
        assert 'id="diag-body"' in html, "diagnostic log panel missing"

        req = urllib.request.Request(BASE + "/connect",
                                     data=json.dumps({"ssid": "HomeNet", "password": "hunter2"}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req).read().decode())
        assert r["success"] is False

        log = json.loads(get("/log"))
        msgs = [e["message"] for e in log["entries"]]
        assert any("Secrets were required" in m for m in msgs), msgs
        assert any("Connect to 'HomeNet' failed" in m for m in msgs), msgs
        assert any("password length 7" in m for m in msgs), msgs
        assert any(e["level"] == "ERROR" for e in log["entries"])

        newer = json.loads(get(f"/log?since={log['latest_seq']}"))["entries"]
        assert newer == [], newer
        print("PASS: captive portal diag log + password reveal")
    finally:
        srv.should_exit = True


if __name__ == "__main__":
    main()
