#!/usr/bin/env python3
"""
WIMZ Power Button Watcher

Manages soft power shutdown via Pololu Mini Pushbutton Power Switch, plus a
long-press "network cancel" escape hatch.

Hardware:
- GPIO21 (Pin 40) → relay module IN (isolates button from Pololu latch)
- GPIO20 (Pin 38) → button press sense (10kΩ external pull-down to GND)
- GPIO26 (Pin 37) → Pololu OFF kill pulse (handled by shutdown hook)

Relay Polarity (CRITICAL - edit per robot):
- Robot 1 (Teyleten optocoupler, active HIGH): active_high=True
- Robots 2-5 (5V SRD module, active LOW): active_high=False

Behavior:
1. On startup: Engages relay to isolate button from Pololu latch
2. Waits for GPIO20 to settle LOW
3. Classifies each press by hold time (100ms debounce against noise):
     0.1s <= hold < 5s  (released) -> graceful shutdown (systemd hook fires
                                       the GPIO26 kill pulse)
     hold reaches 5s               -> NETWORK CANCEL: leave the current WiFi
                                       and raise the WIMZ-xxxx AP so the app
                                       can reach the robot locally (captive
                                       portal / dead network escape). No
                                       shutdown.
     hold reaches 15s              -> ignored (stuck button guard)
   Shutdown is decided on RELEASE (we can't know it's a short press until the
   button comes up), so the ~100ms-after-press shutdown of the old watcher is
   now "shutdown when you let go".

Network cancel path (in order):
  a) POST http://127.0.0.1:8000/system/network-cancel — treatbot raises the
     AP sticky (same as the app's Local Mode) and plays the AP audio cue.
  b) treatbot not answering: touch /run/wimz/net-cancel, which treatbot's
     WiFi monitor consumes (raise AP sticky) on its next tick or next start.
  Nothing about the saved WiFi is modified or forgotten — this is a cancel,
  not a diagnosis. From the AP the app can pick another network or the user
  can send cloud_mode to rejoin the same one.

Installation (NOT OTA-managed — copy per unit, like the updater):
    sudo install -m755 services/power/wimz_power_button.py /usr/local/bin/
    sudo cp systemd/wimz-power-button.service /etc/systemd/system/
    sudo cp systemd/wimz-killpulse /lib/systemd/system-shutdown/
    sudo chmod +x /lib/systemd/system-shutdown/wimz-killpulse
    sudo systemctl daemon-reload
    sudo systemctl enable --now wimz-power-button.service
"""
import os
import subprocess
import sys
import time

# ===== EDIT THIS LINE PER ROBOT =====
# Robot 1 (Teyleten active-HIGH): active_high=True
# Robots 2-5 (SRD active-LOW): active_high=False
RELAY_ACTIVE_HIGH = False  # <-- Change per robot
# ====================================

DEBOUNCE_S = 0.1          # shorter than this = electrical noise
CANCEL_HOLD_S = 5.0       # hold this long = network cancel
STUCK_HOLD_S = 15.0       # hold this long = stuck button, ignore
POLL_S = 0.01

TREATBOT_API = "http://127.0.0.1:8000"
CANCEL_ENDPOINT = f"{TREATBOT_API}/system/network-cancel"
CANCEL_FLAG_DIR = "/run/wimz"
CANCEL_FLAG = f"{CANCEL_FLAG_DIR}/net-cancel"


def log(msg: str) -> None:
    print(f"WIMZ: {msg}", flush=True)


# ── Press classification (pure; unit-tested) ────────────────────────────

def classify_hold(hold_s: float, released: bool) -> str:
    """Map a hold duration to an action.

    released=True  -> the button came up after hold_s seconds.
    released=False -> the button is still down and hold_s has just been
                      reached (called at the CANCEL/STUCK thresholds).

    Returns one of: 'noise', 'shutdown', 'cancel', 'stuck', 'wait'.
    """
    if released:
        if hold_s < DEBOUNCE_S:
            return 'noise'
        if hold_s < CANCEL_HOLD_S:
            return 'shutdown'
        # Released after a cancel/stuck threshold: those already fired
        # (or were ignored) while held — nothing more to do on release.
        return 'wait'
    if hold_s >= STUCK_HOLD_S:
        return 'stuck'
    if hold_s >= CANCEL_HOLD_S:
        return 'cancel'
    return 'wait'


def track_press(is_active, now=time.time, sleep=time.sleep) -> str:
    """Follow one press from first HIGH to its terminal action.

    Blocks until the press resolves. Returns the action taken:
    'noise' | 'shutdown' | 'cancel' | 'stuck'.
    Always waits for release before returning so one hold can't fire twice.
    """
    start = now()
    fired = None
    while is_active():
        held = now() - start
        action = classify_hold(held, released=False)
        if action == 'cancel' and fired is None:
            fired = 'cancel'
            log(f"Long press ({held:.1f}s) — NETWORK CANCEL (no shutdown)")
            network_cancel()
        elif action == 'stuck' and fired is None:
            fired = 'stuck'
            log(f"Button held {held:.0f}s+ — treating as stuck, ignoring")
        sleep(POLL_S)

    held = now() - start
    if fired is not None:
        return fired
    action = classify_hold(held, released=True)
    if action == 'shutdown':
        log(f"Press released after {held:.2f}s — initiating graceful shutdown")
        shutdown()
    else:
        log("Transient on GPIO20 ignored")
    return action


# ── Actions ─────────────────────────────────────────────────────────────

def shutdown() -> None:
    subprocess.call(["sudo", "shutdown", "-h", "now"])


def network_cancel() -> bool:
    """Ask treatbot to raise the AP; fall back to a flag file it consumes."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "8", "-o", "/dev/null", "-w", "%{http_code}",
             "-X", "POST", CANCEL_ENDPOINT],
            capture_output=True, text=True, timeout=12)
        code = (r.stdout or "").strip()
        if r.returncode == 0 and code.startswith("2"):
            log("Network cancel accepted by treatbot")
            return True
        log(f"treatbot did not accept network cancel (curl rc={r.returncode}, http={code or '-'})")
    except Exception as e:
        log(f"treatbot unreachable for network cancel: {e}")

    # Fallback: leave a flag for treatbot's WiFi monitor. We deliberately do
    # NOT touch NetworkManager ourselves — if treatbot is down nobody would
    # raise the AP and we'd only have made things worse.
    try:
        os.makedirs(CANCEL_FLAG_DIR, exist_ok=True)
        with open(CANCEL_FLAG, "w") as f:
            f.write(f"{time.time():.0f}\n")
        os.chmod(CANCEL_FLAG, 0o644)
        log(f"Wrote {CANCEL_FLAG} — treatbot will raise the AP when it next checks")
    except Exception as e:
        log(f"Could not write {CANCEL_FLAG}: {e}")
    return False


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    from gpiozero import OutputDevice, DigitalInputDevice

    relay = OutputDevice(21, active_high=RELAY_ACTIVE_HIGH, initial_value=True)
    log("Relay engaged on GPIO21 — button isolated from Pololu")

    button_input = DigitalInputDevice(20, pull_up=False)

    log("Waiting for GPIO20 to settle LOW...")
    settle_timeout = 5.0
    settle_start = time.time()
    while button_input.is_active:
        if time.time() - settle_start > settle_timeout:
            log("ERROR — GPIO20 never settled LOW after 5s. Check wiring.")
            return 1
        time.sleep(0.05)

    log(f"GPIO20 settled LOW. Armed: release <{CANCEL_HOLD_S:.0f}s = shutdown, "
        f"hold {CANCEL_HOLD_S:.0f}s = network cancel (AP mode).")

    while True:
        button_input.wait_for_active()
        action = track_press(lambda: button_input.is_active)
        if action == 'shutdown':
            return 0
        # cancel/stuck/noise: re-arm for the next press


if __name__ == "__main__":
    sys.exit(main())
