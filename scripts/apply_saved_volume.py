#!/usr/bin/env python3
"""
apply_saved_volume.py - Boot-time volume restore for WIM-Z.

Run by wimz-audio.service (systemd oneshot) BEFORE the WIM-Z services start.
Reads /etc/wimz/audio_state.json and applies the saved volume to the USB
audio hardware mixer via amixer.

Standalone by design: stdlib only, runs under the system /usr/bin/python3 so
it does NOT depend on the project virtualenv being present or healthy at boot.
The path/default constants below must stay in sync with
services/media/volume_manager.py.
"""

import json
import subprocess
import sys
import time

STATE_FILE = "/etc/wimz/audio_state.json"
DEFAULT_VOLUME = 60
SPEAKER_CONTROLS = ("Speaker", "PCM", "Master")

# USB audio may enumerate a few seconds after boot -- retry detection.
CARD_DETECT_ATTEMPTS = 10
CARD_DETECT_DELAY_SEC = 1.0


def log(msg):
    print(f"[wimz-audio] {msg}", flush=True)


def detect_usb_audio_card():
    """Return the USB audio card number as a string, or None."""
    try:
        result = subprocess.run(
            ['aplay', '-l'], capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.split('\n'):
            if 'USB Audio' in line and 'card' in line:
                return line.split('card ')[1].split(':')[0].strip()
    except Exception as e:
        log(f"card detection error: {e}")
    return None


def read_saved_volume():
    """Read the saved volume from the JSON file, or DEFAULT_VOLUME if missing."""
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        vol = int(data.get('volume', DEFAULT_VOLUME))
        vol = max(0, min(100, vol))
        log(f"read {STATE_FILE}: volume={vol}%")
        return vol
    except FileNotFoundError:
        log(f"{STATE_FILE} missing -- using default {DEFAULT_VOLUME}%")
        return DEFAULT_VOLUME
    except Exception as e:
        log(f"could not read {STATE_FILE} ({e}) -- using default {DEFAULT_VOLUME}%")
        return DEFAULT_VOLUME


def apply_volume(card, volume):
    """Apply volume via amixer; try each candidate control. Returns True/False."""
    for control in SPEAKER_CONTROLS:
        try:
            result = subprocess.run(
                ['amixer', '-c', str(card), 'sset', control, f'{volume}%'],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0:
                log(f"applied {volume}% via amixer (card {card}, control '{control}')")
                return True
        except Exception as e:
            log(f"amixer '{control}' failed: {e}")
    return False


def main():
    volume = read_saved_volume()

    card = None
    for attempt in range(1, CARD_DETECT_ATTEMPTS + 1):
        card = detect_usb_audio_card()
        if card is not None:
            break
        log(f"USB audio card not found (attempt {attempt}/{CARD_DETECT_ATTEMPTS})")
        time.sleep(CARD_DETECT_DELAY_SEC)

    if card is None:
        # Non-fatal: don't block boot. WIM-Z will re-apply once it starts.
        log("no USB audio card detected -- skipping (WIM-Z will apply on startup)")
        return 0

    apply_volume(card, volume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
