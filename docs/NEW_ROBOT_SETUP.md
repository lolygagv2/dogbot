# WIM-Z — New Robot Setup

How a freshly-cloned robot reaches a fully-working state. Everything transfers
through three channels — there is nothing hand-configured that isn't captured
by one of them:

| What transfers | How |
|---|---|
| Application code (all features) | `git pull` |
| OS identity — hostname, machine-id, SSH keys, relay credentials | `scripts/wimz_rebadge.sh` |
| System config — volume service, WiFi fix, `/etc/wimz` | `scripts/setup_device.sh` |

The fleet is built by **cloning an existing robot's SD card**, so the Python
virtualenv (`env_new/` — aiortc, psutil, pygame, opencv, …) comes with the
clone. No `pip install` step is required.

## Procedure

Clone an existing robot's SD card, boot the new unit, then on it:

1. **Identity** — reset OS identity and relay credentials:
   ```
   sudo bash /home/morgan/dogbot/scripts/wimz_rebadge.sh <N>
   ```
   `<N>` = unit number. Capture the printed `DEVICE_SECRET`.
2. **Register** `DEVICE_ID` + `DEVICE_SECRET` on the Lightsail relay
   (allowed-devices list) — the robot cannot connect remotely until this is done.
3. `sudo reboot`
4. **Latest code**:
   ```
   cd /home/morgan/dogbot && git pull
   ```
5. **System config** — applies what `git pull` cannot:
   ```
   sudo bash scripts/setup_device.sh
   ```
6. `sudo reboot` — activates the WiFi driver change and loads the new code.

## What `setup_device.sh` installs

Idempotent — safe to re-run.

- `/etc/wimz/` — directory for persistent volume state (`audio_state.json`).
- `wimz-audio.service` — systemd boot service that restores saved volume
  before WIM-Z starts.
- `/etc/modprobe.d/rtw88.conf` — disables the rtw88 USB-WiFi deep power-save
  that causes beacon-loss disconnects.
- NetworkManager `wifi.powersave 2` on all saved connections.
- `dtoverlay=spi0-1cs` in `/boot/firmware/config.txt` — frees GPIO7 (pin 26)
  for the treat through-beam sensor (NeoPixel keeps CE0). Harmless on units
  without the sensor. Needs a power cycle to take effect.

## Per-device manual steps (not scriptable)

- **Xbox controller** — pair over Bluetooth (interactive, one controller per robot).
- **Per-unit calibration** — `config/robot_profiles/treatbotN.yaml`: battery
  calibration factor, motor inversion, gimbal centers/limits. Seed from an
  existing profile and tune. See `.claude/resume_chat.md` bring-up entries.
- **Treat through-beam sensor** — see the section below; hardware install +
  alignment, then one config flip.

## Treat through-beam sensor (per-unit hardware install) {#beam}

Confirms treats physically leave the chute (spec `dispensed_confirmed` +
beam-counted `dispensed_count`; see DISPENSE-VERIFY in
`services/reward/dispenser.py`). Reference install: treatbot5, 2026-07-07
(commits `a052c1e`, `6778432`). First done on treatbot5 — copy that unit.

**Parts:** 5 mm 850 nm IR through-beam pair (2-wire emitter, 3-wire NPN
receiver), 220 Ω (emitter), **1 kΩ (receiver pull-up — required, the Pi's
internal ~50 kΩ is too weak and the line sticks LOW)**.

**Wiring (robot powered off):**

| Wire | Pi header pin |
|---|---|
| Emitter red, through 220 Ω | pin 2 (5 V) |
| Emitter black | pin 6 (GND) |
| Receiver red | pin 4 (5 V) |
| Receiver black | pin 25 (GND) |
| Receiver white (OUT, NPN) | pin 26 (GPIO7) |
| 1 kΩ resistor | between OUT line and pin 17 (**3.3 V — NEVER 5 V**) |

Both sensors run on **5 V** (the "3-5V" marketing is wrong — undervolted, the
output latches LOW). The NPN open-collector output + 3.3 V pull-up keeps the
GPIO Pi-safe.

**Prerequisite:** `setup_device.sh` has run and the unit has been power
cycled, so GPIO7 is free (`pinctrl get 7` shows `ip`/`no`, not `op`).

**Alignment (the #1 failure mode — <10° receive cone):** mount emitter and
receiver coaxial across the drop path, watching live:
`watch -n 0.2 pinctrl get 7` — **`hi` = beam locked (good), `lo` = no beam.**
Adjust until it idles solidly `hi`; a card swiped through the gap must flip
it `lo` and back. Debugging notes: fingertips pass bright IR (use a card),
never wrap sensors in foil (conductive — shorts the output).

**Enable:** in the unit's `config/robot_profiles/treatbotN.yaml` dispenser
section set `beam_enabled: true`, `beam_pin: 7`, `beam_active_low: true`,
`beam_timeout_s: 4.0` (copy the treatbot5 block), restart
`treatbot.service`, and confirm the journal logs
`Through-beam sensor armed on GPIO7 (active_low)`. Verify end-to-end with a
dispense: `dispense_log.dispensed_count` should be ≥1.

## Verify (after step 6)

```
systemctl is-active treatbot.service                      # active
systemctl is-enabled wimz-audio.service                   # enabled
cat /sys/module/rtw88_core/parameters/disable_lps_deep     # Y
cat /etc/wimz/audio_state.json                             # volume JSON
curl -s localhost:8000/audio/volume                        # {"success":true,...}
```
