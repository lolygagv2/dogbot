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

## Per-device manual steps (not scriptable)

- **Xbox controller** — pair over Bluetooth (interactive, one controller per robot).
- **Per-unit calibration** — `config/robot_profiles/treatbotN.yaml`: battery
  calibration factor, motor inversion, gimbal centers/limits. Seed from an
  existing profile and tune. See `.claude/resume_chat.md` bring-up entries.

## Verify (after step 6)

```
systemctl is-active treatbot.service                      # active
systemctl is-enabled wimz-audio.service                   # enabled
cat /sys/module/rtw88_core/parameters/disable_lps_deep     # Y
cat /etc/wimz/audio_state.json                             # volume JSON
curl -s localhost:8000/audio/volume                        # {"success":true,...}
```
