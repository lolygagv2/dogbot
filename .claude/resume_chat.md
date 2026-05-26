# WIM-Z Resume Chat Log

## Session: 2026-05-25 (part 5) — TMC2209 UART FIXED on treatbot4

**Duration:** ~1.5 hours
**Robot:** treatbot4
**Status:** ✅ UART working — `TMC2209 detected: version=0x21` after 5+ wire iterations

### What was actually wrong
Four separate physical wiring issues, all needed at once:
1. **VIO** wire missing — added 3.3V from Pi pin 1
2. **GND** wire missing — added Pi pin 6 → TMC2209 logic-side GND (not motor GND)
3. **1K resistor on wrong leg** — was on RX (pin 10), moved to TX (pin 8). User's resistor is ~700Ω, close enough.
4. **PDN_UART is at chip pin 4** (textbook StepStick) — earlier in the session I incorrectly inferred pin 5 from a 0V-idle voltage reading and sent user on a wild goose chase. Pin 4 was right all along.

### Verification after fix
`TMC2209 configured: IRUN=31, IHOLD=5, 8x microstep, vsense=0, SGTHRS=0, chopper=spreadcycle` — all yaml settings actually taking effect now.

### Code bug found and fixed
`config/config_loader.py::DispenserConfig` lacked a `chopper_mode` @property and had no `__getattr__` fallback. So `getattr(dispenser, 'chopper_mode', None)` always returned None, and yaml's `chopper_mode: "spreadcycle"` silently fell through to "stealthchop". Added explicit @property. Without this fix the spreadcycle setting would have been silently ignored even after UART came up.

### Yaml time bomb defused
treatbot4.yaml had `microstepping: 4` set during the months UART was broken (had no effect — chip was on hardware default 8x). With UART now working, that setting would have caused 2× over-rotation per dispense. Reverted to `microstepping: 8` in same commit as the UART fix.

### Diagnostic mistakes I made this session — for next session's reference
- Multiple incorrect theories proposed: "PDN HIGH = power-down" (datasheet says UART works regardless), "pin 5 = PDN_UART" (was pin 4), "VIO alone fixes it" (needed 3 more changes too). User correctly called out my batting average. Real fixes were the four wiring items; I should have walked through the full list earlier rather than offering them one-at-a-time as new theories.

### For the user's next moves
1. **Apply same 4-item check to treatbot3 and treatbot5** — likely same wiring pattern, same fix
2. **Before bringing UART up on each:** check yaml `microstepping × steps_per_slot` is consistent (treatbot2 baseline = 8 × 137 = 30.8°/slot)
3. **Test the dispenser** — spreadcycle is now on; expect motor whine + significantly more torque (irun=31 vs the ~16 default that was running before)
4. Vref pot tuning is now optional (only needed if irun=31 + spreadcycle still isn't enough — unlikely)

### Commits this session
(pending) yaml time-bomb fix + chopper_mode config_loader property + this session log

---

## Session: 2026-05-25 (part 3) — treatbot5 pull/conflict, Xbox controller swap, gimbal re-center

**Duration:** ~1.5 hours
**Robot:** treatbot5
**Status:** ✅ Complete — committed as `005d78b`, pushed to origin/main

### Work completed
1. **Pull origin/main with conflict** — rebase against `b4f21f3` (later `191ba0c`) hit a conflict in `.claude/resume_chat.md` where the local treatbot5-setup session log overlapped with the incoming treatbot4 night-mode log. Resolved by keeping both entries in newest-first order. New code on this Pi after pull: night mode controller (`5f0dacc`), mood_led LedService routing (`e1f3de8`), tilt_min/dispenser tweaks (treatbot4-only), and a later `feat: night mode kills blue tube` (`191ba0c`).
2. **TMC2209 UART verified on treatbot5** — all OS-side steps from `.claude/TMC2209_UART_SETUP.md` pass (dtparam, cmdline, dialout, /dev/ttyAMA0 perms, SupplementaryGroups). Chip-level probe across all 4 slave addresses returned **echo only** (Pi sees its own TX loopback) — same symptom as treatbot4. Root cause is the same VIO logic-supply wire being disconnected (per part-2 session diagnosis). Dispenser runs on hardware defaults (Vref pot + MS1/MS2 strapping) and is fine for now.
3. **Xbox controller swap on treatbot5** — removed old `AC:8E:BD:4A:0F:97`, paired new `78:86:2E:8C:47:97`. Burned a lot of time chasing a phantom "BlueZ procedure" issue before the actual root cause surfaced in dmesg: `BLE firmware version 5.09, please upgrade for better stability`. The new-out-of-the-box controller had old firmware → xpadneo's welcome-rumble crashed it → infinite reconnect loop (sysfs index raced from .003E to .00BF in seconds). User updated firmware to 5.23 via Xbox Accessories app on Windows. After update, single clean `pair → trust → connect` worked.
4. **Updated `fix_xbox_controller.sh`** — hardcoded MAC swapped to new controller (committed).
5. **Re-centered treatbot5 gimbal** — `pan_center: 64 → 46`, `tilt_center: 89 → 97` (live-read after user physically aimed dead-ahead). Written manually to yaml to avoid the `/camera/calibrate save:true` comment-stripping bug. Not live-reloaded (no `/config/reload`); takes effect next service restart.

### Key learnings (saved to memory)
- **`feedback_xbox_firmware.md`** — When a new Xbox controller won't pair stably (slow-flash LED, BlueZ shows Connected:yes, journal flood of `Error reading event: I/O operation on closed file`, sysfs reconnect-counter racing): FIRST check `dmesg | grep "BLE firmware"`. If xpadneo reports `please upgrade for better stability`, firmware is the blocker — Windows + Xbox Accessories app is the only real fix. Don't waste hours on bluez sequencing.

### Memory cleanup
- Wrote a `feedback_xbox_pairing.md` mid-session claiming `fix_xbox_controller.sh` was the procedure that worked — deleted it immediately when symptoms made clear the script wasn't actually working. Replaced with the firmware-first memory above.

### Procedures confirmed
- **Xbox controller pairing on Pi (with updated firmware):** `bluetoothctl scan on` → wait → `pair MAC` → `trust MAC` → `connect MAC`. Set agent to `NoInputNoOutput` first. `rfkill` cycle the radio if BT state is stuck.
- **Conflict-resolving `resume_chat.md` on rebase:** keep both session blocks, newest first; just delete the conflict markers and add a `---` between sessions.

### Pending / next session
1. **TMC2209 VIO wire** on treatbot5 — same physical fix as treatbot4 (Pi 3V3 → TMC2209 VIO pin). After fix, expect `TMC2209 detected: version=0x21` in dispenser init log instead of the "not responding" warning. Read part-2 session notes about the yaml time-bomb (microstepping mismatch) BEFORE wiring — treatbot5.yaml currently has no `microstepping` field, so it'll use whatever default; check before activating UART.
2. **Crank Vref pot on treatbot5** if dispenser torque feels light.
3. **Restart treatbot.service** to pick up the new gimbal centers (user explicitly said "it's fine" to defer).
4. **Battery false-charging on treatbot5** — still unresolved from prior session; needs per-unit charging-trend threshold tuning.

### Commits this session
- `005d78b` — chore: re-center treatbot5 gimbal + update fix_xbox_controller.sh MAC (pushed)

### Notes
- `.claude/TMC2209_UART_SETUP.md` remains untracked by design (its own header says "Per-Unit, Not in Git"). Treatbot5 passes 100% of its OS-side checks; only the hardware VIO wire is missing — same story as treatbot4.
- Old controller MAC `AC:8E:BD:4A:0F:97` is gone from bluez. If that controller comes back later, it'll need to re-pair from scratch.
## Session: 2026-05-25 (part 4) — treatbot4 VIO fix attempt, UART STILL silent

**Duration:** ~30 min, no code changes
**Robot:** treatbot4
**Status:** ⚠️ Diagnostic still open — VIO alone did not solve UART silence
**Important update for treatbot5 session above:** part-3 (treatbot5) attributed UART silence to the VIO disconnect from part-2's diagnosis. THIS session proves that diagnosis was **incomplete** — wiring VIO on treatbot4 didn't fix it. The actual root cause is still unknown. Don't wire VIO on treatbot5 expecting it to fix UART; do the multimeter test below first.

### What happened
- User wired 3.3V to TMC2209 VIO pin per last session's recommendation
- Re-probed UART after reboot: still silent. Boot log still says `TMC2209 not responding`. Live probe still only sees Pi's own TX loopback.
- User pointed out 2+ other robots have the same silent symptom, ruling out random chip damage / loose solder. This is systematic to their build pattern, not coincidence.

### Where we landed
- Confirmed topology is correct: split is at PDN_UART (labeled at "pin 5" on user's TMC2209 modules — module pinout differs from textbook StepStick), joining Pi pins 8 (TX, through 1K) and 10 (RX, direct)
- Ruled out: OS config, Pi-side wiring, wrong chip pin, VIO disconnect, single damaged chip
- Leading remaining hypothesis: PDN_UART pin sits HIGH at idle on these modules, putting the chip into power-down mode (`pdn_disable=0` is the chip default → PDN_UART idle HIGH = chip off). Adding VIO may have actually activated a module pull-up that holds the line HIGH. treatbot2 working might mean its module lacks that pull-up.

### Open diagnostic (user's next step)
Multimeter from TMC2209 PDN_UART pin to GND with treatbot.service stopped.
- ~3.3V → external pull-up issue; fix by removing module's pull-up jumper or adding stronger external pull-down (~4.7K to GND)
- ~0V → pull-up not the issue; investigate GND reference or chip variant differences

### Memory updated
`project_tmc2209_vio.md` rewritten to reflect that VIO alone wasn't the answer. Leading hypothesis and remaining diagnostic captured.

### No code changes this session — nothing to commit.

---

## Session: 2026-05-25 (part 2) — UART diagnosis, blue-tube night-off, dispenser reality check

**Duration:** ~2 hours
**Robot:** treatbot4 (with implications for whole fleet)
**Status:** ✅ Diagnostic + small fix complete. UART hardware fix is the user's next physical step.

### Major finding
**TMC2209 UART has been silent on treatbot4 (and likely several others) since fleet bring-up — root cause is the VIO logic supply line being disconnected.** STEP/DIR still work via the chip's internal regulator, so motors turn fine; UART transmit needs VIO to drive the line back to the Pi, so the chip echoes nothing. Empirically proven by:
- Direct UART probe across all 4 possible chip addresses → 0/12 reads succeeded (Pi sees own TX loopback, chip silent)
- Side-by-side comparison vs working treatbot2 → OS / cmdline / cmd group / pins.py / dispenser.py all identical, treatbot2 returns chip version 0x21, treatbot4 silent
- Reported by user: "same issue on other treatbots" + "i have the logic wire disconnected, i believed we didn't need it"

### What this means for the yaml
On treatbot4 (and any other unit with VIO disconnected), all of `irun`, `microstepping`, `chopper_mode`, `shaft_invert`, `sgthrs` are **inert** — they describe an intent the chip never receives. The chip runs on hardware defaults (Vref pot + MS1/MS2 pin strapping). The dispenser has been working all along on these defaults; "light torque" is the user's actual mechanical concern.

### Time bomb to be aware of
treatbot4.yaml currently has `microstepping: 4` + `steps_per_slot: 137`. As long as UART stays silent, this is fine (chip ignores microstepping and uses MS1/MS2 default of 8). But the moment the user fixes VIO and UART starts working, `microstepping: 4` will take effect and 137 steps × (1.8°/4) = 61.6° per slot — **2× over-rotation per dispense**. Either revert microstepping to 8 in yaml BEFORE wiring VIO, or update both together.

### Other work this session
1. **Night mode also kills the blue LED tube** (`main_treatbot.py`). Previous callback only stopped the NeoPixel strip; the separate blue tube stayed on. Now calls both `set_pattern('off')` and `ctrl.blue_off()` on day→night.
2. **Confirmed Flutter app's "night mode" toggle is UI-only** — not wired to our `set_night_mode_override` command. Self-contained brief written for whoever updates the Flutter app to send the relay command + listen for `night_mode_state` events.
3. **User reverted treatbot4.yaml `steps_per_slot` 69→137** and bumped `reverse_steps` 50→70. The 69 value was incorrect for the actual (UART-silent) microstepping the chip uses.

### Practical fix for "light torque" (no UART required)
The Vref pot on each TMC2209 module sets the hard current ceiling regardless of UART. Turning it CW raises current. Target ~2.0–2.2V on the Vref pad (multimeter to GND). User has this on their TODO.

### Documented in this session
- `.claude/TMC2209_UART_SETUP.md` (user added) — full per-Pi UART setup guide; treatbot4 already passes 100% of its checks at the OS layer. Only the hardware VIO wire is missing.

### Next session
1. **Wire VIO to Pi 3.3V on each affected robot** — Pi pin 1 (3V3) → TMC2209 VIO pin. Verify after reboot: `TMC2209 detected: version=0x21` in dispenser init log.
2. **Before doing that, fix the yaml time bomb** — change `microstepping: 4 → 8` in treatbot4.yaml to match working treatbot2 geometry, OR plan to update both fields atomically.
3. **Crank Vref pot on treatbot4** — give the carousel more current to handle load.
4. **Wire Flutter app's night-mode toggle to backend** — the relay-command brief from this session.

### Commits this session
- (pending) blue-tube night-off + yaml steps_per_slot/reverse_steps revert

---

## Session: 2026-05-24/25 — treatbot4 night-vision + camera/battery/dispenser tuning

**Duration:** ~6+ hours (long session across the date boundary)
**Robot:** treatbot4
**Status:** ✅ Complete — committed as 5f0dacc, pushed to origin/main

### Problems Solved
1. **NoIR daytime color** — IMX708 Wide NoIR (IR-cut filter removed) had unusable daytime color via AWB (green skin OR magenta blues depending on preset). Manual `ColourGains=(0.887, 1.52)` locked + AWB off + saturation 0.85 + contrast 1.05 is the best achievable in software; fundamental NoIR limitation (green channel dominant, ColourGains only controls R+B). Real fix would be a physical IR-cut filter for daytime use.
2. **Night mode subsystem built** — full implementation per `.claude/nightvisionrobo.md`. Asymmetric hysteresis (5 lux entry, 100 lux exit) was added specifically because IR illuminator pollutes the camera's lux reading; without the high exit threshold the system would oscillate.
3. **Battery showed 148V / 100%** — treatbot4 has the LOW-ratio voltage divider (~5.6:1), not treatbot3's high-ratio (~54:1). Calibrated against multimeter: 15.43V battery → 2.7368V at A0 → factor 5.638. Now reads correctly.
4. **Dispenser stalling** — yaml-side torque was already maxed (irun=31, microstepping=4, slow step_delay). Root cause: TMC2209 was in stealthChop mode (default, ~50-70% torque). Added `chopper_mode` yaml field; setting "spreadcycle" sets GCONF bit 2 for full rated torque. Opt-in per robot so other units unaffected.
5. **`/camera/calibrate` save:true reformats yaml** — discovered this bug mid-session when it stripped comments + sorted keys. Worked around by writing yaml manually. Real fix would be ruamel.yaml in the save path. Avoid `save:true` until that's fixed.
6. **WebRTC stuck reconnecting** — Camera/relay healthy but app never sent SDP answer. Force-quit + relaunch of phone app fixed it (stale peer connection state).

### Key Code Changes
- `modes/night_mode_controller.py` (NEW) — threaded singleton, lux polling, profile switching via `picam2.set_controls()` (preserves WebRTC stream)
- `services/perception/detector.py` — `_apply_saved_calibration` now accepts `awb_enable` + `colour_gains`
- `services/reward/dispenser.py` — `_configure_tmc` reads `chopper_mode` from yaml, sets GCONF bit 2 for spreadCycle
- `services/cloud/relay_client.py` — added `set_night_mode_override` command handler
- `api/server.py` — added `colour_gains`/`awb_enable` to `/camera/calibrate`, plus `GET /night_mode/status` + `POST /night_mode/override`
- `main_treatbot.py` — wired NightModeController into init/start/stop; LED-off callback on day→night
- `config/robot_profiles/treatbot4.yaml` — full daytime camera profile, battery factor 5.638, chopper_mode spreadcycle, dispenser torque tuning

### What Was NOT Done (deferred)
- **Pi onboard ACT/PWR LED dim** — needs a sudoers entry, didn't tackle
- **`tests/hardware/test_battery_adc_channels.py`** — untracked diagnostic, didn't commit (user's call)
- **Xbox MAC files** — `fix_xbox_controller.sh` + `services/control/xbox_persistent.py` still uncommitted (per-robot shared files, intentional)
- **`/camera/calibrate` save:true comment-preservation fix** — known bug, deferred
- **Live yaml reload** — there's no `/config/reload`; all yaml is read-once at service init. User now knows.

### Important Notes / Gotchas for Next Session
- **NoIR daytime color is genuinely a hardware limitation** — don't chase it further in software. Either accept the muted profile or buy a clip-on IR-cut filter.
- **Asymmetric hysteresis (5/100 lux)** is intentional — do not "fix" by making it symmetric, or IR illuminator will cause oscillation.
- **Night mode profile-switching uses `set_controls()` only** — never call `start()`/`stop()`/`configure()` from the night mode path; that would kill WebRTC. The settle pattern is: AE on briefly → 3s settle → read metadata → lock values with AE off.
- **`chopper_mode` field defaults to stealthchop** when absent — safe default for other robots.
- **State file `state/night_mode.json` persists override** across restarts; do not delete unless intentional.
- **cam1 port on treatbot4 Pi 5 is suspect** — camera is wired to **cam0**. Earlier session troubleshooting strongly suggested cam1 port damage. Do not move ribbon to cam1.
- **Service must be restarted to test:** spreadcycle dispenser fix, latest camera profile values, battery factor (already restarted once during session — verified working). User had not done the spreadcycle test by end of session.

### Commit
`5f0dacc — feat: night mode controller + treatbot4 camera/battery/dispenser tuning` (pushed to origin/main)

### Next Session
1. **Verify spreadcycle dispenser** — restart service, listen for motor whine on dispense, confirm `chopper=spreadcycle` in TMC2209 init log
2. **Real-world night mode bench test** — dark room, watch system auto-switch to night, confirm IR illuminator doesn't oscillate the mode
3. **Decide on IR-cut filter** — physical fix for daytime color if dog-training footage quality matters
4. **Pi onboard LED dim** — sudoers entry + writes to `/sys/class/leds/{ACT,PWR}/brightness` if user wants this minor enhancement
---

## Session: 2026-05-22 — treatbot5 device setup + power-button diagnosis

**Robot:** treatbot5
**Status:** ✅ Complete. No repo code changes — working tree clean (only this log updated).

### Work completed
1. **Git pull** — `82dbe8b..652a06f` fast-forward, 27 files (+1986/−177). Brought in volume control system (`services/media/volume_manager.py`, `wimz-audio.service`, `apply_saved_volume.py`), adaptive bitrate streaming (`services/streaming/adaptive_bitrate.py`), and device setup tooling (`scripts/setup_device.sh`, `rtw88.conf`, `docs/NEW_ROBOT_SETUP.md`).
2. **Ran `scripts/setup_device.sh`** on treatbot5 — all verification checks passed:
   - `/etc/wimz/` created (owner morgan) — persistent volume state dir
   - `wimz-audio.service` installed + enabled; ran clean (no saved state → applied default 60% via amixer card 2, control 'Speaker')
   - `/etc/modprobe.d/rtw88.conf` installed — disables rtw88 WiFi deep power-save
   - WiFi power-save disabled on all 4 saved connections
3. **Confirmed xpadneo installed** — DKMS `hid-xpadneo v0.9-226-ga16acb0` built for all 4 kernels incl. running `6.12.62+rpt-rpi-2712`. Module auto-loads on Xbox controller BT connect.

### Diagnosis (no fix applied — user said not needed)
- **`reboot` actually powers the robot off.** Root cause: `/lib/systemd/system-shutdown/wimz-killpulse` pulses GPIO26 → Pololu OFF unconditionally. systemd runs system-shutdown hooks for ALL verbs (poweroff/halt/**reboot**/kexec), passing the verb as `$1`. The script ignores `$1`, so a reboot cuts power instead of restarting.
- **Fix (deferred):** wrap the `gpioset` in `case "$1" in poweroff|halt) ... ;; esac`. This hook is NOT in the repo — if fixed later, also add to `scripts/` + `setup_device.sh` for fleet coverage.

### Pending — NOT yet activated
- **WiFi driver change** (`rtw88.conf`) takes effect only after reboot.
- **New pulled code** (volume manager, adaptive bitrate, etc.) loads only after `treatbot.service` restart. Service currently still running old code from May 20 boot.

### Observed this session
- **False "charging" detection on treatbot5** — robot announced/reported charging while NOT plugged in. `battery_monitor._check_charging()` infers charging from a voltage upward trend (motor-idle gated). On treatbot5 this still misfires — needs a per-unit charging threshold/calibration pass. Not fixed this session.

### Next session (TBD — treatbot5 calibration backlog)
1. **Camera/AI verify** — confirm IMX708, gimbal snaps to (pan=64, tilt=89) on boot, AI pipeline + 4-class behavior model load cleanly.
2. **Dispenser test** — never run on treatbot5; trigger auger advance via API.
3. **Motor calibration** — on-floor drive test (all prior testing bench/wheels-lifted); tune `left_multiplier`/`right_multiplier` if it pulls; coach mode tilt re-tune for asymmetric tilt range.
4. **Battery % calibration** — voltage→percent curve and the false-charging detection (`battery_monitor.py`, `_check_charging()`). treatbot5 reads charging while unplugged; needs per-unit voltage-trend threshold tuning. (`calibration_factor` already set to 53.33 last session for the divider, but charging-trend logic is separate.)
---

## Session: 2026-05-21/22 — treatbot2 hardware bring-up + relay mood_led

**Duration:** ~2 hours
**Robot:** treatbot2
**Status:** ✅ Complete

### Problems Solved
1. **WiFi power-save** — Ran setup_device.sh, disabled power-save on all saved networks
2. **Camera ribbon loose** — Identified camera timeout error after reboot, user reseated ribbon
3. **Blue LED not working via app** — Added `mood_led` command handler to relay_client.py (was only in local ws.py)
4. **Gimbal calibration off** — Recalibrated pan (-60/180/320) and tilt (45/90/200) for treatbot2
5. **Xbox controller pairing** — Installed xpadneo driver, paired using `agent on` + `default-agent` sequence
6. **Blue LED hardware** — Diagnosed MOSFET wiring issue (Source wired to GPIO instead of Gate) — user fixed

### Key Code Changes
- `services/cloud/relay_client.py` — Added mood_led command handler for relay path
- `config/robot_profiles/treatbot2.yaml` — Updated gimbal calibration values

### Commit
`46ec876` — feat: relay mood_led command + treatbot2 gimbal calibration

### Hardware Notes (treatbot2)
- Camera ribbon came loose after reboot — check connections on this unit
- Blue LED MOSFET had Source/Gate wires swapped — now fixed by user
- Xbox controller paired successfully (MAC: 28:EA:0B:DB:82:3F)
- xpadneo driver installed via DKMS (not in git — must install per-robot)
- Gimbal pan went to -80 and tore camera ribbon — safe limit set at -60

### Unresolved / Next Steps
- Blue LED toggle in Flutter app needs to send `mood_led` command via relay WebSocket (app-side fix)
- Verify Xbox controller works with treatbot service after reboot
- Consider adding xpadneo install to setup_device.sh for new robots

### Important Notes
- xpadneo is a kernel driver (DKMS), not in git — must be installed separately on each robot
- Xbox pairing requires: `agent on` + `default-agent` BEFORE `pair` command
- JustWorksRepairing was enabled in /etc/bluetooth/main.conf

---

## Session: 2026-05-20 — Silent Guardian movement tier + Xbox controller bring-up

**Goal:** Add a physical-movement escalation tier to Silent Guardian; pair a new Xbox controller; diagnose drift + right-motor issues.
**Status:** ✅ All complete. Working tree clean, 2 commits pushed (`04295ee`, `f4c4876`). Battery-smoothing commit `337d92d` also pushed early in session.

### Work completed
1. **Silent Guardian — new Level 3 movement tier** (`04295ee`). Escalation ladder is now 4 levels: verbal → firm verbal → **physical movement** → calming music. Level 3 plays "quiet" then runs 3 cycles of in-place fwd/back/left/right (~400ms each, motors halted between moves, 5s pause between cycles), then waits for the progressive quiet period → reward. `_run_movement_sequence()` drives wheels via `motor_command_bus` (`CommandSource.AUTONOMOUS`), always finishes all cycles (no bark abort), bails on mode shutdown. `max_level` 3→4; music handler renamed `_process_level_3`→`_process_level_music`. No conflict with carousel anti-jam (separate stepper) or treat dispense (serialized on SG loop thread). **App-facing:** `get_status()` now reports `sg_max: 4`.
2. **Xbox controller fixes** (`f4c4876`):
   - **Right motor undrivable via Xbox** — root cause: the Xbox subprocess called `get_motor_bus()`, creating a *second* `ProperPIDMotorController` that fought the main process for motor GPIO lines. Fix: `xbox_controller.py` sets `WIMZ_XBOX_SUBPROCESS=1`; subprocess now drives motors via HTTP API only (main process = sole hardware owner). Standalone mode unaffected.
   - **Connected-but-unresponsive after Bluetooth blip** — a BT link drop recreates `/dev/input/js0`, leaving a stale fd (errno 19); the retry path never reopened it. Added `_reopen_device()` to self-heal.
   - **Reverse "sticking"** — neutral-stop sent a single stop then relied on a 2s heartbeat; a dropped stop left the motor creeping. Now re-sends 0 every ~50ms for a 0.5s window on transition to neutral.
   - Added `_calibrate_stick()` + `controller.xbox.stick_centers` (yaml) for worn/off-center sticks; default 0.0 = no-op.

### Hardware findings / warnings
- **Spare Xbox controller `A8:8C:3E:50:62:70` is JUNK** — left stick Y potentiometer worn out: rests at ~−0.72 and the rest position *wanders* (caused runaway-at-rest). Unstable rest can't be software-calibrated. **Unpaired this session — do not re-pair it.**
- treatbot1 is back on the **original controller `AC:8E:BD:4A:0F:97`**. MAC restored in `xbox_persistent.py` + `fix_xbox_controller.sh`.
- Right motor + encoder verified healthy via direct test (−707 enc counts) — it was never a hardware fault.
- No new files created this session; all changes were edits. Directory structure doc unchanged.

### Next steps
- Restart `treatbot.service` after any reboot (it ran the new code live this session — currently active).
- Coordinate `sg_max: 4` with the Flutter app team (Guardian level display "X/4").
- Silent Guardian Level 3 movement tier verified by code/syntax only — exercise it live when a 4th bark intervention naturally escalates.

## Session: 2026-05-17 — treatbot5 bring-up (cloned from treatbot3)

**Goal:** First-boot bring-up of treatbot5, cloned from treatbot3's SD card. Pull latest, customize per-unit calibration, run drive + gimbal test sequences.
**Status:** ✅ Core bring-up complete. Dispenser test skipped; full vision/AI verification deferred (camera was unplugged mid-session). Working tree changes ready to commit: `config/robot_profiles/treatbot5.yaml`, `.claude/resume_chat.md`.

### Capabilities now working on treatbot5
- **Identity**: hostname `treatbot5`, `DEVICE_ID=wimz_robot_05`, machine-id `f7cfcc5dbb3b4210b526449a5e9eb5d9`. Profile auto-selects via hostname_map.
- **Cloud relay**: connected `wss://api.wimzai.com/ws/device` on first boot — Lightsail already had `wimz_robot_05` registered (no relay-side work needed this session).
- **Battery monitoring**: reads 16.87V at 16.8V bench (0.4% error) with `calibration_factor: 53.33`.
- **Drive train**: Cytron MDD10A live, both wheels respond, forward+reverse verified.
- **Gimbal**: center + limits calibrated via Xbox-stick sweep.

### Per-unit calibration deltas vs treatbot3
1. **Battery divider runs ~1.8% leaner**: factor 53.33 (vs treatbot3's 54.28). Same hardware design, normal resistor tolerance variance.
2. **Right motor wired with reversed polarity** (same as treatbot3): `right_invert: true`. Left motor fine (`left_invert: false`).
3. **Gimbal center is at completely different servo positions**: treatbot5 lives at `pan_center: 64, tilt_center: 89` vs treatbot3's `67, -12`. Pan close but tilt convention reads very differently — both use "higher tilt = looking up" but the absolute neutral pulse on this servo lands at +89 instead of -12. Mounting/servo tolerance.
4. **Tilt range is asymmetric on this unit**: physical limits 57 (down, mechanical wall) to 236 (up). Center at 89 → only 29° of down-tilt headroom but 147° of up-tilt headroom. Locked `tilt_min: 60` (3° safety from wall), `tilt_max: 236`.
5. **Pan range**: `pan_min: -80`, `pan_max: 268` (user-chosen "comfortable max" via Xbox sweep, not necessarily mechanical wall).
6. **`coach_pan_*` / `coach_tilt_*` inherited verbatim from treatbot3** but need re-tuning in actual coach mode — the 30%-inset formula produces tilt range biased upward (away from floor-level dog tracking) given treatbot5's asymmetric tilt range. Flagged in yaml comment.

### Hardware findings
- **Xbox controller (AC:8E:BD:4A:0F:97)** paired fresh — the cloned BlueZ bonding database came over empty (or got cleared) so no pre-pair carried from treatbot3.
- **Camera (IMX708 wide)** was physically unplugged sometime during the gimbal calibration → service hung in startup with V4L2 buffer errors on next restart. Service stopped cleanly at end of session. **Next session: plug camera in, restart service, verify gimbal snaps to (pan=64, tilt=89) and AI pipeline starts cleanly.**

### Drive test sequence executed (wheels lifted, 30% throttle)
1. LEFT only +30 → forward ✅ (`left_invert: false` correct)
2. RIGHT only +30 → BACKWARD ❌ → flipped `right_invert: true` → retest → forward ✅
3. BOTH +30 → straight forward ✅
4. BOTH -30 → straight reverse ✅

### Gimbal calibration sequence (live Xbox sweep, position read via GET /camera/gimbal)
- CENTER: pan=64, tilt=89 (`current_position` reading)
- MAX UP: tilt=236 (locked as tilt_max)
- MAX DOWN: tilt=57 (mechanical wall hit, +3° safety → tilt_min=60)
- MAX LEFT: pan=267.8 (locked → pan_max=268)
- MAX RIGHT: pan=-78.9 (locked → pan_min=-80)

### Pending for treatbot5 (next session)
1. **Re-plug IMX708 camera, restart service, verify** new gimbal snaps to (64, 89) on boot and AI detection pipeline starts. Confirm 4-class behavior model loads.
2. **Dispenser test** — skipped this session. Trigger via API, confirm auger advances. Expect cosmetic "TMC2209 not responding on UART" warning (matches treatbot3).
3. **Field test on-floor** — all drive testing was bench/wheels-lifted.
4. **Coach mode tilt range re-tune** — `coach_tilt_min/max` currently inherited from treatbot3 and likely tracks too high given asymmetric tilt range.
5. **Optional motor calibration tuning**: `left_multiplier`/`right_multiplier` are 1.0/1.0; under load may need bias if robot pulls one direction.

### Commits this session
- Pulled 5 commits from origin/main (392c2cd, 0cbaeaf, 7a37458, 903b2c0, b4e74db) — includes 174 IMX500 training sequences, LSTM training guide, charging-detector fix.
- About to commit: treatbot5.yaml per-device calibration + this resume_chat update.

### Files touched
- MODIFIED: `config/robot_profiles/treatbot5.yaml` (full per-device fill-in: battery factor, Cytron driver dispatch, motor inversion, gimbal calibration)
- MODIFIED: `.claude/resume_chat.md` (this entry)

---

## Session: 2026-05-17 — treatbot4 hardware bring-up

**Goal:** Bring treatbot4 from cloned-from-treatbot1 SD card to a fully-calibrated working unit. Same Cytron + brushed-motor + IMX708 hardware family as treatbot3, but on a different physical Pi/PCB.
**Status:** ✅ COMPLETE (modulo battery, which needs actual battery plugged in to verify). Commit `89efa8f` pushed locally (not yet pushed to origin).

### Capabilities now working on treatbot4
- **Drive train**: Xbox left stick → Cytron MDD10A → both wheels respond, correct directions
- **Camera gimbal**: pan inherited from treatbot3 (pan_center=67), tilt re-calibrated after dead-servo replacement (tilt_center=43)
- **Cloud relay**: connected to `wss://api.wimzai.com/ws/device`; wimz_robot_04 registered Lightsail-side
- **Xbox controller**: paired (28:EA:0B:DB:82:3F), connected via /dev/input/js0
- **Camera**: IMX708 Pi Camera Module 3 Wide, 19.4 FPS detection pipeline

### Hardware findings worth remembering
- **Right motor wired with reversed polarity** — same as treatbot3. `right_invert: true`. Now 2-for-2 on Cytron brushed-motor builds → memory saved (`project_cytron_right_invert.md`) so treatbot5 starts with `true` as expected default.
- **Tilt servo was DOA** — confirmed via swap test (pan ↔ tilt cables swapped, neither chan-1 PCA output nor swapped-onto-chan-0 produced response). User installed a replacement servo mid-session in ~5 min.
- **New tilt servo's mechanical "level" position = code-angle 43°** (vs treatbot3's -12°). Different mounting offset. Symmetric ±50° physical range gives tilt_min=-57, tilt_max=143, with coach-mode 30% inset (3 to 83).
- **Battery sense divider on this PCB is wired the same as treatbot3** — but bench supply is going directly into the Pololu output (downstream of the divider tap), so ADS1115 A0 reads ~-27mV (floating). Calibration_factor 54.28 inherited; will read correctly once battery plugs into the JST.

### Commits this session
- `89efa8f` feat: treatbot4 hardware bring-up — Cytron + camera gimbal + new tilt servo

### Files touched
- MODIFIED: `config/robot_profiles/treatbot4.yaml` (Cytron section, battery cal, tilt re-cal, driver dispatch)

### Pre-existing issues still pending (unchanged from treatbot3 session)
- `services/motion/motor.py:MotorService.initialize()` NameError on ENCODER_MOTOR_AVAILABLE — dead code, not blocking
- TMC2209 dispenser UART warning — defaults match yaml, dispenser works fine
- Battery monitor `-1.48V CRITICAL` logs (test-rig artifact, resolves with battery plugged in)
- `/camera/center` and `/camera/position` API endpoints reference missing methods (would 500 if called)

### Still pending for treatbot4 (next session candidates)
1. **Verify battery calibration** — plug actual battery in, confirm reading ~16.6V at 16.8V (matches treatbot3 pattern)
2. **Treat dispenser test** — TMC2209/NEMA settings inherited from treatbot3, never tested on this unit
3. **Field test outdoors / under-load drive** — all testing this session bench/wheels-up
4. **Re-tune motor `left_multiplier`/`right_multiplier`** if drive feels asymmetric under load
5. **Push commit `89efa8f` to origin** — currently local-only

### Notes for treatbot5 (next clone)
Per the documented procedure, treatbot5 should inherit treatbot3.yaml as starting point. NEW datapoint from treatbot4: right_invert=true should be the **expected default** for these Cytron brushed-motor builds (2/2). Servos may need re-calibration regardless of donor yaml (tilt mounting differed even between treatbot3 and treatbot4).

---

## Session: 2026-05-15 — Git sync + new Pi 5 diagnostics (treatbot2)

**Goal:** Pull latest changes, verify new Pi 5 hardware after old one was destroyed (12V GPIO mishap)
**Status:** ✅ Complete

### What Was Accomplished

1. **Git Sync**
   - Pulled 6 commits from origin/main (treatbot3 profiles, charging detector fix, etc.)
   - Fixed stale staged changes in `train_behavior_lstm.py` (4-class alignment, CPU export)
   - Added 174 IMX500 behavior sequences to repo
   - Updated LSTM training guide to remove "speak" (audio-only via bark detector)

2. **Hardware Diagnostics — New Pi 5 (4GB)**
   - Pi 5 Model B Rev 1.1: Healthy (48°C)
   - RAM: 4GB (2.5GB available, swap usage higher than 8GB unit)
   - IMX500 camera: Detected (4056x3040)
   - I2C: PCA9685 servo @ 0x40, ADC @ 0x48
   - USB Audio: Card 0
   - Soft power button: Armed and working

3. **Power Button Note**
   - Service starts ~1.5 min after boot (waits for multi-user.target)
   - User pressed button before service armed — expected behavior, not a bug

### Commits Pushed
- `903b2c0` — fix: Align train_behavior_lstm.py with deployed 4-class model
- `7a37458` — feat: Add IMX500 behavior sequences + LSTM training guide
- `0cbaeaf` — docs: Update LSTM guide — speak removed (audio-only via bark detector)

### Next Session
- No pending tasks

---

## Session: 2026-05-09 — Rebadge cloned Pi as treatbot3 + prep for treatbot4/5

**Goal:** This Pi was cloned from treatbot1's SD card. Give it a unique identity (treatbot3 / wimz_robot_03), pre-create profiles + bootstrap script so future clones for treatbot4 and treatbot5 are one-command operations.
**Status:** ✅ Code/config committed and pushed (`bc3ecb9`). Identity reset script ran successfully on this Pi. **Reboot pending — user will trigger when ready.**

### Hardware delta on units 3-5 (vs 1/2)
- Cytron motor driver + 9V brushed motors (no encoders) — replaces L298N + 6V DFRobot encoder motors. PID disabled because no encoder feedback.
- IMX708 Pi Camera Module 3 Wide — replaces IMX500 used on units 1/2.

### What Was Built

**Repo changes (commit `bc3ecb9`, pushed to origin/main):**
- `config/config_loader.py` — added `treatbot3/4/5` to hostname_map. After hostname change, profile auto-selects, no `/etc/robot_id` needed.
- `config/robot_profiles/treatbot3.yaml`, `treatbot4.yaml`, `treatbot5.yaml` — seeded from treatbot1.yaml. PID disabled, motor multipliers neutral (1.0/1.0), camera rotation 0. HARDWARE_NOTE block flags Cytron driver dependency.
- `scripts/wimz_rebadge.sh` — idempotent post-clone identity reset. Takes one arg (3, 4, or 5). Resets machine-id, regenerates SSH host keys, sets hostname + /etc/hosts, writes new .env with fresh DEVICE_SECRET. Does NOT auto-reboot (so user can read the printed secret).

**System changes on this Pi (via rebadge script):**
- machine-id: `67ec0d9e3e5e47c98d50d87f3684c555` (new, distinct from treatbot1)
- New SSH host keys (ed25519 fingerprint: `SHA256:HtaF5WgUG07LGRQJuEH63X5MewYq9SXGkPJ+yahvl78`) — note: key comment field still says "root@treatbot1" cosmetically because hostname changed AFTER key generation; harmless, will correct on next regeneration
- hostname → `treatbot3`, `/etc/hosts` 127.0.1.1 → `treatbot3`
- `/home/morgan/dogbot/.env` → `DEVICE_ID=wimz_robot_03` + a freshly-generated `DEVICE_SECRET` (stored only in .env, not in git — see file directly)
- Bluetooth pairing database left intact (Xbox controller bond preserved from clone)

### Critical follow-up actions (next session)

1. **Register on Lightsail relay**: `wimz_robot_03` + the secret in `/home/morgan/dogbot/.env` must be added to the relay's allowed-devices list before the robot can connect remotely.
2. **Cytron motor driver** (out of scope this session): `services/motion/motor.py` has no Cytron dispatch. Need a new `core/hardware/motor_controller_cytron.py` plus dispatch wiring before drive control works on units 3/4/5.
3. **Per-unit calibration during bring-up**: motor `left_multiplier`/`right_multiplier`, `camera.rotation`, `pan_center`/`tilt_center`, dispenser `steps_per_slot`. All currently neutral defaults in treatbot3.yaml.
4. **Verify after reboot**:
   ```
   hostname                                    # treatbot3
   journalctl -u treatbot.service -n 80 | grep -iE "profile|robot_id|treatbot3"
   ls /dev/input/js0                           # Xbox controller present
   bluetoothctl devices Connected
   libcamera-hello --list-cameras              # confirm IMX708 detected
   ```

### Cloning to treatbot4 / treatbot5 (future)
After this Pi is fully validated, clone its SD card. On each new clone, first boot:
```bash
sudo bash /home/morgan/dogbot/scripts/wimz_rebadge.sh 4   # or 5
sudo reboot
```
Then register the printed DEVICE_ID/SECRET on Lightsail. That's it.

### Concurrency notes
- treatbot1 stays online. Two separate Xbox controllers (one per robot) — first-come-first-served Bluetooth, no preemptive removal needed.
- Once relay-registered, treatbot3 will appear as a separate device alongside `wimz_robot_01`.
- Laptop SSH known_hosts: `ssh-keygen -R treatbot1.local` and any cached IPs before next SSH from laptop, otherwise host-key-mismatch warning.

---

## Session: 2026-04-27 — Git Pull & Camera Boot Config Docs

**Goal:** Pull latest changes, document camera boot config for fleet
**Status:** COMPLETE

---

### What Was Accomplished

1. **Pulled origin/main** (0806e6a → 954fb42)
   - 35 files updated with significant changes
   - New scripts: capture_behavior_sequences.py, train_behavior_lstm.py
   - New service: camera_detect.py
   - Major updates to main_treatbot.py, api/server.py, wifi_manager.py

2. **Documented Camera Boot Config**
   - Added to `.claude/hardware_specs.md`: camera auto-detect settings
   - `/boot/firmware/config.txt` needs `camera_auto_detect=1` and `#dtoverlay=imx500`
   - Fleet status: treatbot2 fixed, treatbot1/3-5 need edit before camera swap

### Files Modified
- `.claude/hardware_specs.md` — Added camera boot config section

### Commit
- c6e5998 — docs: Add camera boot config for IMX500/IMX708 swaps

### Next Session
- No pending tasks from this session

---

## Session: 2026-04-25 — Dog Tracking Deduplication Fix

**Goal:** Fix duplicate bounding boxes (both "Dog" and "Elsa" on same physical dog)
**Status:** COMPLETE

---

### What Was Accomplished

#### Bug Fix: Duplicate Detection Boxes
**Root Cause:** When ArUco visibility is intermittent, both an ArUco-identified entry (e.g., "elsa" at marker_id=42) and a generic entry ("dog_0" at -1000) coexisted in `last_known_positions`, causing two boxes to render.

**Not a regression** - gap from Build 38 (commit 3119962) which added generic entries but never implemented deduplication.

**Fix Applied** (`core/dog_tracker.py`):
1. When ArUco detected → delete any generic entry for that detection index
2. Before creating generic entry → check for overlapping ArUco-identified dog via IoU (>0.3) and reuse that identity
3. In `get_tracked_dogs()` → skip generic entries overlapping ArUco boxes (safety net)

**New helper methods:** `_find_overlapping_tracked_dog()`, `_bbox_iou()`

### Commits
- `0bd343f` — fix: Deduplicate dog tracking to prevent duplicate bounding boxes

### Also This Session
- Gimbal calibration API added (commit `966b6b3`)
- Xbox controller docs updated

### Next Session
- Restart treatbot to apply fix: `sudo systemctl restart treatbot`
- Test with Elsa to verify single box renders
- Lifecycle handler for clean disconnect (interrupted investigation)

---

## Session: 2026-04-25 — Gimbal Calibration API

**Goal:** Add configurable gimbal limits and API endpoints
**Status:** COMPLETE

---

### What Was Accomplished

1. **Gimbal limits now configurable** in `config/robot_profiles/treatbot1.yaml` under `camera:` section
2. **pan_tilt.py updated** to load limits from robot config at startup
3. **New API endpoints added:**
   - `GET /camera/gimbal` - View current limits
   - `POST /camera/gimbal/calibrate` - Adjust limits at runtime with optional save
4. **Xbox controller docs updated** (`XBOX_CONTROLLER_USAGE.md`) with current button mapping (RT=Good, RB=No, etc.)

### Files Modified
- `config/robot_profiles/treatbot1.yaml` — Added gimbal config params
- `services/motion/pan_tilt.py` — Load limits from config
- `api/server.py` — New gimbal calibration endpoints
- `XBOX_CONTROLLER_USAGE.md` — Updated button mapping

### Config Parameters Added
```yaml
camera:
  pan_min: 10
  pan_max: 200
  tilt_min: 20
  tilt_max: 160
  pan_center: 110
  tilt_center: 90
  coach_pan_min: 55
  coach_pan_max: 145
  coach_tilt_min: 25
  coach_tilt_max: 85
```

---

## Session: 2026-04-25 — WiFiManager Race Condition Fix

**Goal:** Investigate and fix system freezes within 2 minutes of launch on internet
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Root Cause Identified
- System was freezing (hard lock, required power cycle) within ~2 minutes of launch
- Initial wrong diagnosis: LED/SPI bus hammering from rapid pattern changes
- **Actual cause:** Race condition in WiFiManager — multiple threads (WiFi monitor + relay client) creating separate WiFiManager instances and calling nmcli/pgrep concurrently
- Kernel wireless driver deadlocked when hit from multiple threads simultaneously

#### 2. Fix Implemented (c64a160)
- Added `get_wifi_manager()` singleton factory function with global lock
- Added `_op_lock` RLock to WiFiManager to serialize all wireless operations
- Wrapped critical methods: `is_ap_mode()`, `get_connection_status()`, `try_connect_known()`, `start_hotspot()`, `start_demo_hotspot()`, `stop_hotspot()`
- Updated main_treatbot.py and relay_client.py to use singleton
- Fixed WiFi monitor bug: moved thread start to after `self.running=True` (was exiting immediately)

#### 3. CLAUDE.md Updated
- Added "Embodied Context" notice — Claude Code runs directly ON the robot, not external

### Files Modified
- `services/network/wifi_manager.py` — Singleton + thread locks (+277/-236 lines)
- `main_treatbot.py` — Use singleton, fix WiFi monitor start order
- `services/cloud/relay_client.py` — Use singleton (2 locations)
- `.claude/CLAUDE.md` — Added embodied context notice

### Commit
`c64a160` — fix: WiFiManager race condition causing system freeze

### Next Session
- Restart treatbot service to apply fix: `sudo systemctl restart treatbot.service`
- Monitor for 5+ minutes to verify no freezes
- Check logs: `journalctl -u treatbot.service -f`

### Important Notes
- The crash happened specifically "on internet" because relay was active, increasing likelihood of race
- WiFi monitor thread was also broken (exited immediately) — now fixed
- Crashes correlated with LED pattern changes were coincidental, not causal

---

## Session: 2026-04-17 — Documentation Update & Status Review

**Goal:** Update project documentation with current status, verify all systems
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Product Roadmap Updated (`product_roadmap.md`)
- Changed "Unknown Status" → "Verified Working (April 2026)"
- All App/Relay integration items confirmed working
- All Coach Mode items confirmed working
- All Silent Guardian items confirmed working
- Hardware status confirmed (servo calibration per-unit note added)
- Direct LAN Connection changed from "Deferred" to "IMPLEMENTED"
- Weekly Summary marked as tested (not 100% accurate)
- Mission Scheduler marked as NOT TESTED
- Xbox controller: Added joystick push buttons (center camera, anti-jam)

#### 2. Development TODOs Updated (`development_todos.md`)
- Updated to Build 83 (from Build 40)
- All live testing items marked complete
- Fixed incorrect checkboxes (Mission Scheduler was marked done but not tested)
- Added treat inventory tracking note
- Updated Dropped Features section

#### 3. Hardware Specs Updated (`hardware_specs.md`)
- Treat carousel: MG996R servo → NEMA 17 stepper + TMC2209
- Raspberry Pi 5: Added 4GB variant support note
- GPIO pin mapping: Cleaned up formatting, added TMC2209 pins
- Camera: Added RPi Camera Module 3 Wide as alternative option
- PCA9685 channels: Updated (0=pan, 1=tilt, stepper is separate)
- Treat system: 44 treats (11 × 4 carousels), stepper specs

#### 4. Created LLM Update Document (`LLM_UPDATE_APRIL.md`)
Comprehensive status document covering:
- Reliability metrics (what we have, what we need)
- Performance stats (detection, bark, treats)
- Crash recovery documentation
- WiFi reconnection reliability
- MVP definition (all core features complete)
- Demo breakdown (what works, caution areas, disabled features)
- Current failure points (all resolved)
- Target user (high-income pet owner)
- Data storage confirmation (SQLite: barks, treats, sessions, emotions)

#### 5. Memory Updated
- Direct AP Connection memory updated to "FULLY IMPLEMENTED"

### Key Confirmations from User

**Working Systems:**
- Relay forwarding mission_progress events (not all missions tested)
- Video overlay with AI confidence labels
- Servo tracking checkbox in app
- MP3 upload/download flow
- Bark filter (mostly rejects claps/voice)
- Pose thresholds accurate
- Full coaching session end-to-end
- Silent Guardian bark → intervention → reward
- Escalation and cooldown
- Treat dispenser reliable
- Audio playback consistent
- Servo calibration (needs per-unit tweaking)

**Resolved Issues:**
- WiFi instability → AP fallback + captive portal
- AI misfires → None reported
- Treat jams → NEMA 17 + TMC2209 anti-jam
- Docking → Feature disabled (too complex)

**Not Yet Tested:**
- Mission Scheduler auto-scheduling
- Mission Scheduler time windows
- Weekly Summary accuracy needs work

### Files Modified
- `.claude/product_roadmap.md` - Status updates
- `.claude/development_todos.md` - Build 83 updates
- `.claude/hardware_specs.md` - Hardware corrections
- `.claude/LLM_UPDATE_APRIL.md` - NEW comprehensive status doc
- `~/.claude/projects/.../memory/project_direct_ap.md` - Updated to implemented

### No Code Changes
This was a documentation-only session.

### Next Session
- Consider formal reliability testing for metrics
- Test Mission Scheduler
- Improve Weekly Summary accuracy
- Per-unit calibration documentation for manufacturing

---

## Previous Session: 2026-04-08 — Hailo Driver Patch, Audio Fallback

**Goal:** Diagnose SSH freeze, fix Hailo driver crash, treatbot2 audio/dispenser config
**Status:** FIXED — Hailo driver patched, treatbot running stable

### What Was Accomplished
- Hailo-8 PCIe driver patched (find_vma lock fix)
- Audio ALSA fallback implemented
- TMC2209 UART setup guide created
- Treatbot2 stepper config updated

### Critical Lessons
1. NEVER init hardware at Python module import time
2. kernel.warn_limit can make bugs worse

### System Changes Applied
- `/usr/src/hailo_pci-4.21.0/linux/vdma/memory.c` — patched via DKMS
- Patch files: `patches/hailo_pci_find_vma_fix.patch` + `patches/apply_hailo_fix.sh`
