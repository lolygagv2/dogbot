# TreatBot V2.0
AI-powered dog training robot with treat dispensing

## Hardware
- Raspberry Pi 5 with Hailo-8 AI accelerator (26 TOPS)
- Devastator tank chassis = 2 DC Motors
- PCA9685 servo controller
    - Servo Pan for Camera Horizontal Control
    - Servo Pitch for Camera Vertrical Control
    - Servo Hacked Continuous Rotation - A hacked MG9969R Servo that rotates continuously to dispense treats
- Custom 4S2P 14.8V 8000Mah battery system
- Custom NeoPixels 75 LED light String
- Custom Bluetube LED mood lighting
- IMX500 AI Pi Camera
- 2 Single Channel Analog Relays
- DFPLayer Pro mp3 player
- Lavalier Microphone

- Ultrasonic tweeter: UGREEN USB sound card -> LM386 amp -> 10-50 kHz piezo horn (amp +12 V switched by a relay on PCA9685 channel 2; it must switch the positive side, because the audio cable's ground bypasses a low-side switch)

### Sound cards (matched by USB ID, never by card number)
| USB ID | Device | ALSA name | Role |
|---|---|---|---|
| `0d8c:0014` | C-Media "USB Audio Device" | `Device` | Speaker + microphone (via PipeWire) |
| `12d1:0010` | UGREEN 24-bit/96 kHz, shows as "KT USB Audio" | `ultra` (`plughw:ultra,0`) | Ultrasonic tweeter only (`ultrasonic.py`, direct ALSA) |

- `scripts/85-wimz-ultra-audio.rules` -> `/etc/udev/rules.d/` gives the UGREEN the fixed id `ultra`.
- `scripts/51-wimz-ultra-disable.lua` -> `~/.config/wireplumber/main.lua.d/` keeps PipeWire off it.
- Speaker/mic card detection skips `12d1:0010`. `aplay -l` on treatbot5 (2026-11-11):
  `card 2: ultra [KT USB Audio]`, `card 3: Device [USB Audio Device]`.
- Volume control on `ultra` is `Headphone` (it has no `PCM` control).
- Test: `python -m ultrasonic --test` (audible 2 kHz), `python -m ultrasonic --freq 25000 --level 40`.
  Longer / pulsed: `--seconds 3`, `--freq 20000 --alt-freq 25000 --seconds 5 --pulse 0.3`.
- Silent Guardian uses it only when the robot profile sets `ultrasonic_enabled: true` (default false):
  a random 20/25 kHz tone of 1-3 s before the level 1 "quiet"; a 3 s tone 10 s before the calming
  music; a 5 s pulsed episode when barking continues through the music. Settings: `ultrasonic:` in the profile.
- Relay command `ultrasonic` (same flag; replies with `command_result`):
  `{style: before_quiet|before_music|nonstop_barking}` or
  `{freq_hz: 20000|25000|32000|40000, level_pct (max 80), duration_s (max 5), pulse_s?, gap_s?, alt_freq_hz?}`,
  optional `dog_id`; `{action: status}` lists what is available. Errors: `ultrasonic_disabled`, `cooldown`
  (+ `retry_after_s`; 20 s minimum between bursts), `busy`, `unavailable`, `bad_request`, `play_failed`.


## Features
- Dog pose detection
- Bark recognition  
- Autonomous treat dispensing
- Remote control via web interface
- Phone and App interface
- Mobile Behavioral Training Helper
