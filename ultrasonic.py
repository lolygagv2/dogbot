#!/usr/bin/env python3
"""
ultrasonic.py - ultrasonic burst through the dedicated "ultra" sound card.

Signal chain: UGREEN USB sound card (USB 12d1:0010, ALSA id "ultra", see
scripts/85-wimz-ultra-audio.rules) -> LM386 amp -> 10-50 kHz piezo horn tweeter.
The amp's +12 V supply (12 V per Morgan; not measured) goes through a relay driven
by PCA9685 channel 2 (full on = amp powered, full off = amp unpowered). The switch
must be on the POSITIVE side: a low-side switch (the first XY-MOS wiring) is bypassed
by the audio cable's ground, so the amp stayed powered and hissed.

    burst(freq_hz=25000, level_pct=40)

    python -m ultrasonic --freq 25000 --level 40 [--seconds 3]
    python -m ultrasonic --freq 20000 --alt-freq 25000 --seconds 5 --pulse 0.3 --level 80
    python -m ultrasonic --test            # audible 2 kHz file
    python -m ultrasonic --amp on|off|status   # bench: hold the amp power channel
    python -m ultrasonic --generate        # (re)write the tone WAVs

PCA9685 access: inside the robot service the existing servo driver object is
reused. From a separate shell process there is no driver object, and building
one would reset the board under the running service, so only channel 2's own
registers are written over I2C.
"""

import argparse
import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger('Ultrasonic')

ULTRA_USB_ID = '12d1:0010'          # UGREEN / KTMicro "KT USB Audio"
ULTRA_CARD = 'ultra'                # ALSA card id set by the udev rule
ULTRA_PCM = f'plughw:{ULTRA_CARD},0'
ULTRA_MIXER_CONTROL = 'Headphone'   # this card's only playback volume (it has no 'PCM')

AMP_CHANNEL = 2                     # PCA9685 channel driving the amp's MOSFET switch
                                    # (the old winch-servo channel; 0/1 = camera pan/pitch)
AMP_WARMUP_S = 0.25
PCA9685_I2C_BUS = 1
PCA9685_ADDR = 0x40

MIN_COOLDOWN_S = 20.0
MAX_DURATION_S = 5.0
BURST_FREQS = (20000, 25000, 32000, 40000)   # 20 kHz is audible to some people
TEST_FREQ = 2000
ALL_FREQS = BURST_FREQS + (TEST_FREQ,)
SAMPLE_RATE = 96000

TONE_DIR = Path(__file__).resolve().parent / 'assets' / 'ultrasonic'
# Lock + time of the last burst (mtime); shared by the service and the CLI.
STATE_FILE = Path('/home/morgan/dogbot/state/ultrasonic_last')


def is_ultra_card(card) -> bool:
    """True if ALSA card number `card` is the ultrasonic card (matched by USB ID)."""
    try:
        return Path(f'/proc/asound/card{card}/usbid').read_text().strip() == ULTRA_USB_ID
    except OSError:
        return False


def tone_path(freq_hz: int) -> Path:
    return TONE_DIR / f'tone_{int(freq_hz)}hz.wav'


def _tone_pcm(freq_hz: int, seconds: float = 1.0) -> bytes:
    """16-bit mono 96 kHz sine, 0.9 amplitude, 5 ms fades (no click at start/stop)."""
    import numpy as np
    n = int(SAMPLE_RATE * seconds)
    fade = int(0.005 * SAMPLE_RATE)
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    t = np.arange(n) / SAMPLE_RATE
    return (0.9 * env * np.sin(2 * np.pi * freq_hz * t) * 32767).astype('<i2').tobytes()


def _pulsed_pcm(freqs, seconds: float, pulse_s: float, gap_s: float) -> bytes:
    """Pulses of pulse_s separated by gap_s of silence for `seconds`, cycling through freqs."""
    out, t, i = [], 0.0, 0
    silence = b'\x00\x00' * int(SAMPLE_RATE * gap_s)
    while t + pulse_s <= seconds + 1e-9:
        out.append(_tone_pcm(freqs[i % len(freqs)], pulse_s))
        out.append(silence)
        t += pulse_s + gap_s
        i += 1
    return b''.join(out[:-1])


def generate_tones() -> None:
    """Write the 1 s tone files."""
    import wave
    TONE_DIR.mkdir(parents=True, exist_ok=True)
    for freq in ALL_FREQS:
        with wave.open(str(tone_path(freq)), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(_tone_pcm(freq))
        print(f'wrote {tone_path(freq)}')


# ----------------------------------------------------------------- amp power

def _set_amp_raw(on: bool) -> None:
    """Write only the amp channel's registers (no reset, no MODE/prescale changes)."""
    from smbus2 import SMBus
    base = 0x06 + 4 * AMP_CHANNEL          # LEDn_ON_L, ON_H, OFF_L, OFF_H
    with SMBus(PCA9685_I2C_BUS) as bus:
        if on:
            mode1 = bus.read_byte_data(PCA9685_ADDR, 0x00)
            if mode1 & 0x10:               # board asleep (service never ran): wake only
                bus.write_byte_data(PCA9685_ADDR, 0x00, mode1 & ~0x10)
                time.sleep(0.001)
            bus.write_byte_data(PCA9685_ADDR, base + 0, 0x00)
            bus.write_byte_data(PCA9685_ADDR, base + 2, 0x00)
            bus.write_byte_data(PCA9685_ADDR, base + 1, 0x10)   # full on
            bus.write_byte_data(PCA9685_ADDR, base + 3, 0x00)
        else:
            bus.write_byte_data(PCA9685_ADDR, base + 3, 0x10)   # full off (wins over on)
            bus.write_byte_data(PCA9685_ADDR, base + 1, 0x00)


def _set_amp(on: bool) -> None:
    """Amp channel full on / full off, via the existing driver object when there is one."""
    sc = sys.modules.get('core.hardware.servo_controller')
    servo = getattr(sc, '_servo_instance', None) if sc else None
    pca = getattr(servo, 'pca', None)
    if pca is not None:
        pca.channels[AMP_CHANNEL].duty_cycle = 0xFFFF if on else 0
    else:
        _set_amp_raw(on)


# ------------------------------------------------------------------ logging

def _log_burst(payload: Dict[str, Any], dog_id: Optional[str], dog_name: Optional[str],
               session_id: Optional[str], standalone: bool) -> None:
    """Row in wimz.db (the event store Silent Guardian uses). Never raises."""
    try:
        from core.data import get_wimz_store
        store = get_wimz_store()
        wimz_dog = None
        if (dog_id and dog_id != 'unknown') or dog_name:
            wimz_dog = store.get_or_create_dog(
                legacy_id=dog_id if dog_id and dog_id != 'unknown' else None, name=dog_name)
        payload = dict(payload, dog_name=dog_name)
        store.log_event(session_id, 'ultrasonic_burst', payload, dog_id=wimz_dog,
                        label_source='human' if standalone else 'auto_rule')
        if standalone:
            store.flush()
            if not session_id:      # don't leave the CLI's own session open
                store.end_session(store.ensure_ambient_session())
    except Exception as e:
        logger.warning(f"ultrasonic burst not logged: {e}")


# -------------------------------------------------------------------- burst

def burst(freq_hz: int = 25000, level_pct: int = 40, dog_id: Optional[str] = None,
          dog_name: Optional[str] = None, session_id: Optional[str] = None,
          source: str = 'manual', cooldown_s: float = MIN_COOLDOWN_S,
          duration_s: float = 1.0, pulse_s: float = 0.0, gap_s: float = 0.2,
          alt_freq_hz: Optional[int] = None,
          _standalone: bool = False) -> Dict[str, Any]:
    """Power the amp, play one tone on the ultra card, power the amp off.

    duration_s (max 5) is the total length. pulse_s > 0 makes it a pulsed episode:
    pulses of pulse_s with gap_s of silence, alternating freq_hz / alt_freq_hz.

    Returns {'ok': bool, ...}; a refusal (cooldown, busy, missing file/card) is
    ok=False with 'reason' and never powers the amp. cooldown_s is floored at 20 s,
    except for the audible 2 kHz test tone.
    """
    freq_hz = int(freq_hz)
    level_pct = max(0, min(100, int(level_pct)))
    duration_s = max(0.05, min(MAX_DURATION_S, float(duration_s)))
    pulse_s = max(0.0, min(duration_s, float(pulse_s or 0.0)))
    gap_s = max(0.02, float(gap_s))
    freqs = [freq_hz] + ([int(alt_freq_hz)] if pulse_s and alt_freq_hz else [])
    if any(f not in BURST_FREQS for f in freqs[1:]):
        return {'ok': False, 'reason': f'no tone for {alt_freq_hz} Hz'}
    wav = tone_path(freq_hz)
    if freq_hz not in ALL_FREQS or not wav.exists():
        return {'ok': False, 'reason': f'no tone file for {freq_hz} Hz ({wav})'}
    if not Path(f'/proc/asound/{ULTRA_CARD}').exists():
        return {'ok': False, 'reason': f'sound card "{ULTRA_CARD}" not present'}
    cooldown_s = 0.0 if freq_hz == TEST_FREQ else max(MIN_COOLDOWN_S, float(cooldown_s))

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {'ok': False, 'reason': 'another burst is in progress'}
        since = time.time() - os.path.getmtime(STATE_FILE)
        if freq_hz != TEST_FREQ and os.path.getsize(STATE_FILE) and since < cooldown_s:
            return {'ok': False, 'reason': f'cooldown ({cooldown_s - since:.0f}s left)'}

        error = None
        try:
            _set_amp(True)
            time.sleep(AMP_WARMUP_S)
            subprocess.run(['amixer', '-q', '-c', ULTRA_CARD, 'sset', ULTRA_MIXER_CONTROL,
                            f'{level_pct}%', 'unmute'], check=True, timeout=5,
                           capture_output=True, text=True)
            if duration_s == 1.0 and not pulse_s:
                subprocess.run(['aplay', '-q', '-D', ULTRA_PCM, str(wav)], check=True,
                               timeout=10, capture_output=True, text=True)
            else:   # other lengths: same tone synthesised and piped to aplay
                subprocess.run(['aplay', '-q', '-D', ULTRA_PCM, '-t', 'raw', '-f', 'S16_LE',
                                '-r', str(SAMPLE_RATE), '-c', '1'],
                               input=(_pulsed_pcm(freqs, duration_s, pulse_s, gap_s) if pulse_s
                                      else _tone_pcm(freq_hz, duration_s)), check=True,
                               timeout=10 + duration_s, capture_output=True)
        except subprocess.CalledProcessError as e:
            error = e.stderr if isinstance(e.stderr, str) else (e.stderr or b'').decode(errors='replace')
            error = error.strip() or str(e)
        except Exception as e:
            error = str(e)
        finally:
            _set_amp(False)       # amp must never stay powered
            if freq_hz != TEST_FREQ:
                lock.write('.')   # non-empty + fresh mtime = start of cooldown
                lock.flush()

    result = {'ok': error is None, 'freq_hz': freq_hz, 'level_pct': level_pct,
              'duration_s': duration_s}
    if pulse_s:
        result.update(pulse_s=pulse_s, gap_s=gap_s, freqs=freqs)
    if error:
        result['reason'] = error
        logger.error(f"Ultrasonic burst failed: {error}")
    else:
        logger.info(f"Ultrasonic burst {freq_hz} Hz @ {level_pct}% "
                    f"(dog={dog_name or dog_id or 'unknown'}, source={source})")
    _log_burst({**{k: v for k, v in result.items() if k not in ('ok', 'reason')},
                'source': source,
                'ok': error is None, 'error': error},
               dog_id, dog_name, session_id, _standalone)
    return result


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ultrasonic', description=__doc__.split('\n\n')[0])
    ap.add_argument('--freq', type=int, default=25000, choices=BURST_FREQS)
    ap.add_argument('--level', type=int, default=40, help='ultra card volume, percent')
    ap.add_argument('--seconds', type=float, default=1.0, help=f'burst length (max {MAX_DURATION_S:.0f})')
    ap.add_argument('--pulse', type=float, default=0.0, help='pulse length in s (0 = continuous)')
    ap.add_argument('--gap', type=float, default=0.2, help='silence between pulses, s')
    ap.add_argument('--alt-freq', type=int, choices=BURST_FREQS, help='alternate pulses with this')
    ap.add_argument('--test', action='store_true', help='play the audible 2 kHz file')
    ap.add_argument('--amp', choices=('on', 'off', 'status'),
                    help='bench test: switch the amp power channel and leave it (no sound)')
    ap.add_argument('--generate', action='store_true', help='(re)write the tone WAV files')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    if args.generate:
        generate_tones()
        return 0
    if args.amp:
        if args.amp != 'status':
            _set_amp_raw(args.amp == 'on')
        from smbus2 import SMBus
        with SMBus(PCA9685_I2C_BUS) as bus:
            on = bool(bus.read_byte_data(PCA9685_ADDR, 0x07 + 4 * AMP_CHANNEL) & 0x10)
        print(f'PCA9685 channel {AMP_CHANNEL} (amp power): {"ON" if on else "OFF"}')
        return 0
    # A terminated CLI must still run burst()'s finally (amp off).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    res = burst(TEST_FREQ if args.test else args.freq, args.level,
                source='cli_test' if args.test else 'cli', duration_s=args.seconds,
                pulse_s=args.pulse, gap_s=args.gap, alt_freq_hz=args.alt_freq,
                _standalone=True)
    print(res)
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
