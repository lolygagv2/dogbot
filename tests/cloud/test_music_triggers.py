#!/usr/bin/env python3
"""music_plan triggers run on the robot (relay brief 2026-11-18).

Fake audio / volume / clock / emitter. Checks every trigger (on_mode_start
with delay + skip when the mode ended, after_treat rolling window + reset,
on_bark_episode at min_severity without waiting for the end, at_time in the
dog's timezone), enabled:false, quiet hours (no start, and a running session
is stopped), no stacking, plan volume as a temporary level that is restored,
duration_min, music start/stop events, and the settings validation.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_music_triggers.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='musictest_')

import services.cloud.music_triggers as mt  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeAudio:
    def __init__(self):
        self.busy = False
        self.plays, self.stops = [], 0
    def is_busy(self): return self.busy
    def play_file(self, path):
        self.plays.append(path); self.busy = True
        return {'success': True}
    def stop(self):
        self.stops += 1; self.busy = False


class FakeVol:
    def __init__(self, v=70):
        self.v, self.calls = v, []
    def get_volume(self): return self.v
    def set_volume(self, v, persist=True):
        self.calls.append((v, persist)); self.v = v
        return True


class Rig:
    def __init__(self, plan, quiet=None, tz='America/New_York', t0=None):
        # 2026-11-18 12:00:00 America/New_York
        self.t = 1795021200.0 if t0 is None else t0
        self.plan, self.quiet, self.mode = plan, quiet, 'idle'
        self.audio, self.vol, self.events, self.sg_music = FakeAudio(), FakeVol(), [], False
        self.m = mt.MusicTriggers(plan_source=lambda: self.plan, quiet_source=lambda: self.quiet,
                                  tz_source=lambda: tz, audio=self.audio,
                                  emit=lambda t, d: self.events.append((t, dict(d))),
                                  mode_source=lambda: self.mode,
                                  resolve_track=lambda tr: f'/songs/{tr}.mp3' if tr.startswith('song_') else None,
                                  other_music_playing=lambda: self.sg_music, volume=self.vol,
                                  clock=lambda: self.t)


def wait_for(cond, s=2.0):
    end = time.time() + s
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


PLAN = {'enabled': True, 'volume': 40,
        'on_mode_start': {'mode': 'silent_guardian', 'delay_s': 0, 'track': 'song_calm_01', 'duration_min': 30},
        'after_treat': {'every_n': 3, 'minutes': 10, 'track': 'song_calm_02', 'duration_min': 20},
        'on_bark_episode': {'min_severity': 'med', 'track': 'song_calm_01', 'duration_min': 15},
        'at_time': [{'time': '13:00', 'track': 'song_calm_02', 'duration_min': 20}]}
OFF = {'enabled': True, 'volume': 40, 'on_mode_start': None, 'after_treat': None, 'on_bark_episode': None,
       'at_time': []}

# --- all off by default
r = Rig(dict(OFF))
r.mode = 'silent_guardian'; r.m.on_mode_change('silent_guardian')
for _ in range(5):
    r.m.on_treat()
r.m.on_bark_progress('ep1', 'high'); r.m.tick(); time.sleep(0.1)
check(not r.audio.plays and not r.events and r.m.status()['armed'] == [], "every trigger null / [] -> nothing plays")

# --- on_mode_start (delay 0), events, volume, duration, stop
r = Rig(dict(PLAN))
r.mode = 'silent_guardian'; r.m.on_mode_change('silent_guardian')
check(wait_for(lambda: r.audio.plays) and r.audio.plays == ['/songs/song_calm_01.mp3'], "on_mode_start plays its own track")
check(r.events == [('music', {'action': 'start', 'track': 'song_calm_01', 'trigger': 'on_mode_start'})],
      f"music start event {{action, track, trigger}} ({r.events})")
check(r.vol.calls == [(40, False)], f"plan volume applied as a temporary level, not saved ({r.vol.calls})")
r.t += 29 * 60; r.audio.busy = False; r.m.tick()
check(len(r.audio.plays) == 2 and r.m.session_active(), "track ended before duration_min: it plays again")
r.t += 61; r.m.tick()
ev = r.events[-1]
check(ev[1]['action'] == 'stop' and ev[1]['track'] == 'song_calm_01' and 1795 <= ev[1]['duration_s'] <= 1805
      and r.audio.stops == 1 and not r.m.session_active(), f"stops after duration_min; stop event has duration_s ({ev[1]})")
check(r.vol.calls[-1] == (70, False) and r.vol.v == 70, "owner's volume is back afterwards")

# --- on_mode_start: mode ends before the delay is up -> skipped
p = dict(PLAN); p['on_mode_start'] = dict(PLAN['on_mode_start'], delay_s=0.3)
r = Rig(p)
r.mode = 'silent_guardian'; r.m.on_mode_change('silent_guardian')
time.sleep(0.1); r.mode = 'idle'; r.m.on_mode_change('idle'); time.sleep(0.4)
check(not r.audio.plays, "on_mode_start: mode ended before delay_s -> skipped")
r.mode = 'coach'; r.m.on_mode_change('coach'); time.sleep(0.45)
check(not r.audio.plays, "on_mode_start: a different mode does not fire it")
r.mode = 'silent_guardian'; r.m.on_mode_change('silent_guardian')
check(wait_for(lambda: r.audio.plays, 1.5), "on_mode_start: fires delay_s after entering the mode")

# --- after_treat: every_n in a rolling window, then reset
r = Rig(dict(PLAN))
r.m.on_treat(); r.t += 11 * 60; r.m.on_treat(); r.t += 60; r.m.on_treat(); time.sleep(0.1)
check(not r.audio.plays, "after_treat: 3 treats but the first fell out of the 10 min window -> no music")
r.t += 60; r.m.on_treat()
check(wait_for(lambda: r.audio.plays) and r.audio.plays == ['/songs/song_calm_02.mp3']
      and r.events[0][1]['trigger'] == 'after_treat', "after_treat: 3 treats inside the window -> plays")
r.m._end_session('test'); r.audio.plays.clear()
r.m.on_treat(); r.m.on_treat(); time.sleep(0.1)
check(not r.audio.plays, "after_treat: count was reset after firing")

# --- on_bark_episode: at min_severity, once per episode, without waiting for the end
r = Rig(dict(PLAN))
r.m.on_bark_progress('ep1', 'low'); time.sleep(0.05)
check(not r.audio.plays, "on_bark_episode: below min_severity -> nothing")
r.m.on_bark_progress('ep1', 'med')
check(wait_for(lambda: r.audio.plays) and r.events[0][1]['trigger'] == 'on_bark_episode',
      "on_bark_episode: fires as soon as the running episode reaches med")
r.m.on_bark_progress('ep1', 'high'); time.sleep(0.05)
check(len(r.audio.plays) == 1, "on_bark_episode: once per episode")
# no stacking: a second trigger while music plays is dropped
res = r.m.fire('after_treat', PLAN['after_treat'])
check(res['ok'] is False and res['reason'] == 'music_already_playing' and len(r.audio.plays) == 1
      and r.audio.stops == 0, "trigger while music is playing -> dropped, current track untouched")
r = Rig(dict(PLAN)); r.sg_music = True
check(r.m.fire('at_time', PLAN['at_time'][0])['reason'] == 'music_already_playing' and not r.audio.plays,
      "Silent Guardian's own music playing -> trigger dropped")

# --- at_time in the dog's timezone, once a day
r = Rig(dict(PLAN))
r.m.tick(); time.sleep(0.05)
check(not r.audio.plays, "at_time: 12:00 local, slot 13:00 -> nothing")
r.t += 3600; r.m.tick()
check(wait_for(lambda: r.audio.plays) and r.events[0][1] == {'action': 'start', 'track': 'song_calm_02', 'trigger': 'at_time'},
      "at_time: fires at 13:00 America/New_York")
r.m._end_session('test'); r.audio.plays.clear(); r.t += 20; r.m.tick(); time.sleep(0.05)
check(not r.audio.plays, "at_time: the same slot does not fire twice in a day")
r.t += 86400 - 20; r.m.tick()
check(wait_for(lambda: r.audio.plays), "at_time: fires again the next day")

# --- enabled:false and quiet hours
r = Rig(dict(PLAN, enabled=False))
r.m.on_bark_progress('e', 'high'); r.m.on_treat(); r.m.on_treat(); r.m.on_treat(); r.t += 3600; r.m.tick(); time.sleep(0.1)
check(not r.audio.plays, "enabled:false silences every trigger")
r = Rig(dict(PLAN), quiet={'enabled': True, 'start': '11:00', 'end': '14:00'})
check(r.m.fire('at_time', PLAN['at_time'][0])['reason'] == 'quiet_hours' and not r.audio.plays,
      "quiet hours: nothing starts")
r = Rig(dict(PLAN), quiet={'enabled': True, 'start': '12:30', 'end': '07:00'})
r.m.fire('on_bark_episode', dict(PLAN['on_bark_episode'], duration_min=120))
r.t += 31 * 60; r.m.tick()
check(r.audio.stops == 1 and r.events[-1][1]['action'] == 'stop' and r.events[-1][1]['reason'] == 'quiet_hours',
      "quiet hours begin while music plays -> stopped")
check(mt.in_quiet_hours({'enabled': True, 'start': '22:00', 'end': '07:00'}, 23 * 60)
      and mt.in_quiet_hours({'enabled': True, 'start': '22:00', 'end': '07:00'}, 6 * 60)
      and not mt.in_quiet_hours({'enabled': True, 'start': '22:00', 'end': '07:00'}, 12 * 60)
      and not mt.in_quiet_hours({'enabled': False, 'start': '22:00', 'end': '07:00'}, 23 * 60),
      "quiet hours window across midnight; enabled:false = never quiet")
r = Rig(dict(PLAN))
check(r.m.fire('at_time', {'time': '13:00', 'track': 'nope', 'duration_min': 5})['reason'] == 'unknown_track',
      "unknown track -> not started, said so")

# --- settings validation: new shape accepted, old keys ignored, bad values refused
from services.cloud.remote_settings import RemoteSettings  # noqa: E402
v = RemoteSettings._validate_music_plan
ok = True
try:
    v(PLAN); v(OFF); v(dict(OFF, on_departure={'delay_s': -5}, schedule='junk'))
except Exception as e:
    ok = False; print(e)
check(ok, "new music_plan shape validates; on_departure / schedule are ignored")
bad = 0
for p in (dict(OFF, on_mode_start={'mode': 'idle', 'delay_s': 5}), dict(OFF, after_treat={'every_n': 0}),
          dict(OFF, on_bark_episode={'min_severity': 'huge'}), dict(OFF, at_time=[{'time': '25:00'}]),
          dict(OFF, at_time=[{'time': '10:00'}] * 13), dict(OFF, volume=150)):
    try:
        v(p)
    except Exception:
        bad += 1
check(bad == 6, f"out-of-range trigger values refused ({bad}/6)")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
