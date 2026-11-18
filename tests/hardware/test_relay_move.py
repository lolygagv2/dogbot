#!/usr/bin/env python3
"""Relay `move` command (2026-11-07): bounded open-loop base moves.

Fake motor bus + fake pan/tilt + fake clock. Checks: caps (50 cm / 90 deg),
duration from calibration, 5 s hard cap -> timeout, obstacle <30 cm -> stop
'obstacle', cliff -> 'edge', SG mid-intervention -> ok:false busy (before and
during), motors zeroed after every outcome, gimbal centered before + during a
forward move only, ramp never exceeds speed, 'stop' action, bad args, and the
dispatcher wiring (tool 'move' registered).

Run: /home/morgan/dogbot/env_new/bin/python tests/hardware/test_relay_move.py
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='movetest_')

import services.motion.relay_move as rm  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeClock:
    def __init__(self):
        self.t = 0.0
    def now(self):
        return self.t
    def sleep(self, s):
        self.t += s


class FakeBus:
    running = True
    def __init__(self):
        self.cmds = []
    def send_command(self, cmd):
        self.cmds.append((cmd.left_speed, cmd.right_speed, cmd.source.value))
        return True


class FakePT:
    def __init__(self):
        self.centers = 0
        self.manual = []
    def center_camera(self, reason=''):
        self.centers += 1
        return True
    def set_manual_camera(self, a):
        self.manual.append(a)


CFG = dict(rm.DEFAULTS, cm_per_s=10.0, deg_per_s=45.0)   # 50 cm -> 5 s, 90 deg -> 2 s


def make(sg=lambda: False):
    clk = FakeClock(); bus = FakeBus(); pt = FakePT()
    mv = rm.RelayMove(motor_bus=bus, pantilt=pt, config=dict(CFG), sg_busy=sg, clock=clk.now, sleep=clk.sleep)
    return mv, bus, pt, clk


def moving(bus):
    return [c for c in bus.cmds if c[0] or c[1]]


rm.set_obstacle_provider(None); rm.set_cliff_provider(None)

# forward 20 cm -> 2.0 s completed, motors off after, gimbal centered before + during, manual toggled
mv, bus, pt, clk = make()
out = mv.execute('forward', 20)
check(out['ok'] and out['stopped_reason'] == 'completed' and abs(out['planned_s'] - 2.0) < 0.01,
      f"forward 20cm -> completed in planned 2.0s ({out['stopped_reason']}, {out['planned_s']})")
check(bus.cmds[-1][:2] == (0, 0) and bus.cmds[-1][2] == 'relay', f"motors off after completed ({bus.cmds[-1]})")
check(all(c[0] == c[1] and 0 < c[0] <= CFG['speed_pct'] for c in moving(bus)), "forward: both wheels +, never above slow speed")
check(pt.centers >= 2 and pt.manual == [True, False], f"gimbal centered before+during forward, manual on/off ({pt.centers}, {pt.manual})")
check(out['sensors'] == {'obstacle': False, 'cliff': False, 'bump': False}, "reply reports no sensors fitted")

# cap: forward 500 cm -> 50 cm -> 5.0 s; hits hard cap -> timeout
mv, bus, pt, clk = make()
out = mv.execute('forward', 500)
check(out['amount'] == 50 and out['stopped_reason'] == 'timeout' and bus.cmds[-1][:2] == (0, 0),
      f"forward 500 -> capped 50cm, hard cap -> timeout, motors off ({out['amount']}, {out['stopped_reason']})")

# back 10 cm: no gimbal centering, wheels negative
mv, bus, pt, clk = make()
out = mv.execute('back', 10)
check(out['ok'] and pt.centers == 0 and pt.manual == [] and all(c[0] < 0 and c[1] < 0 for c in moving(bus)),
      f"back: no gimbal action, both wheels negative ({pt.centers})")

# turns: cap 90 deg -> 2 s; opposite wheel signs
mv, bus, pt, clk = make()
out = mv.execute('turn_left', 720)
check(out['requested'] == 90 and out['measured'] is False and out['heading_source'] == 'none'
      and abs(out['amount'] - 90) < 3 and out['unit'] == 'deg' and abs(out['planned_s'] - 2.0) < 0.01
      and all(c[0] < 0 < c[1] for c in moving(bus)), f"turn_left 720 -> 90deg 2.0s, L- R+ ({out['planned_s']})")
mv, bus, pt, clk = make()
out = mv.execute('turn_right', 45)
check(out['ok'] and all(c[0] > 0 > c[1] for c in moving(bus)) and abs(out['planned_s'] - 1.0) < 0.01, "turn_right 45 -> 1.0s, L+ R-")

# obstacle under 30 cm during forward -> 'obstacle', motors off
mv, bus, pt, clk = make()
rm.set_obstacle_provider(lambda: 100.0 if clk.t < 0.5 else 25.0)
out = mv.execute('forward', 40)
check(out['ok'] is False and out['stopped_reason'] == 'obstacle' and 0.4 < out['duration_ms'] / 1000 < 0.7 + CFG['settle_s'] and bus.cmds[-1][:2] == (0, 0),
      f"obstacle <30cm mid-move -> stopped 'obstacle' at {out['duration_ms']}ms, motors off")
check(out['sensors']['obstacle'] is True, "reply reports obstacle sensor present")
# obstacle at exactly 30 cm does not stop; back ignores obstacle
rm.set_obstacle_provider(lambda: 30.0)
mv, bus, pt, clk = make()
check(mv.execute('forward', 10)['stopped_reason'] == 'completed', "obstacle at 30cm (not under) -> completed")
rm.set_obstacle_provider(lambda: 5.0)
mv, bus, pt, clk = make()
check(mv.execute('back', 10)['stopped_reason'] == 'completed', "back ignores obstacle sensor")
rm.set_obstacle_provider(None)

# cliff -> 'edge'
rm.set_cliff_provider(lambda: clk.t >= 0.3)
mv, bus, pt, clk = make()
out = mv.execute('forward', 40)
check(out['stopped_reason'] == 'edge' and bus.cmds[-1][:2] == (0, 0), f"cliff -> stopped 'edge', motors off ({out['stopped_reason']})")
rm.set_cliff_provider(None)

# SG mid-intervention: refused before, aborted during (motors off)
mv, bus, pt, clk = make(sg=lambda: True)
out = mv.execute('forward', 10)
check(out['ok'] is False and out['error']['code'] == 'robot_busy' and bus.cmds == [], f"SG intervention -> ok:false robot_busy, no motor command ({out})")
mv, bus, pt, clk = make(sg=lambda: clk.t > 0.4)
out = mv.execute('forward', 40)
check(out['ok'] is False and out['stopped_reason'] == 'busy' and bus.cmds[-1][:2] == (0, 0) and moving(bus),
      f"SG starts mid-move -> aborted busy, motors off ({out['stopped_reason']})")

# cancel event (preemptive standby) stops the move
mv, bus, pt, clk = make()
ev = threading.Event()
mv._sleep = lambda s: (clk.sleep(s), ev.set() if clk.t > 0.3 else None)
out = mv.execute('forward', 40)
check(out['stopped_reason'] == 'timeout' and out['duration_ms'] < 1000 and bus.cmds[-1][:2] == (0, 0), "cancel -> stopped early, motors off") if False else None
mv, bus, pt, clk = make()
ev = threading.Event()
mv._sleep = lambda s: (clk.sleep(s), ev.set() if clk.t > 0.3 else None)
out = mv.execute('forward', 40, cancel=ev)
check(out['duration_ms'] < 1000 and bus.cmds[-1][:2] == (0, 0), f"cancel event -> stopped early ({out['duration_ms']}ms), motors off")

# relay bd3290a reply shape: blocked/reason mapping, 'backward' alias, unit
rm.set_obstacle_provider(lambda: 10.0)
mv, bus, pt, clk = make()
out = mv.execute('forward', 20, unit='cm')
check(out['ok'] is False and out['blocked'] is True and out['reason'] == 'too_close' and out['error']['code'] == 'too_close',
      f"relay shape: obstacle -> ok:false blocked:true reason:too_close ({out['reason']})")
rm.set_obstacle_provider(None)
rm.set_cliff_provider(lambda: True)
mv, bus, pt, clk = make()
out = mv.execute('forward', 20)
check(out['ok'] is False and out['blocked'] and out['reason'] == 'cliff', "relay shape: edge -> reason:cliff")
rm.set_cliff_provider(None)
rm.set_bump_provider(lambda: clk.t > 0.2)
mv, bus, pt, clk = make()
out = mv.execute('forward', 20)
check(out['ok'] is False and out['blocked'] and out['reason'] == 'bump' and bus.cmds[-1][:2] == (0, 0), "relay shape: bumper -> reason:bump, motors off")
rm.set_bump_provider(None)
mv, bus, pt, clk = make()
out = mv.execute('backward', 10, unit='cm')
check(out['ok'] and out['blocked'] is False and out['reason'] == 'completed' and out['action'] == 'back' and out['unit'] == 'cm',
      "relay shape: 'backward' alias, completed -> ok:true blocked:false")
check(mv.execute('forward', 10, unit='furlongs')['error']['code'] == 'bad_unit', "bad unit refused")

# stop action, bad args, motors unavailable
mv, bus, pt, clk = make()
out = mv.execute('stop', None)
check(out['ok'] and out['stopped_reason'] == 'completed' and bus.cmds == [(0, 0, 'relay')], f"stop -> motors off ({bus.cmds})")
check(mv.execute('sideways', 1)['error']['code'] == 'bad_action', "bad action refused")
check(mv.execute('forward', 'lots')['error']['code'] == 'bad_amount' and mv.execute('forward', 0)['error']['code'] == 'bad_amount', "bad amount refused")
bus.running = False
check(mv.execute('forward', 10)['error']['code'] == 'motors_unavailable', "motor bus down -> motors_unavailable")

# config: robot_config.yaml relay_move: is read; unit profile overrides; bad values ignored
import yaml as _yaml  # noqa: E402
tmp = Path(tempfile.mkdtemp(prefix='movecfg_')) / 'robot_config.yaml'
tmp.write_text(_yaml.safe_dump({'relay_move': {'cm_per_s': 35.0, 'deg_per_s': 103.0, 'speed_pct': 40, 'max_cm': 'junk'}}))
cfg = rm.load_config(str(tmp))
check(cfg['cm_per_s'] == 35.0 and cfg['deg_per_s'] == 103.0 and cfg['speed_pct'] == 40 and cfg['max_cm'] == rm.DEFAULTS['max_cm'],
      f"load_config reads robot_config relay_move, ignores bad value ({cfg['cm_per_s']}, {cfg['deg_per_s']}, {cfg['max_cm']})")
check(rm.load_config('/nonexistent/robot_config.yaml')['cm_per_s'] == rm.DEFAULTS['cm_per_s'], "missing robot_config -> code defaults")
live = rm.load_config()
check(live['cm_per_s'] == 35.0 and live['deg_per_s'] == 103.0, f"live robot_config.yaml carries the measured calibration ({live['cm_per_s']}, {live['deg_per_s']})")

# dispatcher wiring
import services.cloud.tool_dispatcher as td  # noqa: E402
d = td.ToolDispatcher()
check(d.has_tool('move') and d._tools['move'].timeout_s >= 5.0, "dispatcher registers 'move' with >=5s timeout")
rm._instance = rm.RelayMove(motor_bus=FakeBus(), pantilt=FakePT(), config=dict(CFG), sg_busy=lambda: False,
                            clock=FakeClock().now, sleep=lambda s: None)
clk2 = FakeClock(); rm._instance._clock = clk2.now; rm._instance._sleep = clk2.sleep
out = d._tool_move({'action': 'Forward', 'amount': 5}, threading.Event())
check(out['ok'] and out['stopped_reason'] == 'completed' and out['amount'] == 5, f"dispatcher move tool -> executes ({out.get('stopped_reason')})")


# ---------------------------------------------------------------- 2026-11-18
# Closed-loop turns on a simulated floor. The fake robot yaws at a rate that
# depends on wheel power (carpet: nothing below a breakaway power, slow above),
# and the fake heading reads that yaw the way the camera would between frames.

class Sim:
    """Fake bus + clock + heading sharing one simulated yaw."""
    running = True

    def __init__(self, rate_at, scale_err=1.0, blind_after=None):
        self.t = 0.0
        self.yaw = 0.0            # + = right
        self.cmd = (0, 0)
        self.cmds = []
        self.rate_at = rate_at
        self.max_power = 0
        self.blind_after = blind_after
        self.frames = 0

    # bus
    def send_command(self, cmd):
        self.cmd = (cmd.left_speed, cmd.right_speed)
        self.cmds.append(self.cmd)
        self.max_power = max(self.max_power, abs(cmd.left_speed))
        return True

    # clock
    def now(self):
        return self.t

    def sleep(self, s):
        l, r = self.cmd
        if l and r and (l > 0) != (r > 0):
            self.yaw += (1 if l > 0 else -1) * self.rate_at(abs(l)) * s
        self.t += s

    # heading
    class Snap:
        usable = True
        def __init__(self, yaw, ts):
            self.yaw, self.ts = yaw, ts

    def grab(self, newer_than=0.0, timeout_s=1.0):
        self.frames += 1
        if self.blind_after is not None and self.frames > self.blind_after:
            return None
        return Sim.Snap(self.yaw, self.t)

    def wait_still(self, after=0.0, timeout_s=1.5):
        self.sleep(0.4)                       # two frames at 5 fps
        snap = self.grab()
        return snap, snap is not None

    def yaw_between(self, a, b):
        return {'yaw_deg': b.yaw - a.yaw, 'inliers': 300, 'matches': 320}


def make_sim(rate_at, **kw):
    sim = Sim(rate_at, **kw)
    mv = rm.RelayMove(motor_bus=sim, pantilt=FakePT(), config=dict(rm.DEFAULTS), sg_busy=lambda: False,
                      clock=sim.now, sleep=sim.sleep, heading=sim, wall=sim.now)
    return mv, sim


hard_floor = lambda p: 2.6 * p                          # 40 % -> ~104 deg/s
carpet = lambda p: 0.0 if p < 38 else 18.0 + (p - 40) * 2.2   # 40 % -> 18 deg/s, 60 % -> 62 deg/s
dead = lambda p: 0.0

mv, sim = make_sim(hard_floor)
out = mv.execute('turn_right', 90)
check(out['ok'] and out['reason'] == 'completed' and abs(sim.yaw - 90) <= 4.5 and abs(out['amount'] - sim.yaw) < 0.2
      and out['measured'] and out['heading_source'] == 'camera' and out['requested'] == 90 and out['stopped_confirmed'],
      f"hard floor turn_right 90 -> real {sim.yaw:.1f}, reported {out['amount']} in {out['bursts']} bursts, {out['duration_ms']}ms")
check(sim.cmds[-1] == (0, 0) and sim.max_power <= 60, f"hard floor: motors off, power never above 60 % ({sim.max_power})")

mv, sim = make_sim(carpet)
out = mv.execute('turn_right', 90)
check(out['amount'] == round(sim.yaw, 1) and sim.yaw > 60 and sim.max_power <= 60
      and out['duration_ms'] <= 9500,
      f"carpet turn_right 90 -> real {sim.yaw:.1f}, reported {out['amount']} ({out['reason']}, {out['bursts']} bursts, "
      f"{out['duration_ms']}ms, up to {sim.max_power}%)")
check(out['reason'] == 'completed' or (out['reason'] == 'timeout' and out['amount'] < 86),
      "carpet: says completed only when inside tolerance, otherwise timeout with the real figure")

mv, sim = make_sim(carpet)
out = mv.execute('turn_left', 45)
check(out['ok'] and abs(-sim.yaw - 45) <= 4.5 and abs(out['amount'] + sim.yaw) < 0.2,
      f"carpet turn_left 45 -> real {-sim.yaw:.1f}, reported {out['amount']}")

mv, sim = make_sim(dead)
out = mv.execute('turn_right', 90)
check(out['ok'] is False and out['error']['code'] == 'no_rotation' and out['amount'] == 0 and sim.cmds[-1] == (0, 0),
      f"wheels spin, no rotation -> ok:false no_rotation, amount 0 ({out['reason']}, {out['bursts']} bursts)")

mv, sim = make_sim(hard_floor, blind_after=0)
out = mv.execute('turn_right', 90)
check(out['ok'] and out['measured'] is False and out['heading_source'] == 'none' and abs(sim.yaw - 90) < 12,
      f"camera cannot see -> timed open-loop turn, flagged measured:false (real {sim.yaw:.1f}, reported {out['amount']})")

mv, sim = make_sim(hard_floor, blind_after=3)
out = mv.execute('turn_right', 90)
check(out['reason'] in ('heading_lost', 'completed') and out['measured'] is False and sim.cmds[-1] == (0, 0),
      f"camera loses track mid-turn -> stops, measured:false ({out['reason']}, reported {out['amount']}, real {sim.yaw:.1f})")

# forward: reply only after the settle wait, and it says whether the camera confirmed the stop
mv, sim = make_sim(hard_floor)
out = mv.execute('forward', 15)
check(out['ok'] and out['stopped_confirmed'] is True and out['duration_ms'] >= 15 / 35.0 * 1000 + 350,
      f"forward: result after the stop is confirmed ({out['duration_ms']}ms incl. settle)")
mv, bus, pt, clk = make()
out = mv.execute('back', 10)
check(out['stopped_confirmed'] is False and out['duration_ms'] >= 1000 + CFG['settle_s'] * 1000 - 60,
      f"no camera: result after the fixed coast time ({out['duration_ms']}ms)")

# calibration spin: camera reads 20 % short (lens model off) -> scale 1.25, saved
import services.motion.visual_heading as vh  # noqa: E402
vh.CALIBRATION_FILE = os.path.join(os.environ['WIMZ_DATA_ROOT'], 'turn_calibration.json')
_orig_save, _orig_load = vh.save_calibration, vh.load_calibration
vh.save_calibration = lambda data, path=None: _orig_save(data, vh.CALIBRATION_FILE)
vh.load_calibration = lambda path=None: _orig_load(vh.CALIBRATION_FILE)
mv, sim = make_sim(hard_floor)
sim.scale, sim.hfov_deg, sim.calibrated = 1.0, 102.0, False
def _yb(a, b, sim=sim):
    d = (b.yaw - a.yaw + 180.0) % 360.0 - 180.0
    if abs(d) > 45:
        return None
    return {'yaw_deg': d * 0.8 * sim.scale, 'inliers': 200, 'matches': 220}
sim.yaw_between = _yb
out = mv.calibrate_turn(surface='hard')
check(out['ok'] and abs(out['yaw_scale'] - 1.25) < 0.03 and abs(sim.scale - 1.25) < 0.03
      and 320 < sim.yaw < 400 and _orig_load(vh.CALIBRATION_FILE).get('surfaces', {}).get('hard'),
      f"calibration spin: one revolution (real {sim.yaw:.0f} deg) -> yaw_scale {out.get('yaw_scale')}, saved per surface")
out = mv.execute('turn_right', 90)
check(abs(out['amount'] - 90) <= 4.5, f"after calibration a 90 turn reads {out['amount']}")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
