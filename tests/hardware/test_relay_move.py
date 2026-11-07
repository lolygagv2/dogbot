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
check(out['amount'] == 90 and out['unit'] == 'deg' and abs(out['planned_s'] - 2.0) < 0.01
      and all(c[0] < 0 < c[1] for c in moving(bus)), f"turn_left 720 -> 90deg 2.0s, L- R+ ({out['planned_s']})")
mv, bus, pt, clk = make()
out = mv.execute('turn_right', 45)
check(out['ok'] and all(c[0] > 0 > c[1] for c in moving(bus)) and abs(out['planned_s'] - 1.0) < 0.01, "turn_right 45 -> 1.0s, L+ R-")

# obstacle under 30 cm during forward -> 'obstacle', motors off
mv, bus, pt, clk = make()
rm.set_obstacle_provider(lambda: 100.0 if clk.t < 0.5 else 25.0)
out = mv.execute('forward', 40)
check(out['ok'] is False and out['stopped_reason'] == 'obstacle' and 0.4 < out['duration_ms'] / 1000 < 0.7 and bus.cmds[-1][:2] == (0, 0),
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
check(out['ok'] is False and out['error']['code'] == 'busy' and bus.cmds == [], f"SG intervention -> ok:false busy, no motor command ({out})")
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

# dispatcher wiring
import services.cloud.tool_dispatcher as td  # noqa: E402
d = td.ToolDispatcher()
check(d.has_tool('move') and d._tools['move'].timeout_s >= 5.0, "dispatcher registers 'move' with >=5s timeout")
rm._instance = rm.RelayMove(motor_bus=FakeBus(), pantilt=FakePT(), config=dict(CFG), sg_busy=lambda: False,
                            clock=FakeClock().now, sleep=lambda s: None)
clk2 = FakeClock(); rm._instance._clock = clk2.now; rm._instance._sleep = clk2.sleep
out = d._tool_move({'action': 'Forward', 'amount': 5}, threading.Event())
check(out['ok'] and out['stopped_reason'] == 'completed' and out['amount'] == 5, f"dispatcher move tool -> executes ({out.get('stopped_reason')})")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
