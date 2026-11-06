#!/usr/bin/env python3
"""TreatPolicy — the one treat-rate mechanism (CONTRACT v2, 2026-11-06).

treat_schedule {daily_cap, default_multiplier, max_per_hour, windows}: the
window multiplier scales the protocol cooldown (120 s / m), m == 0 blocks
protocol/mission/connector treats, owner taps always pass except at the
daily cap, max_per_hour is a rolling-hour cap on non-owner treats, windows
may wrap midnight, midnight resets.

Run: /home/morgan/dogbot/env_new/bin/python tests/reward/test_treat_policy_windows.py
Standalone (no pytest).
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.reward.treat_policy import TreatPolicy, classify_reason, validate_schedule  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


def ts(h, m=0):
    return datetime(2026, 11, 6, h, m).timestamp()


def main():
    tp = TreatPolicy()
    tp._get_dispenser = lambda: None   # no hardware

    check(classify_reason('coaching_reward') == 'protocol', "coaching_reward -> protocol")
    check(classify_reason('manual') == 'owner' and classify_reason('cloud_command') == 'owner', "app taps -> owner")
    check(classify_reason('mission') == 'mission', "mission -> mission")
    check(classify_reason('connector_trick') == 'connector' and classify_reason('agent') == 'connector',
          "agent/connector -> connector")
    check(classify_reason(None) == 'owner', "unknown reason defaults to owner")

    try:
        validate_schedule([{'start': '25:00', 'end': '10:00'}])
        check(False, "bad HH:MM rejected")
    except ValueError:
        check(True, "bad HH:MM rejected")

    tp.set_schedule([{'start': '22:00', 'end': '06:00', 'multiplier': 0},     # night, wraps midnight
                     {'start': '08:00', 'end': '10:00', 'multiplier': 2.0}])
    check(tp.current_multiplier(datetime(2026, 11, 6, 23, 30)) == 0, "23:30 inside wrapped night window (m=0)")
    check(tp.current_multiplier(datetime(2026, 11, 6, 2, 0)) == 0, "02:00 inside wrapped night window (m=0)")
    check(tp.current_multiplier(datetime(2026, 11, 6, 9, 0)) == 2.0, "09:00 morning window m=2")
    check(tp.current_multiplier(datetime(2026, 11, 6, 14, 0)) == 1.0, "14:00 outside windows -> default 1.0")
    tp.set_default_multiplier(0.5)
    check(tp.current_multiplier(datetime(2026, 11, 6, 14, 0)) == 0.5, "default_multiplier applies outside windows")
    check(tp.protocol_cooldown_s(now=datetime(2026, 11, 6, 9, 0)) == 60.0, "cooldown 120/2 = 60 s at m=2")
    check(tp.protocol_cooldown_s(now=datetime(2026, 11, 6, 23, 0)) == float('inf'), "m=0 -> infinite cooldown")
    tp.set_default_multiplier(1.0)

    d = tp.check('coaching_reward', now=ts(23))
    check(not d.allowed and d.code == 'window_blocked', "protocol blocked in m=0 window")
    d = tp.check('manual', now=ts(23))
    check(d.allowed, "owner tap allowed in m=0 window")

    d = tp.check('coaching_reward', now=ts(9))
    check(d.allowed, "protocol allowed at 09:00")
    tp.record_dispense('coaching_reward', now=ts(9))
    d = tp.check('coaching_reward', now=ts(9) + 30)
    check(not d.allowed and d.code == 'cooldown', "second protocol treat 30 s later blocked (cooldown 60 s)")
    d = tp.check('coaching_reward', now=ts(9) + 61)
    check(d.allowed, "protocol allowed after cooldown")
    d = tp.check('manual', now=ts(9) + 5)
    check(d.allowed, "owner tap ignores protocol cooldown")

    tp.set_daily_cap(5)
    d = tp.check('manual', now=ts(9), treats_today=5)
    check(not d.allowed and d.code == 'daily_cap', "daily cap blocks owner too")
    d = tp.check('manual', now=ts(9), treats_today=4)
    check(d.allowed, "under cap allowed")
    try:
        tp.set_daily_cap(0)
        check(False, "daily_cap 0 rejected")
    except ValueError:
        check(True, "daily_cap 0 rejected")

    tp2 = TreatPolicy()
    tp2._get_dispenser = lambda: None
    tp2.set_max_per_hour(2)
    t0 = ts(14)
    for i in range(2):
        check(tp2.check('coaching_reward', now=t0 + i * 200).allowed, f"max_per_hour: treat {i+1} allowed")
        tp2.record_dispense('coaching_reward', now=t0 + i * 200)
    d = tp2.check('coaching_reward', now=t0 + 500)
    check(not d.allowed and d.code == 'max_per_hour', "third protocol treat within the hour blocked")
    check(tp2.check('manual', now=t0 + 500).allowed, "owner tap not counted against max_per_hour")
    check(tp2.check('coaching_reward', now=t0 + 3700).allowed, "window rolls after an hour")

    secs = TreatPolicy.seconds_until_midnight(datetime(2026, 11, 6, 23, 59, 0))
    check(59 <= secs <= 61, f"seconds_until_midnight at 23:59 = {secs:.0f}")
    tp2.reset_daily()
    check(tp2.check('coaching_reward', now=t0 + 500).allowed, "reset clears hourly counter")

    s = tp.get_settings()
    check(s['daily_cap'] == 5 and len(s['windows']) == 2 and 'current_multiplier' in s, "get_settings shape")


if __name__ == '__main__':
    main()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
