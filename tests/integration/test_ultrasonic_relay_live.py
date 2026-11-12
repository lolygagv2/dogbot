#!/usr/bin/env python3
"""Live: the relay `ultrasonic` command, end to end on the running robot.

Sends the same frames the relay does on ws://127.0.0.1:8000/ws/local and reads the
command_result. Without --fire it is silent (status + bad request only). With
--fire it plays ONE mild burst (25 kHz, 1 s, 40%) through the service's own
PCA9685 driver object, then checks that a second request is refused by the
cooldown and that the amp channel is off again.

Run: /home/morgan/dogbot/env_new/bin/python tests/integration/test_ultrasonic_relay_live.py [--fire]
Standalone (no pytest). Needs treatbot.service running.
"""
import asyncio
import json
import os
import sys
import time

import websockets

WS = os.environ.get('WIMZ_WS', 'ws://127.0.0.1:8000/ws/local')
FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


async def call(ws, n, **args):
    cid = f"ultra-{n}-{int(time.time() * 1000)}"
    await ws.send(json.dumps({"type": "command", "command": "ultrasonic", "command_id": cid,
                              "source": "test", "timestamp": time.time(), **args}))
    t_end = time.time() + 20
    while time.time() < t_end:
        try:
            f = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.time())))
        except (asyncio.TimeoutError, ValueError):
            continue
        if f.get("type") == "command_result" and f.get("command_id") == cid:
            return f
    return {}


def amp_on() -> bool:
    from smbus2 import SMBus
    with SMBus(1) as bus:
        return bool(bus.read_byte_data(0x40, 0x07 + 4 * 2) & 0x10)


async def main(fire: bool) -> int:
    async with websockets.connect(WS, max_size=8 * 1024 * 1024) as ws:
        f = await call(ws, 1, action='status')
        st = f.get('result') or {}
        print('status:', st)
        check(f.get('ok') is True and 'styles' in st and 'enabled' in st, 'status answered over the relay path')
        f = await call(ws, 2)
        code = ((f.get('error') or {}).get('code'))
        check(f.get('ok') is False and code in ('bad_request', 'ultrasonic_disabled'),
              f'empty request refused ({code})')
        if fire and st.get('enabled') and st.get('available'):
            f = await call(ws, 3, freq_hz=25000, level_pct=40, duration_s=1)
            print('burst:', f.get('result'), f.get('error'))
            check(f.get('ok') is True, 'burst played through the service (in-process PCA9685 driver)')
            check(amp_on() is False, 'amp channel is off after the burst')
            f = await call(ws, 4, style='before_quiet')
            err = f.get('error') or {}
            check(f.get('ok') is False and err.get('code') == 'cooldown',
                  f"second request refused by the cooldown ({f.get('result') or err})")
        elif fire:
            print('not firing: ultrasonic disabled or card not present')
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main('--fire' in sys.argv)))
