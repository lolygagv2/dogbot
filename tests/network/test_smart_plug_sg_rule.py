#!/usr/bin/env python3
"""Smart plug + Silent Guardian rule (services/network/smart_plug.py).

A fake Kasa plug (real TCP server on 127.0.0.1 speaking the XOR protocol) and a
fake clock drive the service. No LAN traffic: discovery is stubbed and the fake
uses a made-up deviceId, so the real plug can never be switched by this test.

Checks: protocol round trip, level gate, on-at-level-2 + 2 h auto-off, plug that
was already on is left alone, off when SG ends (after the grace), manual on/off
override, timer restart on re-trigger, ownership survives a restart, off retried
while the plug is unreachable, wrong device at the address is refused, inert
without config.

Run: /home/morgan/dogbot/env_new/bin/python tests/network/test_smart_plug_sg_rule.py
Standalone (no pytest).
"""
import json
import socket
import socketserver
import struct
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.network import smart_plug as sp  # noqa: E402
from services.network.smart_plug import KasaPlug, SmartPlugService, _kasa_decrypt, _kasa_encrypt  # noqa: E402

FAILS = []
FAKE_ID = 'FAKE0000000000000000000000000000TESTPLUG'


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeKasa:
    """Minimal HS103: get_sysinfo + set_relay_state over the real wire format."""

    def __init__(self, device_id=FAKE_ID, on=False):
        self.device_id, self.on, self.down, self.sets = device_id, on, False, []
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                if outer.down:
                    return
                head = self.request.recv(4)
                if len(head) < 4:
                    return
                n = struct.unpack('>I', head)[0]
                body = b''
                while len(body) < n:
                    body += self.request.recv(n - len(body))
                req = json.loads(_kasa_decrypt(body))
                cmd = req.get('system', {})
                if 'set_relay_state' in cmd:
                    outer.on = bool(cmd['set_relay_state']['state'])
                    outer.sets.append(outer.on)
                    rep = {'system': {'set_relay_state': {'err_code': 0}}}
                else:
                    rep = {'system': {'get_sysinfo': {'deviceId': outer.device_id, 'alias': 'Test plug',
                                                      'relay_state': int(outer.on), 'err_code': 0}}}
                out = _kasa_encrypt(json.dumps(rep))
                self.request.sendall(struct.pack('>I', len(out)) + out)

        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def make(fake, clock, in_sg, state_file=None, device_id=FAKE_ID):
    plug = KasaPlug('127.0.0.1', device_id, None, fake.port, timeout_s=1.0)
    plug.discover = lambda: None          # never broadcast on the real LAN
    cfg = {'enabled': True, 'host': '127.0.0.1', 'device_id': device_id, 'alias': 'Test plug',
           'trigger_level': 2, 'auto_off_minutes': 120, 'exit_grace_s': 60, 'retry_s': 30}
    sf = state_file or Path(tempfile.mkdtemp(prefix='plugtest_')) / 'smart_plug.json'
    return SmartPlugService(config=cfg, plug=plug, state_file=sf, clock=clock,
                            mode_getter=lambda: in_sg['v']), sf


def main():
    # No bus side effects needed; _changed publishes best-effort.
    clock, in_sg = Clock(), {'v': True}

    # --- protocol
    fake = FakeKasa(on=False)
    plug = KasaPlug('127.0.0.1', FAKE_ID, None, fake.port, 1.0)
    plug.discover = lambda: None
    check(plug.is_on() is False, "protocol: reads relay_state off")
    plug.set(True)
    check(fake.on is True and plug.is_on() is True, "protocol: set on reaches the plug")
    check(_kasa_decrypt(_kasa_encrypt('{"a":1}')) == '{"a":1}', "protocol: xor round trip")
    fake.close()

    # --- level gate + on at level 2 + 2 h auto-off
    fake = FakeKasa(on=False)
    svc, _ = make(fake, clock, in_sg)
    check(svc.on_sg_trigger(1) is None and fake.on is False, "level 1 does not touch the plug")
    check(svc.on_sg_trigger(2) == 'turned_on' and fake.on is True, "level 2 with plug off -> robot turns it on")
    check(svc.owner == 'robot' and svc.status()['auto_off_in_s'] == 7200, "robot owns it, auto-off in 7200 s")
    clock.t += 7199
    check(svc.tick() is None and fake.on is True, "still on at 1 h 59 m 59 s")
    clock.t += 2
    check(svc.tick() == 'turned_off' and fake.on is False, "turned off 2 h after the trigger")
    check(svc.owner is None and svc.status()['owner'] == 'manual', "ownership released after auto-off")
    check(svc.tick() is None and fake.sets == [True, False], "no further switching once released")

    # --- re-trigger restarts the timer; level 4 jump also counts
    svc.on_sg_trigger(4)
    clock.t += 3600
    check(svc.on_sg_trigger(2) == 'extended', "re-trigger while robot-owned restarts the timer")
    clock.t += 3601
    check(svc.tick() is None and fake.on is True, "not off 2 h after the FIRST trigger")
    clock.t += 3600
    check(svc.tick() == 'turned_off', "off 2 h after the LAST trigger")
    fake.close()

    # --- already on: leave it on, forever
    fake = FakeKasa(on=True)
    svc, _ = make(fake, clock, in_sg)
    check(svc.on_sg_trigger(2) == 'left_on' and svc.owner is None, "plug already on -> left alone, not owned")
    clock.t += 10 * 3600
    in_sg['v'] = False
    svc.tick(); clock.t += 600; svc.tick()
    check(fake.on is True and fake.sets == [], "plug the robot did not turn on is never turned off")
    in_sg['v'] = True
    fake.close()

    # --- SG ends: off after the grace, not before
    fake = FakeKasa(on=False)
    svc, _ = make(fake, clock, in_sg)
    svc.on_sg_trigger(3)
    in_sg['v'] = False
    check(svc.tick() is None, "left SG: not off immediately")
    clock.t += 30
    in_sg['v'] = True
    svc.tick()
    in_sg['v'] = False
    clock.t += 45
    check(svc.tick() is None and fake.on is True, "brief mode blip (<60 s) does not turn it off")
    clock.t += 61
    check(svc.tick() == 'turned_off' and fake.on is False, "off 60 s after leaving Silent Guardian")
    in_sg['v'] = True
    fake.close()

    # --- manual ON override: robot never turns it off
    fake = FakeKasa(on=False)
    svc, _ = make(fake, clock, in_sg)
    svc.on_sg_trigger(2)
    res = svc.set_manual('on', source='relay')
    check(res['ok'] and res['owner'] == 'manual' and res['auto_off_in_s'] is None,
          "manual on while robot-owned -> ownership cleared")
    clock.t += 3 * 3600
    in_sg['v'] = False
    svc.tick(); clock.t += 120; svc.tick()
    check(fake.on is True, "manually-on plug survives the timer and SG exit")
    in_sg['v'] = True

    # --- manual OFF, then a new level-2 trigger turns it on again
    res = svc.set_manual('off')
    check(res['ok'] and fake.on is False and res['on'] is False, "manual off switches it off")
    check(svc.on_sg_trigger(2) == 'turned_on' and svc.owner == 'robot', "next level 2 turns it on again (robot-owned)")
    res = svc.set_manual('toggle')
    check(res['ok'] and fake.on is False and svc.owner is None, "toggle works and clears ownership")
    check(svc.set_manual('sideways')['error']['code'] == 'invalid_action', "bad action rejected")
    fake.close()

    # --- restart inside the window: persisted ownership still turns it off
    fake = FakeKasa(on=False)
    svc, sf = make(fake, clock, in_sg)
    svc.on_sg_trigger(2)
    svc2, _ = make(fake, clock, in_sg, state_file=sf)
    check(svc2.owner == 'robot' and svc2.auto_off_at == svc.auto_off_at, "ownership + deadline reloaded after restart")
    clock.t += 7201
    check(svc2.tick() == 'turned_off' and fake.on is False, "restarted service turns the plug off at the deadline")
    svc3, _ = make(fake, clock, in_sg, state_file=sf, device_id='OTHER' + FAKE_ID[5:])
    check(svc3.owner is None, "state of a different plug is not inherited")

    # --- unreachable at off time: keep ownership, retry
    svc, _ = make(fake, clock, in_sg)
    svc.on_sg_trigger(2)
    clock.t += 7201
    fake.down = True
    check(svc.tick() == 'error' and svc.owner == 'robot', "plug unreachable at auto-off -> ownership kept")
    fake.down = False
    check(svc.tick() is None and fake.on is True, "no hammering inside the retry interval")
    clock.t += 31
    check(svc.tick() == 'turned_off' and fake.on is False, "off on the retry once reachable")
    fake.down = True
    check(svc.set_manual('on')['error']['code'] == 'plug_unreachable', "manual command reports unreachable plug")
    check(svc.on_sg_trigger(2) == 'error' and svc.owner is None, "trigger with unreachable plug takes no ownership")
    fake.down = False
    fake.close()

    # --- wrong device answering at the address is never switched
    fake = FakeKasa(device_id='SOMEONE_ELSES_PLUG', on=False)
    svc, _ = make(fake, clock, in_sg)
    check(svc.on_sg_trigger(2) == 'error' and fake.on is False and fake.sets == [],
          "different device at the configured address is not switched")
    fake.close()

    # --- no config -> inert
    inert = SmartPlugService(config={}, state_file=Path(tempfile.mkdtemp()) / 's.json')
    check(inert.enabled is False and inert.start() is False, "no smart_plug section -> service inert")
    check(inert.on_sg_trigger(4) is None and inert.tick() is None, "inert service ignores triggers")
    check(inert.set_manual('on')['error']['code'] == 'no_plug', "inert service answers no_plug")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
