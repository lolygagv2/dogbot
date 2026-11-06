#!/usr/bin/env python3
"""BehaviorEventService derivations (CONTRACT v2 FINAL, 2026-11-06).

Drives the service with synthetic bus payloads and a fake clock and checks
the emitted rows: bark_episode (gap close, bark_count, peak_db, severity,
loudness gate), intervention (kind/result/episode_id), settle, pacing
(oscillation yes / straight walk no), treat source + command_id echo,
mission_run result names, status fields.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_behavior_events_derivation.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='bevtest_')

# Fake SG module so the loudness gate / playlist lookups don't touch hardware
sg_mod = types.ModuleType('modes.silent_guardian')
class _SG:
    config = {'bark_detection': {'loudness_threshold_db': -28}}
    calming_music_playing = False
    def _get_calming_playlist(self):
        return ['songs/default/bach.mp3'], '/tmp'
sg_mod.get_silent_guardian_mode = lambda: _SG()
sys.modules['modes.silent_guardian'] = sg_mod

from services.cloud.behavior_events import BehaviorEventService, bark_severity, treat_source  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeEmitter:
    def __init__(self):
        self.events = []

    def emit(self, etype, data, **kw):
        ev = {'type': etype, 'data': dict(data), 'kw': kw, 'id': kw.get('event_id') or f'fake-{len(self.events)}'}
        self.events.append(ev)
        return {'type': etype, 'dog_id': kw.get('dog_id'), 'id': ev['id'], 'data': data}

    def of(self, etype):
        return [e for e in self.events if e['type'] == etype]


def svc():
    em = FakeEmitter()
    s = BehaviorEventService(emitter=em, config={'status_period_s': 10_000})
    s.read_status = lambda: {'battery': 80, 'mode': 'idle', 'treats_remaining': 12, 'docked': False, 'light_level': 90}
    return s, em


def test_bark_episode():
    s, em = svc()
    t = 1000.0
    s.on_bark({'loudness_db': -40}, t)                     # below gate -> ignored
    check(s._ep_start is None, "sub-threshold bark does not open an episode")
    for i in range(4):
        s.on_bark({'loudness_db': -20 + i, 'dog_id': 'aruco_1', 'dog_name': 'Bezik'}, t + i * 5)
    s.tick(t + 15 + 10)
    check(not em.of('bark_episode'), "episode stays open while gap < 30 s")
    s.tick(t + 15 + 31)
    eps = em.of('bark_episode')
    check(len(eps) == 1, "episode closed after 30 s gap")
    d = eps[0]['data']
    check(d['bark_count'] == 4 and d['peak_db'] == -17.0 and d['severity'] == 'med',
          f"bark_count/peak_db/severity ({d})")
    check(eps[0]['kw'].get('dog_id') == 'aruco_1' and eps[0]['kw'].get('event_id'), "episode carries dog and a pre-generated id")
    check(bark_severity(2, 10) == 'low' and bark_severity(9, 10) == 'med' and bark_severity(10, 10) == 'high'
          and bark_severity(2, 61) == 'high', "severity thresholds")


def test_intervention_and_settle():
    s, em = svc()
    t = 2000.0
    s.on_bark({'loudness_db': -20}, t)
    ep_id = s.current_episode_id()
    s.on_sg_escalation({'escalation_level': 2, 'dog_id': 'aruco_1'}, t + 5)
    s.on_sg_intervention_record({'phase': 'triggered', 'escalation_level': 2}, t + 5)
    check(not em.of('intervention'), "no intervention row until the outcome")
    s.on_sg_intervention_record({'phase': 'outcome', 'escalation_level': 2, 'quiet_achieved': True,
                                 'treat_given': False, 'music_played': False}, t + 65)
    iv = em.of('intervention')
    check(len(iv) == 1 and iv[0]['data']['kind'] == 'voice' and iv[0]['data']['result'] == 'quiet'
          and iv[0]['data']['sg_level'] == 2, f"L2 quiet -> kind=voice result=quiet ({iv[0]['data'] if iv else None})")
    check(iv and iv[0]['data']['episode_id'] == ep_id, "intervention references the open bark episode id")
    # settle 300 s after outcome with no barks
    s.tick(t + 65 + 299)
    check(not em.of('settle'), "no settle before 300 s calm")
    s.tick(t + 65 + 301)
    st = em.of('settle')
    check(len(st) == 1 and st[0]['data']['calm_seconds'] == 300
          and st[0]['data']['seconds_since_last_intervention'] >= 300, "settle emitted once after 300 s calm")
    s.tick(t + 65 + 400)
    check(len(em.of('settle')) == 1, "settle not repeated")

    # kinds
    k = BehaviorEventService.intervention_kind
    check(k(1, False, False) == 'sound' and k(3, False, False) == 'approach' and k(4, False, False) == 'music'
          and k(2, True, False) == 'treat' and k(2, False, True) == 'music', "intervention kind mapping")
    # L4 no quiet
    s.on_sg_escalation({'escalation_level': 4}, t + 1000)
    s.on_sg_intervention_record({'phase': 'outcome', 'escalation_level': 4, 'quiet_achieved': False,
                                 'music_played': True}, t + 1300)
    iv = em.of('intervention')
    check(iv[-1]['data']['kind'] == 'music' and iv[-1]['data']['result'] == 'no_quiet', "L4 music no_quiet")
    # outcome never arrives -> unknown closes after timeout
    s.on_sg_escalation({'escalation_level': 1}, t + 2000)
    s.tick(t + 2000 + 901)
    check(em.of('intervention')[-1]['data']['result'] == 'no_quiet' and em.of('intervention')[-1]['data']['kind'] == 'sound',
          "intervention without outcome closes as no_quiet after 15 min")


def test_pacing():
    s, em = svc()
    t = 3000.0
    # oscillate across 60% of the frame every 2 s for 40 s
    for i in range(21):
        cx = 200 if i % 2 == 0 else 1000
        s.on_detection('dog_0', cx, t + i * 2, frame_w=1280)
    check(len(s._pace_open) == 1, "pacing opened on sustained oscillation")
    s.tick(t + 42 + 21)
    p = em.of('pacing')
    check(len(p) == 1 and p[0]['data']['reversals'] >= 6, f"pacing emitted after dog leaves ({p[0]['data'] if p else None})")
    # straight walk: no pacing
    s2, em2 = svc()
    for i in range(30):
        s2.on_detection('dog_0', 100 + i * 30, t + i, frame_w=1280)
    s2.tick(t + 30 + 30)
    check(not em2.of('pacing') and not s2._pace_open, "straight walk is not pacing")
    # kill switch
    s3, em3 = svc()
    s3.config['pacing_enabled'] = False
    ev = types.SimpleNamespace(subtype='dog_detected', data={'bbox': [0, 0, 100, 100], 'dog_id': 'd'}, timestamp=t)
    s3._on_vision(ev)
    check(not s3._pace_hist, "pacing_enabled=False ignores detections")


def test_treat_and_mission_and_status():
    s, em = svc()
    check(treat_source('coaching_reward') == 'auto' and treat_source('manual') == 'app'
          and treat_source('cloud_command') == 'app' and treat_source('connector_trick') == 'agent'
          and treat_source('connector') == 'connector' and treat_source('schedule') == 'schedule'
          and treat_source('mission') == 'schedule', "treat source mapping")
    ev = types.SimpleNamespace(subtype='treat_dispensed', timestamp=4000.0,
                               data={'reason': 'cloud_command', 'command_id': 'c-9', 'dog_id': 'aruco_1',
                                     'timestamp': 4000.0})
    s._on_reward(ev)
    tr = em.of('treat')
    check(len(tr) == 1 and tr[0]['data']['source'] == 'app' and tr[0]['data']['command_id'] == 'c-9',
          "treat row carries source and command_id echo")
    check(em.of('status'), "status emitted on treat (treats_remaining change)")

    s._on_system(types.SimpleNamespace(subtype='mission.started', timestamp=5000.0,
                                       data={'mission_id': 7, 'mission_name': 'sit_stay', 'dog_id': 'aruco_1'}))
    s._on_system(types.SimpleNamespace(subtype='mission.completed', timestamp=5100.0,
                                       data={'mission_id': 7, 'mission_name': 'sit_stay', 'success': True, 'rewards_given': 2}))
    mr = em.of('mission_run')
    check(len(mr) == 1 and mr[0]['data']['result'] == 'success' and mr[0]['data']['mission_id'] == 'sit_stay'
          and mr[0]['data']['start'] and mr[0]['data']['end'], f"mission_run success ({mr[0]['data'] if mr else None})")
    s._on_system(types.SimpleNamespace(subtype='mission.completed', timestamp=5200.0,
                                       data={'mission_id': 8, 'mission_name': 'x', 'success': False}))
    s._on_system(types.SimpleNamespace(subtype='mission.stopped', timestamp=5300.0,
                                       data={'mission_id': 9, 'mission_name': 'y', 'reason': 'user'}))
    check([m['data']['result'] for m in em.of('mission_run')] == ['success', 'fail', 'aborted'],
          "mission_run result vocabulary success|fail|aborted")

    st = em.of('status')[-1]['data']
    check(set(st) >= {'battery', 'mode', 'treats_remaining', 'docked', 'light_level'}, "status fields per contract")

    d = s.mark_departure('aruco_1')
    a = s.mark_arrival('aruco_1')
    check(em.of('departure')[0]['data']['method'] == 'app' and em.of('arrival')[0]['data']['method'] == 'app',
          "departure/arrival carry method=app")


def test_music():
    s, em = svc()
    s.on_audio_state({'state': 'playing', 'playing': True, 'track': 'songs/default/bach.mp3'}, 6000.0)
    s.on_audio_state({'state': 'stopped', 'playing': False}, 6120.0)
    m = em.of('music')
    check(len(m) == 1 and m[0]['data']['track_class'] == 'calming' and m[0]['data']['start'] < m[0]['data']['end'],
          "music row with track_class from the SG playlist")


if __name__ == '__main__':
    test_bark_episode()
    test_intervention_and_settle()
    test_pacing()
    test_treat_and_mission_and_status()
    test_music()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
