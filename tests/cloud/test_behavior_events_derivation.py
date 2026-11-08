#!/usr/bin/env python3
"""BehaviorEventService derivations (CONTRACT v2 FINAL, 2026-11-06).

Drives the service with synthetic bus payloads and a fake clock and checks
the emitted rows: bark_episode (gap close, bark_count, peak_db, severity,
loudness gate), intervention (kind/result/episode_id), settle, pacing
(oscillation yes / straight walk no), treat source + command_id echo,
mission_run result names, status fields, mode_change segments,
trick_attempt rows + coach_session aggregation (2026-11-08).

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

from services.cloud.behavior_events import BehaviorEventService, bark_severity, treat_source, _iso  # noqa: E402

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


def svc(stills=False):
    em = FakeEmitter()
    s = BehaviorEventService(emitter=em, config={'status_period_s': 10_000, 'stills_enabled': stills,
                                                 'treat_still_delay_s': 0.05})
    s.read_status = lambda: {'battery': 80, 'mode': 'idle', 'treats_remaining': 12, 'docked': False, 'light_level': 90}
    s.capture_still_url = lambda: 'https://relay/media/still.jpg'
    return s, em


def wait_for(em, etype, n=1, timeout=2.0):
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end and len(em.of(etype)) < n:
        _t.sleep(0.02)
    return em.of(etype)


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
    tr = wait_for(em, 'treat')
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


def test_stills():
    s, em = svc(stills=True)
    ev = types.SimpleNamespace(subtype='treat_dispensed', timestamp=7000.0,
                               data={'reason': 'manual', 'timestamp': 7000.0})
    s._on_reward(ev)
    check(not em.of('treat'), "treat emit deferred until the still is captured")
    tr = wait_for(em, 'treat')
    check(tr and tr[0]['data'].get('image_url') == 'https://relay/media/still.jpg', "treat carries image_url")
    # high severity episode gets a still, low does not
    t = 8000.0
    for i in range(12):
        s.on_bark({'loudness_db': -15}, t + i)
    s.tick(t + 12 + 31)
    eps = wait_for(em, 'bark_episode')
    check(eps and eps[0]['data']['severity'] == 'high' and eps[0]['data'].get('image_url'),
          "high-severity bark_episode carries image_url")
    s.on_bark({'loudness_db': -15}, t + 500)
    s.tick(t + 500 + 31)
    eps = wait_for(em, 'bark_episode', 2)
    check(len(eps) == 2 and eps[1]['data']['severity'] == 'low' and 'image_url' not in eps[1]['data'],
          "low-severity bark_episode has no still")
    # capture failure still emits
    s2, em2 = svc(stills=True)
    s2.capture_still_url = lambda: None
    s2._on_reward(ev)
    tr = wait_for(em2, 'treat')
    check(tr and 'image_url' not in tr[0]['data'], "treat still emitted when capture/upload fails")


def test_music():
    s, em = svc()
    s.on_audio_state({'state': 'playing', 'playing': True, 'track': 'songs/default/bach.mp3'}, 6000.0)
    s.on_audio_state({'state': 'stopped', 'playing': False}, 6120.0)
    m = em.of('music')
    check(len(m) == 1 and m[0]['data']['track_class'] == 'calming' and m[0]['data']['start'] < m[0]['data']['end'],
          "music row with track_class from the SG playlist")


def test_mode_change_segments():
    """Relay 2026-11-07: one mode_change row per completed stay in a mode."""
    s, em = svc()
    s._mode_seg = {'mode': 'idle', 'start': 1000.0}
    s._on_mode_change({'previous_mode': 'idle', 'new_mode': 'coach', 'reason': 'app', 'timestamp': 1060.0})
    mc = em.of('mode_change')
    check(len(mc) == 1 and mc[0]['data'] == {'mode': 'idle', 'start': _iso(1000.0), 'end': _iso(1060.0),
                                            'duration_s': 60.0, 'next_mode': 'coach', 'reason': 'app'},
          f"mode_change row for the ended mode {{mode, start, end}} ({mc[0]['data'] if mc else None})")
    check(mc[0]['kw'].get('resolve_dog') is False, "mode_change is not attributed to a dog")
    s._on_mode_change({'previous_mode': 'coach', 'new_mode': 'silent_guardian', 'reason': 'dropdown', 'timestamp': 1100.0})
    s._on_mode_change({'previous_mode': 'silent_guardian', 'new_mode': 'idle', 'reason': 'user', 'timestamp': 1400.0})
    mc = em.of('mode_change')
    check([r['data']['mode'] for r in mc] == ['idle', 'coach', 'silent_guardian']
          and [r['data']['duration_s'] for r in mc] == [60.0, 40.0, 300.0]
          and mc[1]['data']['start'] == mc[0]['data']['end'] and mc[2]['data']['start'] == mc[1]['data']['end'],
          "segments are contiguous: each start == previous end, durations add up")
    s._close_mode_segment(1450.0, next_mode='shutdown', reason='service_stop')
    mc = em.of('mode_change')
    check(len(mc) == 4 and mc[3]['data']['mode'] == 'idle' and mc[3]['data']['next_mode'] == 'shutdown'
          and mc[3]['data']['duration_s'] == 50.0, "service stop closes the open segment with next_mode shutdown")
    check(s._close_mode_segment(1500.0, 'x', 'y') is None and len(em.of('mode_change')) == 4,
          "closing twice emits nothing")
    s._mode_seg = {'mode': 'idle', 'start': 2000.0}
    s._on_mode_change({'previous_mode': 'idle', 'new_mode': 'shutdown', 'reason': 'System shutdown', 'timestamp': 2010.0})
    s._close_mode_segment(2013.0, next_mode='shutdown', reason='service_stop')
    mc = em.of('mode_change')
    check(len(mc) == 5 and mc[4]['data']['mode'] == 'idle' and mc[4]['data']['next_mode'] == 'shutdown',
          "state -> shutdown then service stop: one idle row, no 'shutdown' segment row")


def test_trick_attempt_and_coach_session():
    """Relay 2026-11-08: trick_attempt per cue (coach / mission / run_trick) and
    coach_session {start, end, cues_given, successes, treats} per Coach activation."""
    import types as _types
    s, em = svc()

    def sysev(subtype, data, ts):
        return _types.SimpleNamespace(subtype=subtype, data=data, timestamp=ts, source='t')

    def rewev(data, ts):
        return _types.SimpleNamespace(subtype='treat_dispensed', data=data, timestamp=ts, source='t')

    # A run_trick-only COACH activation (engine held, one agent cue) -> no coach_session row
    s._on_system(sysev('coaching_started', {}, 2000.0))
    s._on_system(sysev('trick_attempt', {'source': 'run_trick', 'cue': 'sit', 'result': 'success',
                                         'confidence': 0.91, 'dog_id': 'aruco_1', 'attempt': 1,
                                         'behavior': 'sit', 'time': 2010.0}, 2010.0))
    s._on_system(sysev('coaching_stopped', {}, 2020.0))
    ta = em.of('trick_attempt')
    check(len(ta) == 1 and ta[0]['data']['source'] == 'run_trick' and ta[0]['data']['cue'] == 'sit'
          and ta[0]['data']['result'] == 'success' and ta[0]['data']['confidence'] == 0.91
          and ta[0]['data']['attempt'] == 1 and ta[0]['kw'].get('dog_id') == 'aruco_1',
          f"run_trick cue -> trick_attempt {{cue, result, confidence, source, attempt}} ({ta[0]['data'] if ta else None})")
    check(ta[0]['data']['time'] == _iso(2010.0), "trick_attempt.time is the producer's cue-resolution time")
    check(len(em.of('coach_session')) == 0, "COACH entered only for run_trick (no coach cues) -> no coach_session row")

    # A real Coach activation: 3 cues, 2 successes, 2 coaching treats; an app treat inside is not counted
    s._on_system(sysev('coaching_started', {}, 3000.0))
    s._on_system(sysev('trick_attempt', {'source': 'coach', 'cue': 'sit', 'result': 'timeout',
                                         'confidence': None, 'dog_id': 'aruco_1', 'attempt': 1,
                                         'time': 3030.0}, 3030.0))
    s._on_system(sysev('trick_attempt', {'source': 'coach', 'cue': 'sit', 'result': 'success',
                                         'confidence': 0.8, 'dog_id': 'aruco_1', 'attempt': 2,
                                         'behavior': 'sit', 'time': 3050.0}, 3050.0))
    s._on_reward(rewev({'reason': 'coaching_reward', 'dog_id': 'aruco_1', 'timestamp': 3052.0}, 3052.0))
    s._on_reward(rewev({'reason': 'manual', 'dog_id': 'aruco_1', 'timestamp': 3060.0}, 3060.0))
    s._on_system(sysev('trick_attempt', {'source': 'coach', 'cue': 'speak', 'result': 'success',
                                         'confidence': 1.0, 'dog_id': 'aruco_1', 'attempt': 1,
                                         'behavior': 'bark', 'time': 3400.0}, 3400.0))
    s._on_reward(rewev({'reason': 'coaching_reward', 'dog_id': 'aruco_1', 'timestamp': 3402.0}, 3402.0))
    # a mission cue arriving mid-activation counts as its own trick_attempt, not a coach cue
    s._on_system(sysev('trick_attempt', {'source': 'mission', 'cue': 'laydown', 'result': 'no_dog',
                                         'dog_id': 'aruco_1', 'attempt': 1, 'time': 3500.0}, 3500.0))
    check(len(em.of('coach_session')) == 0, "coach_session is not sent before the activation ends")
    s._on_system(sysev('coaching_stopped', {}, 3600.0))
    cs = em.of('coach_session')
    check(len(cs) == 1 and cs[0]['data']['start'] == _iso(3000.0) and cs[0]['data']['end'] == _iso(3600.0)
          and cs[0]['data']['cues_given'] == 3 and cs[0]['data']['successes'] == 2
          and cs[0]['data']['treats'] == 2 and cs[0]['data']['duration_s'] == 600.0,
          f"coach_session {{start, end, cues_given 3, successes 2, treats 2}} ({cs[0]['data'] if cs else None})")
    check(cs[0]['kw'].get('resolve_dog') is False, "coach_session is per activation, not attributed to one dog")
    check(len(em.of('trick_attempt')) == 5 and em.of('trick_attempt')[-1]['data']['result'] == 'no_dog'
          and em.of('trick_attempt')[-1]['data']['source'] == 'mission',
          "mission cue -> trick_attempt source=mission result=no_dog")

    # Empty activation (dog never came): still a row with zero counts
    s._on_system(sysev('coaching_started', {}, 4000.0))
    s._on_system(sysev('coaching_stopped', {}, 4100.0))
    cs = em.of('coach_session')
    check(len(cs) == 2 and cs[1]['data']['cues_given'] == 0 and cs[1]['data']['treats'] == 0,
          "Coach activation with no cues at all -> coach_session row with zero counts")

    # Unknown result string is clamped to the contract set
    s._on_system(sysev('trick_attempt', {'source': 'coach', 'cue': 'sit', 'result': 'weird', 'time': 4200.0}, 4200.0))
    check(em.of('trick_attempt')[-1]['data']['result'] == 'timeout', "off-contract result clamps to 'timeout'")

    # Open activation closed on service stop
    s._on_system(sysev('coaching_started', {}, 5000.0))
    s._running = True
    s.stop()
    check(len(em.of('coach_session')) == 3, "service stop closes the open coach activation")


if __name__ == '__main__':
    test_bark_episode()
    test_intervention_and_settle()
    test_pacing()
    test_treat_and_mission_and_status()
    test_stills()
    test_music()
    test_mode_change_segments()
    test_trick_attempt_and_coach_session()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
