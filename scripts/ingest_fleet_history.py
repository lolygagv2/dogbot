#!/usr/bin/env python3
"""Fleet history consolidation -> one analysis archive (wimz_fleet.db).

Builds a spec-v0.7-shaped SQLite archive from every robot's legacy files:
  incoming/<robot>/treatbot.db      barks, rewards, coaching_sessions,
                                    silent_guardian_sessions+sg_interventions,
                                    dog_events, missions, settings.dog_profiles
  incoming/<robot>/wimz.db          direct spec-shaped merge (dog ids remapped)
  incoming/<robot>/dogbot.db        behavior_events, rewards
  incoming/<robot>/treatbot.log*    parsed bark + treat lines (--logs)

Principles (Amendment A, extended fleet-wide):
- STORAGE-ONLY: writes SQL rows, imports no bus/relay modules, emits nothing.
- Idempotent: deterministic uuid7_at ids seeded per (robot, source, row),
  INSERT OR IGNORE everywhere. Re-running is always safe.
- Provenance: every row carries origin 'backfill:<robot>:<source>' (or
  'import:<robot>:wimz' for rows copied from a robot's own spec store, with
  any pre-existing origin preserved as 'import:<robot>:<orig>').
- Timestamps normalized to epoch-ms UTC at ingest: unix-float REAL as-is;
  SQLite CURRENT_TIMESTAMP text parsed as UTC; naive ISO / log lines parsed
  as robot-local time.
- Identity: one fleet-wide dog list. Canonical app UUID (from each robot's
  settings.dog_profiles) IS the dog_id where known; else a deterministic
  name-seeded id. dog_N / 'unknown' / test junk -> NULL, never guessed.
- device: one row per robot; every ingested row carries that device_id.
- Telemetry and the raw bus `events` dump are deliberately skipped
  (operational noise, not behavioral history).

Usage:
  env_new/bin/python scripts/ingest_fleet_history.py [--logs] [--archive PATH]
                     [--incoming DIR]
Scans incoming/<robot>/ folders; folder name = robot name.
"""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '/home/morgan/dogbot')
from core.data.ids import uuid7_at          # noqa: E402
from core.data.schema import SPEC_DDL       # noqa: E402

ROOT = Path('/home/morgan/wimz/fleet_archive')
NON_DOG = {'dog', 'dog_0', 'dog_1', 'dog_2', 'unknown', 'mytestdog', 't2',
           'test', 'test_dog_001', 'integration_test_dog', 'xbox_test',
           'default_dog', 'none', ''}

def ms_unix(v):            # REAL unix seconds -> epoch ms
    return int(float(v) * 1000)

def ms_utc_text(s):        # 'YYYY-MM-DD HH:MM:SS[.f]' stored as UTC
    for f in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return int(datetime.strptime(s, f).replace(tzinfo=timezone.utc)
                       .timestamp() * 1000)
        except ValueError:
            continue
    return ms_local_text(s)

def ms_local_text(s):      # naive local ISO / log prefix -> epoch ms
    s = s.replace('T', ' ').split('+')[0]
    for f in ('%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return int(datetime.strptime(s, f).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f'unparseable ts: {s!r}')


CLONE_TABLES = {'barks': 'timestamp', 'rewards': 'timestamp',
                'coaching_sessions': 'timestamp',
                'silent_guardian_sessions': 'session_start',
                'sg_interventions': 'timestamp', 'dog_events': 'timestamp'}


class Fleet:
    def __init__(self, archive: Path):
        fresh = not archive.exists()
        archive.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(archive)
        self.db.execute('PRAGMA journal_mode=WAL')
        if fresh:
            self.db.executescript(SPEC_DDL)
            self.db.commit()
        self.stats = {}
        # Fleet SD cards were cloned from one image, so every robot's
        # treatbot.db shares the canonical unit's historical prefix
        # (verified 2026-09-01: identical id+timestamp fingerprints on
        # tb1/tb2/tb3/tb5). The shared rows belong to the canonical robot;
        # other robots contribute only their post-divergence rows.
        self.canonical = None
        self.clone_ref = {}

    def load_clone_ref(self, robot: str, path: Path):
        self.canonical = robot
        src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        tabs = {r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for tab, col in CLONE_TABLES.items():
            if tab in tabs:
                self.clone_ref[tab] = {
                    (r[0], str(r[1])) for r in
                    src.execute(f"SELECT id, {col} FROM {tab}")}
        src.close()
        print(f'clone-dedupe ref: canonical={robot}, '
              + ', '.join(f'{t}={len(v)}' for t, v in self.clone_ref.items()))

    def _cloned(self, robot, tab, rid, ts) -> bool:
        if robot == self.canonical:
            return False
        hit = (rid, str(ts)) in self.clone_ref.get(tab, ())
        if hit:
            self.bump(f'{robot}:cloned_rows_skipped')
        return hit

    def bump(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n

    # ------------------------------------------------------------- identity
    def device(self, robot: str) -> str:
        did = uuid7_at(0, seed=f'fleet:device:{robot}'.encode())
        self.db.execute(
            "INSERT OR IGNORE INTO device (device_id, hardware_rev, created_at, "
            "updated_at) VALUES (?,?,0,0)", (did, robot))
        return did

    def dog(self, name=None, app_uuid=None, aruco=None, create=False):
        """Fleet dog resolution. app UUID IS the dog_id when known."""
        name_l = (name or '').strip().lower()
        if name_l in NON_DOG:
            name = None
        if not (app_uuid or name):
            return None
        c = self.db
        if app_uuid:
            row = c.execute("SELECT dog_id FROM dog WHERE dog_id=? OR app_dog_id=?",
                            (app_uuid, app_uuid)).fetchone()
            if row:
                return row[0]
        if name:
            row = c.execute("SELECT dog_id FROM dog WHERE lower(name)=lower(?)",
                            (name,)).fetchone()
            if row:
                if app_uuid:   # learn the canonical id
                    c.execute("UPDATE dog SET app_dog_id=COALESCE(app_dog_id,?) "
                              "WHERE dog_id=?", (app_uuid, row[0]))
                return row[0]
        if not create:
            return None
        did = app_uuid or uuid7_at(0, seed=f'fleet:dog:{name_l}'.encode())
        c.execute("INSERT OR IGNORE INTO dog (dog_id, name, qr_code_id, id_method, "
                  "app_dog_id, created_at, updated_at) VALUES (?,?,?,?,?,0,0)",
                  (did, name, f'aruco_{aruco}' if aruco is not None else None,
                   'qr' if aruco is not None else 'manual', app_uuid))
        return did

    def resolve(self, key=None, name=None):
        """Row-level ladder, SELECT-only: uuid/tag/name -> dog_id or None."""
        for k in (key, name):
            if not k or str(k).strip().lower() in NON_DOG:
                continue
            k = str(k).strip()
            row = self.db.execute(
                "SELECT dog_id FROM dog WHERE dog_id=? OR app_dog_id=? OR "
                "qr_code_id=? OR lower(name)=lower(?)", (k, k, k, k)).fetchone()
            if row:
                return row[0]
        return None

    # ------------------------------------------------------------- plumbing
    def session(self, robot, device_id, source, day_ms, mode):
        day = datetime.fromtimestamp(day_ms / 1000).strftime('%Y-%m-%d')
        d0 = int(datetime.strptime(day, '%Y-%m-%d').timestamp() * 1000)
        sid = uuid7_at(d0, seed=f'fleet:{robot}:session:{source}:{day}'.encode())
        self.db.execute(
            "INSERT OR IGNORE INTO session (session_id, device_id, mode, "
            "initiated_by, started_at, ended_at, origin, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, device_id, mode, 'autonomous', d0, d0 + 86_399_999,
             f'backfill:{robot}:{source}', d0, d0))
        return sid

    def event(self, robot, device_id, seed, sid, dog_id, t, etype, payload,
              conf, origin):
        eid = uuid7_at(t, seed=seed.encode())
        self.db.execute(
            "INSERT OR IGNORE INTO event (event_id, session_id, device_id, dog_id, "
            "ts, seq, event_type, payload, confidence, label_source, synced, "
            "origin, created_at) VALUES (?,?,?,?,?,NULL,?,?,?,?,0,?,?)",
            (eid, sid, device_id, dog_id, t, etype, json.dumps(payload), conf,
             'machine', origin, t))

    # -------------------------------------------------------- treatbot.db
    def ingest_treatbot(self, robot, device_id, path: Path, skip=frozenset()):
        src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        tabs = {r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}

        # Dog profiles first (identity source of record per robot)
        if 'settings' in tabs:
            row = src.execute(
                "SELECT value FROM settings WHERE key='dog_profiles'").fetchone()
            if row and row[0]:
                for p in json.loads(row[0]):
                    self.dog(name=p.get('name'), app_uuid=p.get('dog_id'),
                             aruco=p.get('aruco_id'), create=True)
                    self.bump(f'{robot}:profiles')

        if 'barks' in tabs and 'barks' not in skip:
            for (rid, ts, d, dn, emo, conf, db_, dur, *_x) in src.execute(
                    "SELECT id, timestamp, dog_id, dog_name, emotion, confidence, "
                    "loudness_db, duration_ms, NULL FROM barks"):
                if self._cloned(robot, 'barks', rid, ts):
                    continue
                t = ms_utc_text(str(ts))
                pl = {'db': db_, 'duration_ms': dur, 'class': 'bark',
                      'emotion': emo}
                dog = self.resolve(d, dn)
                if dog is None and (dn or d):
                    pl['claimed_dog'] = dn or d
                sid = self.session(robot, device_id, 'barks', t, 'monitor')
                self.event(robot, device_id, f'fleet:{robot}:barks:{rid}', sid,
                           dog, t, 'bark', pl, conf, f'backfill:{robot}:barks')
                self.bump(f'{robot}:barks')

        if 'rewards' in tabs and 'rewards' not in skip:
            for (rid, ts, d, beh, conf, succ, treats, mission) in src.execute(
                    "SELECT id, timestamp, dog_id, behavior, confidence, success, "
                    "treats_dispensed, mission_name FROM rewards"):
                if self._cloned(robot, 'rewards', rid, ts):
                    continue
                t = ms_unix(ts)
                origin = (f'backfill:{robot}:rewards:bark_reward'
                          if (beh or '').startswith('bark_')
                          else f'backfill:{robot}:rewards')
                pl = {'behavior': beh, 'success': bool(succ),
                      'treats_dispensed': treats or 0}
                if mission and mission not in ('unknown', ''):
                    pl['mission'] = mission
                sid = self.session(robot, device_id, 'rewards', t, 'monitor')
                self.event(robot, device_id, f'fleet:{robot}:rewards:{rid}', sid,
                           self.resolve(d), t, 'treat_dispensed', pl, conf, origin)
                self.bump(f'{robot}:rewards')

        if 'coaching_sessions' in tabs and 'coaching_sessions' not in skip:
            for (rid, ts, d, dn, trick, done, att, rt, treat) in src.execute(
                    "SELECT id, timestamp, dog_id, dog_name, trick_requested, "
                    "trick_completed, attention_duration, response_time, "
                    "treat_dispensed FROM coaching_sessions"):
                if self._cloned(robot, 'coaching_sessions', rid, ts):
                    continue
                t = ms_unix(ts)
                sid = self.session(robot, device_id, 'coach', t, 'coach')
                lat = int(float(rt) * 1000) if rt else None
                aid = uuid7_at(t, seed=f'fleet:{robot}:coach:{rid}'.encode())
                self.db.execute(
                    "INSERT OR IGNORE INTO training_attempt (attempt_id, session_id, "
                    "dog_id, trick_label, cue_type, cue_ts, detected_response, "
                    "response_ts, latency_ms, success, reward_dispensed, "
                    "label_source, synced, origin, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)",
                    (aid, sid, self.resolve(d, dn), trick, 'voice', t,
                     trick if done else None, t + lat if (done and lat) else None,
                     lat, 1 if done else 0, 1 if treat else 0, 'machine',
                     f'backfill:{robot}:coaching', t, t))
                self.bump(f'{robot}:coaching')

        if 'silent_guardian_sessions' in tabs and 'silent_guardian_sessions' not in skip:
            cols = {r[1] for r in src.execute('PRAGMA table_info(silent_guardian_sessions)')}
            panic = 'COALESCE(panic_episodes,0)' if 'panic_episodes' in cols else '0'
            for (rid, t0, t1, barks, ivs, q, tr, esc, pe) in src.execute(
                    f"SELECT id, session_start, session_end, total_barks, "
                    f"interventions, successful_quiets, treats_dispensed, "
                    f"max_escalation_level, {panic} FROM silent_guardian_sessions"):
                if self._cloned(robot, 'silent_guardian_sessions', rid, t0):
                    continue
                s0 = ms_unix(t0)
                sid = uuid7_at(s0, seed=f'fleet:{robot}:sg_session:{rid}'.encode())
                outcome = {'total_barks': barks or 0, 'bark_types': {},
                           'interventions_triggered': ivs or 0,
                           'successful_quiets': q or 0, 'treats_dispensed': tr or 0,
                           'max_escalation_level': esc or 0, 'panic_episodes': pe or 0}
                self.db.execute(
                    "INSERT OR IGNORE INTO session (session_id, device_id, mode, "
                    "initiated_by, started_at, ended_at, outcome_json, origin, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (sid, device_id, 'sg', 'autonomous', s0,
                     ms_unix(t1) if t1 else None, json.dumps(outcome),
                     f'backfill:{robot}:sg_sessions', s0, s0))
                self.bump(f'{robot}:sg_sessions')

        if 'sg_interventions' in tabs and 'sg_interventions' not in skip:
            for (rid, lsid, ts, lvl, d, dn, bt, qa, qd, tg, mp, ps) in src.execute(
                    "SELECT i.id, i.session_id, i.timestamp, i.escalation_level, "
                    "i.dog_id, i.dog_name, i.barks_triggering, i.quiet_achieved, "
                    "i.quiet_duration, i.treat_given, i.music_played, "
                    "s.session_start FROM sg_interventions i LEFT JOIN "
                    "silent_guardian_sessions s ON i.session_id = s.id"):
                if self._cloned(robot, 'sg_interventions', rid, ts):
                    continue
                t = ms_unix(ts)
                sid = (uuid7_at(ms_unix(ps), seed=f'fleet:{robot}:sg_session:{lsid}'.encode())
                       if lsid and ps else
                       self.session(robot, device_id, 'sg', t, 'sg'))
                pl = {'phase': 'outcome', 'escalation_level': lvl,
                      'quiet_achieved': bool(qa), 'treat_given': bool(tg),
                      'music_played': bool(mp), 'barks_triggering': bt or 0}
                if qd:
                    pl['quiet_duration_sec'] = qd
                self.event(robot, device_id, f'fleet:{robot}:sg_iv:{rid}', sid,
                           self.resolve(d, dn), t, 'sg_intervention', pl, None,
                           f'backfill:{robot}:sg_interventions')
                self.bump(f'{robot}:sg_interventions')

        if 'dog_events' in tabs:
            for (rid, ts, et, d, dn, det, mode, _s) in src.execute(
                    "SELECT id, timestamp, event_type, dog_id, dog_name, details, "
                    "mode, session_id FROM dog_events"):
                if self._cloned(robot, 'dog_events', rid, ts):
                    continue
                try:
                    t = ms_local_text(str(ts))
                except ValueError:
                    continue
                pl = {'legacy_type': et, 'mode': mode}
                try:
                    pl.update(json.loads(det) if det else {})
                except Exception:
                    pl['details'] = det
                sid = self.session(robot, device_id, 'dog_events', t, 'monitor')
                self.event(robot, device_id, f'fleet:{robot}:dog_events:{rid}', sid,
                           self.resolve(d, dn), t, et or 'legacy_event', pl, None,
                           f'backfill:{robot}:dog_events')
                self.bump(f'{robot}:dog_events')
        src.close()

    # ---------------------------------------------------------- wimz.db
    def ingest_wimz(self, robot, device_id, path: Path):
        src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        src.row_factory = sqlite3.Row
        tabs = {r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'event' not in tabs:
            src.close()
            return
        # dog remap: source-local dog_id -> fleet dog_id
        remap = {}
        for r in src.execute("SELECT * FROM dog"):
            r = dict(r)
            fid = self.dog(name=r.get('name'), app_uuid=r.get('app_dog_id'),
                           aruco=None, create=True) if (r.get('name') or r.get('app_dog_id')) else None
            remap[r['dog_id']] = fid

        def cols(tab):
            return [r[1] for r in src.execute(f'PRAGMA table_info({tab})')]

        for tab, idcol in (('session', 'session_id'), ('event', 'event_id'),
                           ('training_attempt', 'attempt_id'),
                           ('dispense_log', 'dispense_id')):
            if tab not in tabs:
                continue
            cl = cols(tab)
            dst_cl = [r[1] for r in self.db.execute(f'PRAGMA table_info({tab})')]
            use = [c for c in cl if c in dst_cl]
            for r in src.execute(f"SELECT * FROM {tab}"):
                row = {c: r[c] for c in use}
                if 'device_id' in dst_cl:
                    row['device_id'] = device_id
                if 'dog_id' in row and row['dog_id']:
                    row['dog_id'] = remap.get(row['dog_id'])
                prev = row.get('origin')
                row['origin'] = (f'import:{robot}:{prev}' if prev
                                 else f'import:{robot}:wimz')
                ph = ','.join('?' * len(row))
                self.db.execute(
                    f"INSERT OR IGNORE INTO {tab} ({','.join(row)}) VALUES ({ph})",
                    list(row.values()))
                self.bump(f'{robot}:wimz.{tab}')
        src.close()

    # ---------------------------------------------------------- dogbot.db
    def ingest_dogbot(self, robot, device_id, path: Path):
        src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        tabs = {r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'behavior_events' in tabs:
            for rid, d, beh, conf, ts in src.execute(
                    "SELECT id, dog_id, behavior, confidence, timestamp "
                    "FROM behavior_events"):
                t = ms_utc_text(str(ts))
                sid = self.session(robot, device_id, 'dogbot', t, 'training')
                self.event(robot, device_id, f'fleet:{robot}:dogbot:{rid}', sid,
                           self.resolve(d), t, 'pose', {'pose': beh}, conf,
                           f'backfill:{robot}:dogbot.behavior_events')
                self.bump(f'{robot}:dogbot.behavior_events')
        src.close()

    # -------------------------------------------------------------- logs
    # Line families discovered by template-mining the actual fleet logs
    # (2026-09-01). Seeds use the timestamp SECOND so (a) the two log lines
    # emitted per bark (bark_detector + silent_guardian) collapse to one
    # event — the richer detector line logs first and wins — and (b)
    # duplicate log file copies dedupe for free. BarkGate's 1s cooldown
    # guarantees max one real bark per second.
    LOG_TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.]+) - \w+ - (.*)')
    BARK_DET = re.compile(r'Bark detected: (?P<name>.+?) barked - (?P<dist>\w+) '
                          r'(?P<emo>\w+) \(conf: (?P<conf>[\d.]+), loudness: '
                          r'(?P<db>-?[\d.]+)dB\)')
    BARK_DET2 = re.compile(r'Bark detected: (?P<dist>\w+) (?P<emo>\w+) '
                           r'\(conf: (?P<conf>[\d.]+), loudness: (?P<db>-?[\d.]+)dB\)')
    BARK_SG = re.compile(r'Bark detected: (?P<name>[^(]+?) \(conf: (?P<conf>[\d.]+), '
                         r'loud: (?P<db>-?[\d.]+)dB\)')
    SG_START = re.compile(r'Starting Level (?P<lvl>\d) intervention for (?P<name>.+)$')
    SG_OK = re.compile(r'Level (?P<lvl>\d)[^-]*- SUCCESS')
    SG_FAIL = re.compile(r'Intervention timed out after (?P<sec>[\d.]+)s - giving up')
    COACH_S = re.compile(r'\[COACH\] Session started: dog=(?P<name>[^,]+), trick=(?P<trick>\w+)')
    COACH_C = re.compile(r'\[COACH\] Command issued: (?P<trick>\w+)')
    MODE = re.compile(r'MODE[_ ]CHANGE: (?P<frm>\w+) -> (?P<to>\w+)')
    DOGDET = re.compile(r'Dog detected: (?P<slot>dog_\d+)')
    TREAT = re.compile(r'Dispensed (?P<n>\d+) treat\(s\)')
    TREAT_DOG = re.compile(r'Dog: (?P<dog>[\w-]+)')
    TREAT_WHY = re.compile(r'Reason: (?P<reason>\w+)')
    PANIC = re.compile(r'(?i)enter(ing|ed)? panic|panic episode|PANIC state')

    def ingest_log(self, robot, device_id, path: Path):
        n = 0
        origin = f'backfill:{robot}:log'

        def emit(kind, sec_key, t, etype, payload, dog=None, conf=None):
            nonlocal n
            sid = self.session(robot, device_id, 'log', t, 'monitor')
            self.event(robot, device_id, f'fleet:{robot}:log:{kind}:{sec_key}',
                       sid, dog, t, etype, payload, conf, origin)
            self.bump(f'{robot}:log.{kind}')
            n += 1

        with open(path, errors='replace') as f:
            for line in f:
                m = self.LOG_TS.match(line)
                if not m:
                    continue
                ts_str, logger_name, msg = m.groups()
                try:
                    t = ms_local_text(ts_str)
                except ValueError:
                    continue
                sec = t // 1000
                pl = {'source': 'log'}

                if 'Bark detected:' in msg:
                    b = self.BARK_DET.search(msg) or self.BARK_DET2.search(msg)                         or self.BARK_SG.search(msg)
                    if not b:
                        continue
                    g = b.groupdict()
                    pl.update({'db': float(g['db']), 'class': 'bark'})
                    if g.get('emo'):
                        pl['emotion'] = g['emo']
                    dog = self.resolve(None, g.get('name'))
                    if dog is None and g.get('name'):
                        pl['claimed_dog'] = g['name'].strip()
                    emit('bark', sec, t, 'bark', pl, dog, float(g['conf']))

                elif 'Starting Level' in msg and (b := self.SG_START.search(msg)):
                    g = b.groupdict()
                    pl.update({'phase': 'triggered',
                               'escalation_level': int(g['lvl'])})
                    emit('sg_start', sec, t, 'sg_intervention', pl,
                         self.resolve(None, g['name']))

                elif (b := self.SG_OK.search(msg)):
                    pl.update({'phase': 'outcome', 'quiet_achieved': True,
                               'escalation_level': int(b.group('lvl'))})
                    emit('sg_ok', sec, t, 'sg_intervention', pl)

                elif (b := self.SG_FAIL.search(msg)):
                    pl.update({'phase': 'outcome', 'quiet_achieved': False,
                               'timeout_sec': float(b.group('sec'))})
                    emit('sg_fail', sec, t, 'sg_intervention', pl)

                elif '[COACH]' in msg:
                    if (b := self.COACH_S.search(msg)):
                        pl.update({'trick': b.group('trick'),
                                   'stage': 'session_start', 'cue_type': 'voice'})
                        emit('coach', sec, t, 'cue_issued', pl,
                             self.resolve(None, b.group('name')))
                    elif (b := self.COACH_C.search(msg)):
                        pl.update({'trick': b.group('trick'),
                                   'stage': 'command', 'cue_type': 'voice'})
                        emit('coach', sec, t, 'cue_issued', pl)

                elif (b := self.MODE.search(msg)):
                    pl.update({'from': b.group('frm'), 'to': b.group('to')})
                    emit('mode', sec, t, 'mode_change', pl)

                elif 'Dog detected:' in msg and (b := self.DOGDET.search(msg)):
                    pl.update({'class': 'dog', 'slot': b.group('slot')})
                    emit('detect', sec, t, 'detection', pl)

                elif 'Dispensed' in msg and 'treat(s)' in msg:
                    tm = self.TREAT.search(msg)
                    if not tm:
                        continue
                    pl['treats_dispensed'] = int(tm.group('n'))
                    dm = self.TREAT_DOG.search(msg)
                    wm = self.TREAT_WHY.search(msg)
                    if wm:
                        pl['behavior'] = wm.group('reason')
                    emit('treat', sec, t, 'treat_dispensed', pl,
                         self.resolve(dm.group('dog') if dm else None))

                elif self.PANIC.search(msg):
                    pl['detail'] = msg[:160]
                    emit('panic', sec, t, 'panic_episode', pl)
        return n

    # --------------------------------------------------------------- run
    def ingest_robot(self, folder: Path, with_logs: bool):
        robot = folder.name
        device_id = self.device(robot)
        dbs = sorted(folder.glob('*.db'))
        # treatbot.db-likes FIRST (they seed dog profiles), then wimz, then dogbot
        def kind(p):
            try:
                c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
                tabs = {r[0] for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                c.close()
            except Exception:
                return 'skip'
            if 'silent_guardian_sessions' in tabs or 'rewards' in tabs and 'settings' in tabs:
                return 'treatbot'
            if 'schema_meta' in tabs and 'event' in tabs:
                return 'wimz'
            if 'behavior_events' in tabs:
                return 'dogbot'
            return 'skip'
        classified = [(kind(p), p) for p in dbs]
        # A robot's own wimz.db may already hold its treatbot backfill
        # (Amendment A ran on-device) — skip those sources to avoid
        # double-counting in the archive.
        skip = set()
        for k, p in classified:
            if k == 'wimz':
                try:
                    c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
                    for (key,) in c.execute(
                            "SELECT key FROM schema_meta WHERE key LIKE 'backfill:treatbot.%'"):
                        skip.add(key.split('backfill:treatbot.', 1)[1])
                    c.close()
                except Exception:
                    pass
        if skip:
            print(f'  {robot}: skipping treatbot tables already in its wimz backfill: {sorted(skip)}')
        for k, p in classified:
            if k == 'treatbot':
                print(f'  {robot}: treatbot <- {p.name}')
                self.ingest_treatbot(robot, device_id, p, skip=frozenset(skip))
        for want, fn in (('wimz', self.ingest_wimz), ('dogbot', self.ingest_dogbot)):
            for k, p in classified:
                if k == want:
                    print(f'  {robot}: {want} <- {p.name}')
                    fn(robot, device_id, p)
        for k, p in classified:
            if k == 'skip':
                print(f'  {robot}: SKIPPED (unrecognized): {p.name}')
        if with_logs:
            for p in sorted(folder.glob('treatbot.log*')):
                print(f'  {robot}: log <- {p.name} ({self.ingest_log(robot, device_id, p)} rows)')
        self.db.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default=str(ROOT / 'wimz_fleet.db'))
    ap.add_argument('--incoming', default=str(ROOT / 'incoming'))
    ap.add_argument('--logs', action='store_true', help='also parse treatbot.log*')
    ap.add_argument('--canonical', default='tb5',
                    help='robot whose treatbot.db owns the cloned shared prefix')
    args = ap.parse_args()
    fleet = Fleet(Path(args.archive))
    canon = Path(args.incoming) / args.canonical / 'treatbot.db'
    if canon.exists():
        fleet.load_clone_ref(args.canonical, canon)
    for folder in sorted(Path(args.incoming).iterdir()):
        if folder.is_dir() and any(folder.iterdir()):
            print(f'== {folder.name} ==')
            fleet.ingest_robot(folder, args.logs)
    print('\n== stats ==')
    for k in sorted(fleet.stats):
        print(f'  {k}: {fleet.stats[k]}')
    for t in ('device', 'dog', 'session', 'event', 'training_attempt', 'dispense_log'):
        print(f'  archive {t}: {fleet.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]}')

if __name__ == '__main__':
    main()
