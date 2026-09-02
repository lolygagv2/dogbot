#!/usr/bin/env python3
"""One-time backfill of historical data into /data/wimz.db (STORE-A).

Sources -> spec targets (WIMZ_Data_Architecture_Spec.md §6 "Migrating today's
data"):
- recordings/coach_{dog}_{trick}_{ts}.mp4_{ts}.mp4 (59)  -> media_asset +
  training_attempt in per-day synthetic training sessions; files MOVED into
  the spec media tree (Morgan's decision 2026-07-05)
- data/dogbot.db behavior_events (380)   -> event('pose'), model dogpose_14
- data/missions.db detections (341)      -> event('pose'), per-mission sessions
- data/treatbot.db barks (207)           -> event('bark'), per-day monitor sessions

Idempotent by construction — safe to re-run:
- deterministic ids: uuid7_at(historical_ts, seed='source:table:key')
- INSERT OR IGNORE everywhere
- watermarks in schema_meta 'backfill:*'

Identity: only verified tag ids (aruco_NNN) attribute directly. Filename dog
names create id_method='manual' rows (human-labeled at recording time) except
obvious non-dogs. Bark rows keep their claimed name in the payload but stay
dog_id=NULL — the live 'dog_...' ids were never tag-verified.

Usage: env_new/bin/python scripts/backfill_wimz.py [--dry-run]
"""
import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/morgan/dogbot')

from core.data.ids import uuid7_at
from core.data.wimz_store import get_wimz_store, DATA_ROOT

REPO = Path('/home/morgan/dogbot')
RECORDINGS = REPO / 'recordings'
TRICKS = {'sit', 'laydown', 'down', 'come', 'spin', 'speak', 'stay', 'shake'}
NON_DOG_NAMES = {'dog', 'dog_0', 'dog_1', 'unknown', 'mytestdog', 't2', 'test'}

FILE_RE = re.compile(r'^coach_(?P<rest>.+)_(?P<ts>\d{8}_\d{6})\.mp4_(?P=ts)\.mp4$')


def ts_ms(s: str) -> int:
    """Legacy timestamp string (local) -> epoch ms."""
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y%m%d_%H%M%S'):
        try:
            return int(datetime.strptime(s, fmt).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {s}")


class Backfill:
    def __init__(self, dry_run: bool):
        self.dry = dry_run
        self.wimz = get_wimz_store()
        self.conn = self.wimz._conn
        self.lock = self.wimz._lock
        self.device_id = self.wimz.get_device_id()
        self.stats = {}

    # ---------------------------------------------------------------- utils

    def _watermark(self, key: str) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM schema_meta WHERE key=?", (f'backfill:{key}',)).fetchone()
        return int(row[0]) if row else 0

    def _set_watermark(self, key: str, value: int) -> None:
        if self.dry:
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (f'backfill:{key}', str(value)))
            self.conn.commit()

    def _resolve_dog_strict(self, key=None, name=None):
        """Amendment A identity ladder — SELECT-only, NEVER creates a row:
        canonical app UUID -> aruco tag -> case-insensitive name. Junk
        (dog_N, 'unknown', test ids) falls through to None, never a guess."""
        with self.lock:
            for k in (key, name):
                if not k or str(k).lower() in NON_DOG_NAMES:
                    continue
                row = self.conn.execute(
                    "SELECT dog_id FROM dog WHERE dog_id=? OR app_dog_id=? "
                    "OR qr_code_id=? OR lower(name)=lower(?) LIMIT 1",
                    (k, k, k, k)).fetchone()
                if row:
                    return row[0]
        return None

    def retro_tag_store_a(self):
        """One-time: stamp origin on the August STORE-A rows, written before
        the origin column existed. Fingerprints: backfilled event rows never
        carry seq (native log_event always does); synthetic sessions span
        exactly one day (ended_at - started_at = 86399999ms)."""
        if self._watermark('retro_tag_store_a'):
            self.stats['retro_tag'] = 'already done'
            return
        if self.dry:
            self.stats['retro_tag'] = 'dry run'
            return
        with self.lock:
            c = self.conn.execute(
                "UPDATE session SET origin='backfill:store_a' WHERE origin IS NULL "
                "AND initiated_by='autonomous' AND ended_at - started_at = 86399999")
            sessions = c.rowcount
            events = self.conn.execute(
                "UPDATE event SET origin='backfill:store_a' "
                "WHERE origin IS NULL AND seq IS NULL").rowcount
            attempts = self.conn.execute(
                "UPDATE training_attempt SET origin='backfill:store_a' "
                "WHERE origin IS NULL AND session_id IN "
                "(SELECT session_id FROM session WHERE origin='backfill:store_a')").rowcount
            self.conn.commit()
        self._set_watermark('retro_tag_store_a', 1)
        self.stats['retro_tag'] = f'{sessions} sessions, {events} events, {attempts} attempts'

    def _session(self, seed: str, mode: str, row_ts_ms: int) -> str:
        """Deterministic synthetic per-day session for legacy rows.

        The id derives from the DAY BOUNDARY, not the row timestamp — every
        row of the same (source, day) lands in one session regardless of
        processing order or re-runs.
        """
        day = datetime.fromtimestamp(row_ts_ms / 1000).strftime('%Y-%m-%d')
        day_start = int(datetime.strptime(day, '%Y-%m-%d').timestamp() * 1000)
        day_end = day_start + 86_399_999
        sid = uuid7_at(day_start, seed=f'backfill:session:{seed}:{day}'.encode())
        if not self.dry:
            with self.lock:
                self.conn.execute(
                    "INSERT OR IGNORE INTO session (session_id, device_id, mode, "
                    "initiated_by, started_at, ended_at, origin, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (sid, self.device_id, mode, 'autonomous', day_start, day_end,
                     f'backfill:{seed.split(":")[0]}', day_start, day_start))
                self.conn.commit()
        return sid

    def _insert_event(self, event_id: str, session_id: str, dog_id, ts: int,
                      event_type: str, payload: dict, confidence, model_name: str = None,
                      origin: str = 'backfill:store_a', label_source: str = 'machine'):
        if self.dry:
            return
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO event (event_id, session_id, device_id, dog_id, "
                "ts, seq, event_type, payload, confidence, model_id, label_source, "
                "synced, origin, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
                (event_id, session_id, self.device_id, dog_id, ts, None, event_type,
                 json.dumps(payload), confidence,
                 self.wimz.model_id_for(model_name) if model_name else None,
                 label_source, origin, ts))

    def _dog_for_name(self, name: str):
        """Filename/legacy names -> manual dog rows; obvious non-dogs -> None."""
        if not name or name.lower() in NON_DOG_NAMES:
            return None
        return self.wimz.get_or_create_dog(name=name, allow_unverified=True)

    # ------------------------------------------------------------ recordings

    def recordings(self):
        done = set()
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM schema_meta WHERE key='backfill:recordings'").fetchone()
        if row:
            done = set(json.loads(row[0]))

        import cv2
        moved = skipped = 0
        for f in sorted(RECORDINGS.glob('*.mp4')):
            if f.name in done:
                continue
            m = FILE_RE.match(f.name)
            if not m:
                print(f"  SKIP unparseable: {f.name}")
                skipped += 1
                continue
            rest, ts_str = m.group('rest'), m.group('ts')
            dog_name, trick = None, None
            parts = rest.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in TRICKS:
                dog_name, trick = parts
            elif rest in TRICKS:
                trick = rest
            else:
                print(f"  SKIP no trick token: {f.name}")
                skipped += 1
                continue

            cue_ms = ts_ms(ts_str)
            dog_id = self._dog_for_name(dog_name)
            day = datetime.fromtimestamp(cue_ms / 1000).strftime('%Y-%m-%d')
            session_id = self._session('recordings', 'training', cue_ms)

            # probe video
            width = height = duration_msec = None
            try:
                cap = cv2.VideoCapture(str(f))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    duration_msec = int(frames / fps * 1000) if fps else None
                cap.release()
            except Exception:
                pass

            media_id = uuid7_at(cue_ms, seed=f'backfill:media:{f.name}'.encode())
            attempt_id = uuid7_at(cue_ms, seed=f'backfill:attempt:{f.name}'.encode())
            dest_dir = (DATA_ROOT / 'media' / (dog_id or '_unassigned') / day / session_id)
            dest = dest_dir / f'{media_id}.mp4'
            rel_path = str(dest.relative_to(DATA_ROOT))

            if not self.dry:
                dest_dir.mkdir(parents=True, exist_ok=True)
                import hashlib
                h = hashlib.sha256()
                with open(f, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b''):
                        h.update(chunk)
                sha = h.hexdigest()
                size = f.stat().st_size
                shutil.move(str(f), dest)
                if not dest.exists() or dest.stat().st_size != size:
                    raise RuntimeError(f"move verification failed: {f.name}")
                with self.lock:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO media_asset (media_id, session_id, dog_id, "
                        "kind, rel_path, codec, width, height, duration_ms, size_bytes, "
                        "sha256, start_ts, retention_class, synced, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                        (media_id, session_id, dog_id, 'video', rel_path, 'mp4v',
                         width, height, duration_msec, size, sha, cue_ms,
                         'standard', cue_ms))
                    self.conn.execute(
                        "INSERT OR IGNORE INTO training_attempt (attempt_id, session_id, "
                        "dog_id, trick_label, cue_type, cue_ts, success, reward_dispensed, "
                        "label_source, media_id, synced, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,NULL,0,?,?,0,?,?)",
                        (attempt_id, session_id, dog_id, trick, 'voice', cue_ms,
                         'machine', media_id, cue_ms, cue_ms))
                    self.conn.commit()
                done.add(f.name)
                with self.lock:
                    self.conn.execute(
                        "INSERT INTO schema_meta (key, value) VALUES ('backfill:recordings', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (json.dumps(sorted(done)),))
                    self.conn.commit()
            moved += 1
        self.stats['recordings'] = f"{moved} moved+rows, {skipped} skipped"

    # -------------------------------------------------------- behavior_events

    def behavior_events(self):
        if not (REPO / 'data' / 'dogbot.db').exists():
            return  # source archived (Phase 4) — fleet tarball has it
        src = sqlite3.connect(REPO / 'data' / 'dogbot.db')
        wm = self._watermark('dogbot.behavior_events')
        rows = src.execute(
            "SELECT id, dog_id, behavior, confidence, timestamp FROM behavior_events "
            "WHERE id > ? ORDER BY id", (wm,)).fetchall()
        n = 0
        for rid, legacy_dog, behavior, conf, ts in rows:
            t = ts_ms(ts)
            dog_id = self.wimz.get_or_create_dog(legacy_id=legacy_dog)  # aruco_* verified
            day = datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d')
            sid = self._session('dogbot', 'training', t)
            eid = uuid7_at(t, seed=f'backfill:dogbot.behavior_events:{rid}'.encode())
            self._insert_event(eid, sid, dog_id, t, 'pose', {'pose': behavior},
                               conf, 'dogpose_14',
                               origin='backfill:dogbot.behavior_events')
            self._set_watermark('dogbot.behavior_events', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['behavior_events'] = n

    # ------------------------------------------------------------- detections

    def mission_detections(self):
        if not (REPO / 'data' / 'missions.db').exists():
            return  # source archived (Phase 4) — fleet tarball has it
        src = sqlite3.connect(REPO / 'data' / 'missions.db')
        wm = self._watermark('missions.detections')
        rows = src.execute(
            "SELECT id, mission_id, timestamp, pose, confidence, duration, bbox "
            "FROM detections WHERE id > ? ORDER BY id", (wm,)).fetchall()
        n = 0
        for rid, mission_id, ts, pose, conf, dur, bbox in rows:
            t = ts_ms(ts)
            sid = self._session(f'mission:{mission_id}', 'training', t)
            payload = {'pose': pose}
            if dur is not None:
                payload['duration_ms'] = int(float(dur) * 1000)
            if bbox:
                try:
                    payload['bbox'] = json.loads(bbox) if isinstance(bbox, str) else bbox
                except Exception:
                    pass
            eid = uuid7_at(t, seed=f'backfill:missions.detections:{rid}'.encode())
            self._insert_event(eid, sid, None, t, 'pose', payload, conf, 'dogpose_14',
                               origin='backfill:missions.detections')
            self._set_watermark('missions.detections', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['mission_detections'] = n

    # ------------------------------------------------------------------ barks

    def barks(self):
        src = sqlite3.connect(REPO / 'data' / 'treatbot.db')
        wm = self._watermark('treatbot.barks')
        rows = src.execute(
            "SELECT id, timestamp, dog_id, dog_name, emotion, confidence, loudness_db, "
            "duration_ms FROM barks WHERE id > ? ORDER BY id", (wm,)).fetchall()
        n = 0
        for rid, ts, legacy_dog, dog_name, emotion, conf, db, dur in rows:
            t = ts_ms(ts)
            # legacy bark dog ids ('dog_...') were never tag-verified: keep the
            # claim in the payload, leave dog_id NULL (identity integrity)
            dog_id = self.wimz.get_or_create_dog(legacy_id=legacy_dog)
            payload = {'db': db, 'duration_ms': dur, 'class': 'bark', 'emotion': emotion}
            if dog_id is None and dog_name:
                payload['claimed_dog'] = dog_name
            day = datetime.fromtimestamp(t / 1000).strftime('%Y-%m-%d')
            sid = self._session('barks', 'monitor', t)
            eid = uuid7_at(t, seed=f'backfill:treatbot.barks:{rid}'.encode())
            self._insert_event(eid, sid, dog_id, t, 'bark', payload, conf,
                               'dog_bark_classifier', origin='backfill:treatbot.barks')
            self._set_watermark('treatbot.barks', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['barks'] = n

    # ---------- Amendment A sources (2026-09-01, R5 reversed) ----------
    # STORAGE-ONLY by design: this script writes SQL rows and nothing else.
    # It imports no bus/relay modules — migrated history must never re-emit
    # as live events, flood the Activity feed, or fire push notifications.

    def rewards(self):
        """treatbot.rewards (unix-float REAL ts) -> event('treat_dispensed').
        Rows from the removed bark-reward lottery (behavior LIKE 'bark_%')
        get origin 'backfill:rewards:bark_reward' so owner-facing treat
        totals can exclude them."""
        src = sqlite3.connect(REPO / 'data' / 'treatbot.db')
        wm = self._watermark('treatbot.rewards')
        rows = src.execute(
            "SELECT id, timestamp, dog_id, behavior, confidence, success, "
            "treats_dispensed, mission_name FROM rewards WHERE id > ? ORDER BY id",
            (wm,)).fetchall()
        n = 0
        for rid, ts, legacy_dog, behavior, conf, success, treats, mission in rows:
            t = int(float(ts) * 1000)
            dog_id = self._resolve_dog_strict(legacy_dog)
            origin = ('backfill:rewards:bark_reward'
                      if (behavior or '').startswith('bark_') else 'backfill:rewards')
            sid = self._session('rewards', 'monitor', t)
            payload = {'behavior': behavior, 'success': bool(success),
                       'treats_dispensed': treats or 0}
            if mission and mission not in ('unknown', ''):
                payload['mission'] = mission
            eid = uuid7_at(t, seed=f'backfill:treatbot.rewards:{rid}'.encode())
            self._insert_event(eid, sid, dog_id, t, 'treat_dispensed', payload,
                               conf, origin=origin, label_source='auto_rule')
            self._set_watermark('treatbot.rewards', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['rewards'] = n

    def coaching(self):
        """treatbot.coaching_sessions (per-attempt rows, unix-float ts)
        -> training_attempt in per-day synthetic coach sessions."""
        src = sqlite3.connect(REPO / 'data' / 'treatbot.db')
        wm = self._watermark('treatbot.coaching_sessions')
        rows = src.execute(
            "SELECT id, timestamp, dog_id, dog_name, trick_requested, "
            "trick_completed, attention_duration, response_time, treat_dispensed "
            "FROM coaching_sessions WHERE id > ? ORDER BY id", (wm,)).fetchall()
        n = 0
        for (rid, ts, legacy_dog, dog_name, trick, completed,
             attention, response_time, treat) in rows:
            t = int(float(ts) * 1000)
            dog_id = self._resolve_dog_strict(legacy_dog, dog_name)
            sid = self._session('coach', 'coach', t)
            latency = int(float(response_time) * 1000) if response_time else None
            aid = uuid7_at(t, seed=f'backfill:treatbot.coaching_sessions:{rid}'.encode())
            if not self.dry:
                with self.lock:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO training_attempt (attempt_id, session_id, "
                        "dog_id, trick_label, cue_type, cue_ts, detected_response, "
                        "response_ts, latency_ms, success, reward_dispensed, "
                        "label_source, synced, origin, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)",
                        (aid, sid, dog_id, trick, 'voice', t,
                         trick if completed else None,
                         t + latency if (completed and latency) else None,
                         latency, 1 if completed else 0, 1 if treat else 0,
                         'machine', 'backfill:coaching_sessions', t, t))
            self._set_watermark('treatbot.coaching_sessions', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['coaching'] = n

    def sg_sessions(self):
        """treatbot.silent_guardian_sessions -> real session rows (mode='sg')
        with outcome_json rebuilt from the legacy columns. Bark totals are
        HOUSEHOLD-LEVEL (dog_bark_counts_json was always '{}') — bark_types
        stays empty, no per-dog invention (Amendment A flag 2)."""
        src = sqlite3.connect(REPO / 'data' / 'treatbot.db')
        wm = self._watermark('treatbot.silent_guardian_sessions')
        rows = src.execute(
            "SELECT id, session_start, session_end, total_barks, interventions, "
            "successful_quiets, treats_dispensed, max_escalation_level, "
            "COALESCE(panic_episodes, 0) FROM silent_guardian_sessions "
            "WHERE id > ? ORDER BY id", (wm,)).fetchall()
        n = 0
        for rid, start, end, barks, ivs, quiets, treats, max_esc, panics in rows:
            t0 = int(float(start) * 1000)
            t1 = int(float(end) * 1000) if end else None
            sid = uuid7_at(t0, seed=f'backfill:sg_session:{rid}'.encode())
            outcome = {'total_barks': barks or 0, 'bark_types': {},
                       'interventions_triggered': ivs or 0,
                       'successful_quiets': quiets or 0,
                       'treats_dispensed': treats or 0,
                       'max_escalation_level': max_esc or 0,
                       'panic_episodes': panics or 0}
            if not self.dry:
                with self.lock:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO session (session_id, device_id, mode, "
                        "initiated_by, started_at, ended_at, outcome_json, origin, "
                        "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (sid, self.device_id, 'sg', 'autonomous', t0, t1,
                         json.dumps(outcome), 'backfill:silent_guardian_sessions',
                         t0, t0))
            self._set_watermark('treatbot.silent_guardian_sessions', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['sg_sessions'] = n

    def sg_interventions(self):
        """treatbot.sg_interventions -> event('sg_intervention', phase
        'outcome') linked to the backfilled parent sg session."""
        src = sqlite3.connect(REPO / 'data' / 'treatbot.db')
        wm = self._watermark('treatbot.sg_interventions')
        rows = src.execute(
            "SELECT i.id, i.session_id, i.timestamp, i.escalation_level, i.dog_id, "
            "i.dog_name, i.barks_triggering, i.quiet_achieved, i.quiet_duration, "
            "i.treat_given, i.music_played, s.session_start "
            "FROM sg_interventions i "
            "LEFT JOIN silent_guardian_sessions s ON i.session_id = s.id "
            "WHERE i.id > ? ORDER BY i.id", (wm,)).fetchall()
        n = 0
        for (rid, legacy_sid, ts, level, legacy_dog, dog_name, barks_trig,
             quiet, quiet_dur, treat, music, parent_start) in rows:
            t = int(float(ts) * 1000)
            if legacy_sid and parent_start:
                sid = uuid7_at(int(float(parent_start) * 1000),
                               seed=f'backfill:sg_session:{legacy_sid}'.encode())
            else:
                sid = self._session('sg', 'sg', t)  # orphan rows
            dog_id = self._resolve_dog_strict(legacy_dog, dog_name)
            payload = {'phase': 'outcome', 'escalation_level': level,
                       'quiet_achieved': bool(quiet), 'treat_given': bool(treat),
                       'music_played': bool(music),
                       'barks_triggering': barks_trig or 0}
            if quiet_dur:
                payload['quiet_duration_sec'] = quiet_dur
            eid = uuid7_at(t, seed=f'backfill:treatbot.sg_interventions:{rid}'.encode())
            self._insert_event(eid, sid, dog_id, t, 'sg_intervention', payload,
                               None, origin='backfill:sg_interventions',
                               label_source='auto_rule')
            self._set_watermark('treatbot.sg_interventions', rid)
            n += 1
        if not self.dry:
            with self.lock:
                self.conn.commit()
        src.close()
        self.stats['sg_interventions'] = n

    def run(self):
        # Seed the two tag-verified dogs
        for tag, name in (('aruco_315', 'Elsa'), ('aruco_832', 'Bezik')):
            self.wimz.get_or_create_dog(legacy_id=tag, name=name)
        self.retro_tag_store_a()
        self.recordings()
        self.behavior_events()
        self.mission_detections()
        self.barks()
        # Amendment A sources (2026-09-01)
        self.rewards()
        self.coaching()
        self.sg_sessions()
        self.sg_interventions()
        print(f"\n{'DRY RUN — no writes' if self.dry else 'BACKFILL COMPLETE'}")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        with self.lock:
            for t in ('session', 'event', 'training_attempt', 'media_asset', 'dog'):
                print(f"  db {t}: {self.conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    Backfill(args.dry_run).run()
