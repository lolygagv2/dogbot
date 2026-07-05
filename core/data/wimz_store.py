"""WimzStore — the spec data store (/data/wimz.db).

Implements WIMZ_Data_Architecture_Spec.md v0.3 exactly: verbatim DDL from
core/data/schema.py, WAL + batched writes for SD wear (spec §3/§7), UUIDv7
keys, label provenance on every machine event.

Runs alongside the legacy core/store.py (dual-write transition). Every public
write method is failure-isolated: a store problem logs an error but never
raises into robot control code.
"""
import hashlib
import json
import logging
import os
import socket
import sqlite3
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from core.data.ids import uuid7
from core.data.schema import PRAGMAS, SPEC_DDL, SCHEMA_VERSION

logger = logging.getLogger('WimzStore')


def resolve_data_root() -> Path:
    root = os.environ.get('WIMZ_DATA_ROOT')
    if root:
        return Path(root)
    if Path('/data').is_dir() and os.access('/data', os.W_OK):
        return Path('/data')
    return Path('/home/morgan/dogbot/data')


DATA_ROOT = resolve_data_root()
DB_PATH = DATA_ROOT / 'wimz.db'
MEDIA_DIR = DATA_ROOT / 'media'

# Models seeded into model_registry on first open (name, kind, version, artifact)
_SEED_MODELS = [
    ('dogdetector_14', 'detection', '14', '/home/morgan/dogbot/ai/models/dogdetector_14.hef'),
    ('dogpose_14', 'pose', '14', '/home/morgan/dogbot/ai/models/dogpose_14.hef'),
    ('dog_bark_classifier', 'audio', '1', '/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite'),
]

_FLUSH_INTERVAL_S = 3.0
_FLUSH_BATCH_ROWS = 50


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.warning(f"sha256 failed for {path}: {e}")
        return None


def _safe(fn):
    """Decorator: store failures log, never propagate into robot control."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"WimzStore.{fn.__name__} failed: {e}")
            return None
    return wrapper


class WimzStore:
    def __init__(self, data_root: Optional[Path] = None):
        self._root = Path(data_root) if data_root else DATA_ROOT
        self._db_path = self._root / 'wimz.db'
        self._media_dir = self._root / 'media'
        self._root.mkdir(parents=True, exist_ok=True)
        self._media_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._closed = False
        self._writable = True
        self._seq: Dict[str, int] = {}         # session_id -> last seq
        self._model_ids: Dict[str, str] = {}   # model name -> model_id
        self._ambient_session_id: Optional[str] = None

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        for pragma in PRAGMAS:
            self._conn.execute(pragma)

        self._bootstrap_schema()
        if self._writable:
            self._sweep_orphan_sessions()
            self._device_id = self._bootstrap_device()
            self._seed_models()
        else:
            self._device_id = None

        # Event batcher (spec §3/§7: batched transactions, not per-row)
        self._queue: deque = deque()
        self._wake = threading.Event()
        self._drained = threading.Event()
        self._drained.set()
        self._batcher = threading.Thread(
            target=self._batch_loop, daemon=True, name='WimzStoreBatcher')
        self._batcher.start()

        logger.info(f"WimzStore ready: {self._db_path} (schema {SCHEMA_VERSION}, "
                    f"device {self._device_id})")

    # ------------------------------------------------------------- bootstrap

    def _bootstrap_schema(self) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
            ).fetchone()
            if row is None:
                self._conn.executescript(SPEC_DDL)
                self._conn.commit()
                logger.info(f"wimz.db created (schema {SCHEMA_VERSION})")
                return
            version = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            version = version[0] if version else None
            if version != SCHEMA_VERSION:
                # Spec §10: refuse writes on mismatch — migration is explicit.
                self._writable = False
                logger.error(
                    f"wimz.db schema_version={version} != expected {SCHEMA_VERSION}; "
                    f"REFUSING WRITES until migrated")

    def _sweep_orphan_sessions(self) -> None:
        """Close sessions left open by a crash; best-effort end from last event."""
        with self._lock:
            now = _now_ms()
            cur = self._conn.execute(
                """UPDATE session SET
                     ended_at = COALESCE(
                       (SELECT MAX(ts) FROM event e WHERE e.session_id = session.session_id),
                       started_at),
                     updated_at = ?
                   WHERE ended_at IS NULL""", (now,))
            self._conn.commit()
            if cur.rowcount:
                logger.info(f"Closed {cur.rowcount} orphaned session(s) from previous run")

    def _bootstrap_device(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key='device_id'").fetchone()
            firmware = self._firmware_version()
            now = _now_ms()
            if row:
                device_id = row[0]
                self._conn.execute(
                    "UPDATE device SET firmware_version=?, updated_at=? WHERE device_id=?",
                    (firmware, now, device_id))
                self._conn.commit()
                return device_id
            device_id = uuid7()
            hardware_rev = f"rpi5-hailo8-{socket.gethostname()}"
            self._conn.execute(
                "INSERT INTO device (device_id, hardware_rev, firmware_version, created_at, updated_at) "
                "VALUES (?,?,?,?,?)", (device_id, hardware_rev, firmware, now, now))
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('device_id', ?)", (device_id,))
            self._conn.commit()
            logger.info(f"Device registered: {device_id} ({hardware_rev})")
            return device_id

    @staticmethod
    def _firmware_version() -> str:
        try:
            out = subprocess.run(
                ['git', 'describe', '--tags', '--always', '--dirty'],
                cwd='/home/morgan/dogbot', capture_output=True, text=True, timeout=5)
            return out.stdout.strip() or 'unknown'
        except Exception:
            return 'unknown'

    def _seed_models(self) -> None:
        for name, kind, version, artifact in _SEED_MODELS:
            try:
                self._model_ids[name] = self.register_model(name, kind, version, artifact)
            except Exception as e:
                logger.warning(f"Model seed failed for {name}: {e}")

    # ------------------------------------------------------------- identity

    def get_device_id(self) -> str:
        return self._device_id

    @_safe
    def get_or_create_dog(self, legacy_id: str = None, name: str = None,
                          id_method: str = 'qr') -> Optional[str]:
        """Stable dog_id lookup by tag id (qr_code_id) or case-insensitive name.

        Per spec v0.3: ArUco markers are represented via qr_code_id with
        id_method='qr' ("QR" covers ArUco throughout the fleet).
        """
        if not legacy_id and not name:
            return None
        with self._lock:
            if legacy_id:
                row = self._conn.execute(
                    "SELECT dog_id FROM dog WHERE qr_code_id=?", (legacy_id,)).fetchone()
                if row:
                    return row[0]
            if name:
                row = self._conn.execute(
                    "SELECT dog_id FROM dog WHERE lower(name)=lower(?)", (name,)).fetchone()
                if row:
                    # learn the tag mapping if we now have one
                    if legacy_id:
                        self._conn.execute(
                            "UPDATE dog SET qr_code_id=?, id_method=?, updated_at=? "
                            "WHERE dog_id=? AND qr_code_id IS NULL",
                            (legacy_id, id_method, _now_ms(), row[0]))
                        self._conn.commit()
                    return row[0]
            dog_id = uuid7()
            now = _now_ms()
            self._conn.execute(
                "INSERT INTO dog (dog_id, name, qr_code_id, id_method, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (dog_id, name, legacy_id, id_method if legacy_id else 'manual', now, now))
            self._conn.commit()
            logger.info(f"Dog registered: {name or legacy_id} -> {dog_id}")
            return dog_id

    def register_model(self, name: str, kind: str, version: str,
                       artifact_path: str = None) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT model_id FROM model_registry WHERE name=? AND version=?",
                (name, version)).fetchone()
            if row:
                self._model_ids[name] = row[0]
                return row[0]
            artifact_hash = None
            if artifact_path and Path(artifact_path).exists():
                artifact_hash = _sha256_file(Path(artifact_path))
            model_id = uuid7()
            self._conn.execute(
                "INSERT INTO model_registry (model_id, name, kind, version, artifact_hash, deployed_at) "
                "VALUES (?,?,?,?,?,?)",
                (model_id, name, kind, version, artifact_hash, _now_ms()))
            self._conn.commit()
            self._model_ids[name] = model_id
            return model_id

    def model_id_for(self, name: str) -> Optional[str]:
        return self._model_ids.get(name)

    # ------------------------------------------------------------- sessions

    @_safe
    def start_session(self, mode: str, initiated_by: str,
                      model_versions: Dict[str, Any] = None,
                      app_version: str = None) -> Optional[str]:
        session_id = uuid7()
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "INSERT INTO session (session_id, device_id, mode, initiated_by, app_version, "
                "model_versions, started_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, self._device_id, mode, initiated_by, app_version,
                 json.dumps(model_versions) if model_versions else None, now, now, now))
            self._conn.commit()
        logger.info(f"Session started: {mode}/{initiated_by} {session_id}")
        return session_id

    @_safe
    def end_session(self, session_id: str, ended_at_ms: int = None) -> None:
        if not session_id:
            return
        self.flush(timeout=2.0)  # events for this session land before it closes
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "UPDATE session SET ended_at=?, updated_at=? "
                "WHERE session_id=? AND ended_at IS NULL",
                (ended_at_ms or now, now, session_id))
            self._conn.commit()
        if self._ambient_session_id == session_id:
            self._ambient_session_id = None

    def ensure_ambient_session(self) -> Optional[str]:
        """Lazily-opened monitor session for writes outside an explicit mode."""
        with self._lock:
            if self._ambient_session_id is None:
                self._ambient_session_id = self.start_session('monitor', 'autonomous')
            return self._ambient_session_id

    # ------------------------------------------------------------- writes

    def _next_seq(self, session_id: str) -> int:
        with self._lock:
            if session_id not in self._seq:
                row = self._conn.execute(
                    "SELECT MAX(seq) FROM event WHERE session_id=?", (session_id,)).fetchone()
                self._seq[session_id] = row[0] or 0
            self._seq[session_id] += 1
            return self._seq[session_id]

    @_safe
    def log_event(self, session_id: str, event_type: str, payload: Dict[str, Any] = None,
                  dog_id: str = None, ts_ms: int = None, confidence: float = None,
                  model_id: str = None, label_source: str = 'machine',
                  media_id: str = None) -> Optional[str]:
        """Batched — row hits disk on next flush (<=3s / 50 rows)."""
        if not self._writable or self._closed:
            return None
        if not session_id:
            session_id = self.ensure_ambient_session()
        event_id = uuid7()
        now = _now_ms()
        row = (event_id, session_id, self._device_id, dog_id, ts_ms or now,
               self._next_seq(session_id), event_type,
               json.dumps(payload) if payload else None,
               confidence, model_id, label_source, media_id, 0, now)
        self._queue.append(row)
        self._drained.clear()
        if len(self._queue) >= _FLUSH_BATCH_ROWS:
            self._wake.set()
        return event_id

    @_safe
    def log_training_attempt(self, session_id: str, trick_label: str,
                             dog_id: str = None, cue_ts_ms: int = None,
                             cue_event_id: str = None, cue_type: str = None,
                             detected_response: str = None, response_ts_ms: int = None,
                             response_event_id: str = None, latency_ms: int = None,
                             success: int = None, confidence: float = None,
                             reward_dispensed: int = 0, dispense_id: str = None,
                             model_versions: Dict[str, Any] = None,
                             media_id: str = None) -> Optional[str]:
        """Write-through — the moat row is never queued."""
        if not self._writable or self._closed:
            return None
        self.flush(timeout=2.0)  # referenced event ids must exist first (FKs)
        attempt_id = uuid7()
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "INSERT INTO training_attempt (attempt_id, session_id, dog_id, trick_label, "
                "cue_event_id, cue_type, cue_ts, response_event_id, detected_response, "
                "response_ts, latency_ms, success, confidence, reward_dispensed, dispense_id, "
                "model_versions, label_source, media_id, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (attempt_id, session_id, dog_id, trick_label, cue_event_id, cue_type,
                 cue_ts_ms or now, response_event_id, detected_response, response_ts_ms,
                 latency_ms, success, confidence, reward_dispensed, dispense_id,
                 json.dumps(model_versions) if model_versions else None,
                 'machine', media_id, now, now))
            self._conn.commit()
        return attempt_id

    @_safe
    def log_dispense(self, session_id: str, trigger: str, dog_id: str = None,
                     slot: int = None, attempt_id: str = None,
                     dispensed_confirmed: int = 0, confirm_latency_ms: int = None,
                     ts_ms: int = None) -> Optional[str]:
        """Write-through; also enqueues the paired treat_dispensed event (spec §6)."""
        if not self._writable or self._closed:
            return None
        if not session_id:
            session_id = self.ensure_ambient_session()
        dispense_id = uuid7()
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "INSERT INTO dispense_log (dispense_id, session_id, dog_id, ts, slot, trigger, "
                "attempt_id, dispensed_confirmed, confirm_latency_ms, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (dispense_id, session_id, dog_id, ts_ms or now, slot, trigger,
                 attempt_id, dispensed_confirmed, confirm_latency_ms, now))
            self._conn.commit()
        self.log_event(session_id, 'treat_dispensed',
                       payload={'slot': slot} if slot is not None else {},
                       dog_id=dog_id, ts_ms=ts_ms or now, label_source='auto_rule')
        return dispense_id

    @_safe
    def register_media(self, path: str, kind: str, session_id: str = None,
                       dog_id: str = None, codec: str = None, width: int = None,
                       height: int = None, duration_ms: int = None,
                       start_ts_ms: int = None, end_ts_ms: int = None,
                       retention_class: str = 'standard',
                       compute_hash: bool = True) -> Optional[str]:
        """Write-through. rel_path stored relative to DATA_ROOT (spec §3)."""
        if not self._writable or self._closed:
            return None
        p = Path(path)
        try:
            rel_path = str(p.relative_to(self._root))
        except ValueError:
            # Outside the data root — interim deviation, flagged in spec 0.3 work
            rel_path = str(p)
            logger.warning(f"media outside data root, storing absolute path: {p}")
        with self._lock:
            row = self._conn.execute(
                "SELECT media_id FROM media_asset WHERE rel_path=?", (rel_path,)).fetchone()
            if row:
                return row[0]
        size_bytes = p.stat().st_size if p.exists() else None
        sha = _sha256_file(p) if (compute_hash and p.exists()) else None
        media_id = uuid7()
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                "INSERT INTO media_asset (media_id, session_id, dog_id, kind, rel_path, codec, "
                "width, height, duration_ms, size_bytes, sha256, start_ts, end_ts, "
                "retention_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (media_id, session_id, dog_id, kind, rel_path, codec, width, height,
                 duration_ms, size_bytes, sha, start_ts_ms, end_ts_ms, retention_class, now))
            self._conn.commit()
        return media_id

    def list_media(self, kind: str = None, retention_class: str = None,
                   limit: int = 100) -> list:
        """Media rows newest-first: (media_id, rel_path, size_bytes, created_at)."""
        q = "SELECT media_id, rel_path, size_bytes, created_at FROM media_asset"
        clauses, args = [], []
        if kind:
            clauses.append("kind=?"); args.append(kind)
        if retention_class:
            clauses.append("retention_class=?"); args.append(retention_class)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return self._conn.execute(q, args).fetchall()

    @_safe
    def cull_media(self, media_id: str) -> None:
        """Delete a media file + its row (retention culling — ephemeral only
        by policy; behavioral rows are never culled, spec §3)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT rel_path FROM media_asset WHERE media_id=?", (media_id,)).fetchone()
            if not row:
                return
            p = self._root / row[0]
            if p.exists():
                p.unlink()
            self._conn.execute("DELETE FROM media_asset WHERE media_id=?", (media_id,))
            self._conn.commit()

    def media_path_for(self, session_id: str = None, dog_id: str = None,
                       ext: str = 'jpg') -> Path:
        """Allocate a spec §3 media path: media/{dog|_unassigned}/{date}/{session}/{uuid}.{ext}"""
        dog_part = dog_id or '_unassigned'
        date_part = time.strftime('%Y-%m-%d')
        session_part = session_id or self.ensure_ambient_session() or 'no_session'
        d = self._media_dir / dog_part / date_part / session_part
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{uuid7()}.{ext}"

    # ------------------------------------------------------------- batcher

    def _batch_loop(self) -> None:
        while not self._closed:
            self._wake.wait(_FLUSH_INTERVAL_S)
            self._wake.clear()
            self._drain()

    def _drain(self) -> None:
        if not self._queue:
            self._drained.set()
            return
        rows = []
        while self._queue:
            rows.append(self._queue.popleft())
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO event (event_id, session_id, device_id, dog_id, ts, "
                    "seq, event_type, payload, confidence, model_id, label_source, media_id, "
                    "synced, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                self._conn.commit()
        except sqlite3.Error as e:
            # Retry rows one-by-one; dead-letter failures to the log, never wedge
            logger.error(f"Event batch failed ({e}); retrying rows individually")
            for row in rows:
                try:
                    with self._lock:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO event (event_id, session_id, device_id, "
                            "dog_id, ts, seq, event_type, payload, confidence, model_id, "
                            "label_source, media_id, synced, created_at) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                        self._conn.commit()
                except sqlite3.Error as e2:
                    logger.error(f"Dead-lettered event {row[0]} ({row[6]}): {e2}")
        if not self._queue:
            self._drained.set()

    def flush(self, timeout: float = 2.0) -> bool:
        """Drain the event queue; True if fully drained within timeout."""
        if not self._queue:
            return True
        self._wake.set()
        return self._drained.wait(timeout)

    def close(self) -> None:
        if self._closed:
            return
        # Drain synchronously — the batcher thread may already be gone
        self._drain()
        self._closed = True
        self._wake.set()
        try:
            with self._lock:
                self._conn.commit()
                self._conn.close()
        except sqlite3.Error:
            pass
        logger.info("WimzStore closed")


# ------------------------------------------------------------------ singleton

_instance: Optional[WimzStore] = None
_instance_lock = threading.Lock()


def get_wimz_store() -> WimzStore:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = WimzStore()
    return _instance
