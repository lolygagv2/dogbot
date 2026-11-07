"""Durable relay outbox — ordered, crash-safe delivery queue for relay events.

Spec v0.8 (relay protocol v2, 2026-11-06). Replaces the in-memory 500-entry
backlog in RelayClient for the new structured event types: a row is inserted
synchronously at emission time, the relay client's single pump coroutine
sends rows in `seq` order and marks `sent_at` on a successful socket write.
A restart while offline therefore loses nothing, and because the relay
dedupes on the event `id`, a resend after a crash between write and mark is
harmless (at-least-once).

Robot-internal: rows never sync. Sent rows are pruned after 24 h; pending
rows are capped so a robot that is offline for weeks cannot fill the disk
(oldest `status` rows go first — a stale status has no value).

Falls back to an in-memory list with the same API if wimz.db is unwritable,
so the robot keeps running with no relay and no database.
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.data.schema import PRAGMAS

logger = logging.getLogger('RelayOutbox')

PENDING_CAP = 20_000           # hard cap on unsent rows
PENDING_TRIM_TO = 15_000       # after a cap breach, trim down to this many
SENT_RETENTION_MS = 24 * 3600 * 1000
LOW_VALUE_TYPES = ('status',)  # pruned first when the pending cap is hit


def _now_ms() -> int:
    return int(time.time() * 1000)


class RelayOutbox:
    """sqlite-backed FIFO keyed by autoincrement seq."""

    def __init__(self, db_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._mem: List[Tuple[int, dict, bool]] = []   # fallback: (seq, event, sent)
        self._mem_seq = 0
        if db_path is None:
            from core.data.wimz_store import DB_PATH
            db_path = DB_PATH
        self._db_path = Path(db_path)
        try:
            # Make sure WimzStore has bootstrapped/migrated the schema first
            # (it owns the schema; we only use the relay_outbox table).
            from core.data.wimz_store import get_wimz_store
            get_wimz_store()
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            for pragma in PRAGMAS:
                self._conn.execute(pragma)
            self._conn.execute("SELECT 1 FROM relay_outbox LIMIT 1").fetchall()
            logger.info(f"[OUTBOX] ready at {self._db_path} "
                        f"(pending={self.pending_count()})")
        except Exception as e:
            logger.error(f"[OUTBOX] wimz.db unavailable ({e}); using in-memory outbox")
            self._conn = None

    # ------------------------------------------------------------- writes

    def enqueue(self, event: dict) -> Optional[int]:
        """Insert one wire event (must carry 'event_id' and 'type'). Returns seq.

        Legacy rows keyed 'id' are still accepted; the pump rewrites them to
        'event_id' on the wire (CONTRACT v2 FINAL).
        """
        event_id = event.get('event_id') or event.get('id')
        event_type = event.get('type') or event.get('event', 'unknown')
        if not event_id:
            raise ValueError("outbox event needs an 'event_id'")
        payload = json.dumps(event, separators=(',', ':'), default=str)
        with self._lock:
            if self._conn is None:
                self._mem_seq += 1
                self._mem.append((self._mem_seq, event, False))
                if len(self._mem) > PENDING_CAP:
                    self._mem = self._mem[-PENDING_TRIM_TO:]
                return self._mem_seq
            try:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO relay_outbox "
                    "(event_id, event_type, payload, created_at) VALUES (?,?,?,?)",
                    (event_id, event_type, payload, _now_ms()))
                self._conn.commit()
                return cur.lastrowid if cur.rowcount else None
            except sqlite3.Error as e:
                logger.error(f"[OUTBOX] enqueue failed ({e}); event {event_type} dropped")
                return None

    def mark_sent(self, seqs: List[int]) -> None:
        if not seqs:
            return
        with self._lock:
            if self._conn is None:
                sent = set(seqs)
                self._mem = [(s, e, True if s in sent else st) for s, e, st in self._mem]
                return
            try:
                now = _now_ms()
                self._conn.executemany(
                    "UPDATE relay_outbox SET sent_at=?, attempts=attempts+1 WHERE seq=?",
                    [(now, s) for s in seqs])
                self._conn.commit()
            except sqlite3.Error as e:
                logger.error(f"[OUTBOX] mark_sent failed: {e}")

    def mark_attempt(self, seqs: List[int]) -> None:
        """Count a failed send attempt (row stays pending)."""
        if not seqs or self._conn is None:
            return
        with self._lock:
            try:
                self._conn.executemany(
                    "UPDATE relay_outbox SET attempts=attempts+1 WHERE seq=?",
                    [(s,) for s in seqs])
                self._conn.commit()
            except sqlite3.Error as e:
                logger.debug(f"[OUTBOX] mark_attempt failed: {e}")

    # ------------------------------------------------------------- reads

    def pending(self, limit: int = 200) -> List[Tuple[int, dict]]:
        """Oldest unsent rows first, strictly by seq."""
        with self._lock:
            if self._conn is None:
                return [(s, e) for s, e, st in self._mem if not st][:limit]
            try:
                rows = self._conn.execute(
                    "SELECT seq, payload FROM relay_outbox WHERE sent_at IS NULL "
                    "ORDER BY seq LIMIT ?", (int(limit),)).fetchall()
            except sqlite3.Error as e:
                logger.error(f"[OUTBOX] pending query failed: {e}")
                return []
        out = []
        for seq, payload in rows:
            try:
                out.append((seq, json.loads(payload)))
            except ValueError:
                logger.error(f"[OUTBOX] corrupt payload at seq {seq}; marking sent to skip")
                self.mark_sent([seq])
        return out

    def pending_count(self) -> int:
        with self._lock:
            if self._conn is None:
                return sum(1 for _, _, st in self._mem if not st)
            try:
                return self._conn.execute(
                    "SELECT COUNT(*) FROM relay_outbox WHERE sent_at IS NULL").fetchone()[0]
            except sqlite3.Error:
                return 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            if self._conn is None:
                return {'backend': 'memory', 'pending': self.pending_count(),
                        'sent_retained': sum(1 for _, _, st in self._mem if st)}
            try:
                pending, sent = self._conn.execute(
                    "SELECT SUM(sent_at IS NULL), SUM(sent_at IS NOT NULL) FROM relay_outbox"
                ).fetchone()
                return {'backend': 'sqlite', 'pending': pending or 0, 'sent_retained': sent or 0}
            except sqlite3.Error:
                return {'backend': 'sqlite', 'pending': 0, 'sent_retained': 0}

    # ------------------------------------------------------------- housekeeping

    def prune(self) -> None:
        """Drop old sent rows; enforce the pending cap (low-value types first)."""
        with self._lock:
            if self._conn is None:
                self._mem = [t for t in self._mem if not t[2]]
                return
            try:
                self._conn.execute(
                    "DELETE FROM relay_outbox WHERE sent_at IS NOT NULL AND sent_at < ?",
                    (_now_ms() - SENT_RETENTION_MS,))
                pending = self._conn.execute(
                    "SELECT COUNT(*) FROM relay_outbox WHERE sent_at IS NULL").fetchone()[0]
                if pending > PENDING_CAP:
                    excess = pending - PENDING_TRIM_TO
                    for low in LOW_VALUE_TYPES:
                        cur = self._conn.execute(
                            "DELETE FROM relay_outbox WHERE seq IN (SELECT seq FROM relay_outbox "
                            "WHERE sent_at IS NULL AND event_type=? ORDER BY seq LIMIT ?)",
                            (low, excess))
                        excess -= cur.rowcount
                        if excess <= 0:
                            break
                    if excess > 0:
                        self._conn.execute(
                            "DELETE FROM relay_outbox WHERE seq IN (SELECT seq FROM relay_outbox "
                            "WHERE sent_at IS NULL ORDER BY seq LIMIT ?)", (excess,))
                    logger.warning(f"[OUTBOX] pending cap hit ({pending}); trimmed to "
                                   f"{PENDING_TRIM_TO}")
                self._conn.commit()
            except sqlite3.Error as e:
                logger.debug(f"[OUTBOX] prune failed: {e}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None


_instance: Optional[RelayOutbox] = None
_instance_lock = threading.Lock()


def get_relay_outbox() -> RelayOutbox:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = RelayOutbox()
        return _instance
