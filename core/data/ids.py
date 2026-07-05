"""UUIDv7 generation (RFC 9562) — Python 3.11 has no uuid.uuid7.

uuid7()    — time-ordered random ids for live writes.
uuid7_at() — v7-format ids at a historical timestamp, optionally seeded for
             determinism. Backfill use ONLY: re-runs of the migration produce
             identical ids for identical source rows, so INSERT OR IGNORE
             makes the whole backfill idempotent.
"""
import hashlib
import os
import threading
import time
import uuid

_lock = threading.Lock()
_last_ms = 0
_seq = 0  # 12-bit sub-millisecond counter in rand_a for same-ms monotonicity


def uuid7() -> str:
    """Time-ordered UUIDv7, monotonic within this process."""
    global _last_ms, _seq
    with _lock:
        ms = time.time_ns() // 1_000_000
        if ms == _last_ms:
            _seq = (_seq + 1) & 0x0FFF
        else:
            _last_ms, _seq = ms, 0
        b = bytearray(
            ms.to_bytes(6, 'big')
            + _seq.to_bytes(2, 'big')  # rand_a used as sub-ms counter
            + os.urandom(8)            # rand_b
        )
    b[6] = (b[6] & 0x0F) | 0x70  # version 7
    b[8] = (b[8] & 0x3F) | 0x80  # variant 10x
    return str(uuid.UUID(bytes=bytes(b)))


def uuid7_at(ts_ms: int, seed: bytes = None) -> str:
    """v7-format id stamped at a historical epoch-ms timestamp.

    With a seed, the id is deterministic (sha256-derived tail) — used by the
    backfill script so re-runs cannot duplicate rows. Without a seed the tail
    is random.
    """
    tail = hashlib.sha256(seed).digest()[:10] if seed else os.urandom(10)
    b = bytearray(ts_ms.to_bytes(6, 'big') + tail)
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))
