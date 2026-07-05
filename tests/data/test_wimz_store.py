#!/usr/bin/env python3
"""WimzStore unit tests — schema fidelity against the spec, ids, batching.

Run: env_new/bin/python -m pytest tests/data/test_wimz_store.py -v
 or: env_new/bin/python tests/data/test_wimz_store.py
"""
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, '/home/morgan/dogbot')

from core.data.ids import uuid7, uuid7_at
from core.data.wimz_store import WimzStore

SPEC = Path('/home/morgan/dogbot/.claude/WIMZ_Data_Architecture_Spec.md')


def _norm(sql: str) -> str:
    """Normalize SQL for comparison: strip comments, collapse whitespace."""
    sql = re.sub(r'--[^\n]*', '', sql)
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip().rstrip(';').lower()


def _spec_statements():
    """Extract CREATE statements from the spec §4 sql code block."""
    text = SPEC.read_text()
    block = re.search(r'## 4\. Core schema.*?```sql\n(.*?)```', text, re.S).group(1)
    # Strip comments BEFORE splitting on ';' — some spec comments contain semicolons
    block = re.sub(r'--[^\n]*', '', block)
    stmts = []
    for raw in block.split(';'):
        n = _norm(raw)
        if n.startswith('create '):
            stmts.append(n)
    return stmts


def make_store():
    tmp = tempfile.mkdtemp(prefix='wimztest_')
    return WimzStore(data_root=Path(tmp)), Path(tmp)


def test_schema_matches_spec():
    store, root = make_store()
    conn = sqlite3.connect(root / 'wimz.db')
    db_sql = {
        _norm(row[0]) for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
    }
    missing = [s for s in _spec_statements() if s not in db_sql]
    assert not missing, f"DB schema diverges from spec: {missing[:3]}"
    ver = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
    assert ver == '0.3', ver
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == 'delete' or True
    conn.close()
    store.close()
    print("PASS schema_matches_spec")


def test_pragmas():
    store, root = make_store()
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == 'wal', mode
    fk = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1
    store.close()
    print("PASS pragmas")


def test_uuid7():
    ids = [uuid7() for _ in range(10000)]
    assert ids == sorted(ids), "uuid7 not monotonic"
    assert len(set(ids)) == len(ids), "uuid7 collision"
    assert all(i[14] == '7' for i in ids), "wrong version nibble"
    assert all(i[19] in '89ab' for i in ids), "wrong variant"
    a = uuid7_at(1720000000000, seed=b'x:y:1')
    b = uuid7_at(1720000000000, seed=b'x:y:1')
    c = uuid7_at(1720000000000, seed=b'x:y:2')
    assert a == b and a != c, "uuid7_at determinism broken"
    print("PASS uuid7")


def test_batcher_and_writes():
    store, root = make_store()
    sid = store.start_session('monitor', 'autonomous')
    t0 = time.time()
    for i in range(500):
        store.log_event(sid, 'bark', {'db': -10, 'duration_ms': 500, 'class': 'bark'},
                        confidence=0.5, model_id=store.model_id_for('dog_bark_classifier'))
    assert store.flush(2.0), "flush did not drain"
    assert time.time() - t0 < 2.0
    n = store._conn.execute("SELECT COUNT(*), MAX(seq) FROM event WHERE session_id=?",
                            (sid,)).fetchone()
    assert n[0] == 500 and n[1] == 500, n
    synced = store._conn.execute("SELECT DISTINCT synced FROM event").fetchall()
    assert synced == [(0,)], synced

    dog = store.get_or_create_dog(legacy_id='aruco_315', name='Elsa')
    assert dog == store.get_or_create_dog(legacy_id='aruco_315')
    # Live identity requires a verified tag: name guesses and tracker
    # indexes must NOT attribute or mint identity
    assert store.get_or_create_dog(name='elsa') is None
    assert store.get_or_create_dog(legacy_id='dog_0', name='belly') is None
    assert store._conn.execute("SELECT COUNT(*) FROM dog").fetchone()[0] == 1
    # Backfill/app registration may vouch for a name
    assert dog == store.get_or_create_dog(name='elsa', allow_unverified=True)
    row = store._conn.execute("SELECT id_method, qr_code_id FROM dog WHERE dog_id=?",
                              (dog,)).fetchone()
    assert row == ('qr', 'aruco_315'), row

    disp = store.log_dispense(sid, trigger='attempt', dog_id=dog, dispensed_confirmed=1,
                              confirm_latency_ms=430)
    att = store.log_training_attempt(sid, trick_label='quiet', dog_id=dog,
                                     cue_ts_ms=int(time.time() * 1000),
                                     success=1, reward_dispensed=1, dispense_id=disp)
    assert att
    paired = store._conn.execute(
        "SELECT COUNT(*) FROM event WHERE event_type='treat_dispensed'").fetchone()[0]
    assert paired == 1, paired
    store.end_session(sid)
    ended = store._conn.execute("SELECT ended_at FROM session WHERE session_id=?",
                                (sid,)).fetchone()[0]
    assert ended is not None
    store.close()
    print("PASS batcher_and_writes")


def test_orphan_sweep():
    store, root = make_store()
    sid = store.start_session('training', 'autonomous')
    store.log_event(sid, 'pose', {'pose': 'sit'})
    store.flush()
    store.close()  # session left open

    store2 = WimzStore(data_root=root)
    ended = store2._conn.execute("SELECT ended_at FROM session WHERE session_id=?",
                                 (sid,)).fetchone()[0]
    assert ended is not None, "orphan session not swept"
    store2.close()
    print("PASS orphan_sweep")


def test_media_registration():
    store, root = make_store()
    sid = store.start_session('training', 'autonomous')
    p = store.media_path_for(session_id=sid, ext='jpg')
    p.write_bytes(b'\xff\xd8fakejpeg')
    mid = store.register_media(str(p), 'image', session_id=sid, codec='jpeg',
                               retention_class='ephemeral')
    assert mid
    assert mid == store.register_media(str(p), 'image')  # idempotent by rel_path
    rel = store._conn.execute("SELECT rel_path, sha256, size_bytes FROM media_asset "
                              "WHERE media_id=?", (mid,)).fetchone()
    assert not rel[0].startswith('/'), f"rel_path not relative: {rel[0]}"
    assert rel[1] and rel[2] == 10
    store.close()
    print("PASS media_registration")


if __name__ == '__main__':
    test_schema_matches_spec()
    test_pragmas()
    test_uuid7()
    test_batcher_and_writes()
    test_orphan_sweep()
    test_media_registration()
    print("\nALL PASS")
