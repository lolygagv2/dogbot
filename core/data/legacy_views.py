"""Phase 4: wimz-backed replacements for REST endpoints that read the
retired legacy tables (dog_events, sg catch-up). Response shapes mirror
the legacy getters; timestamps serialize per the boundary rule
(ISO8601 UTC 'Z') except where the legacy wire shape was unix-seconds.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.data import DATA_ROOT


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(f'file:{DATA_ROOT}/wimz.db?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    return c


def _iso_z(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return (datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            .isoformat().replace('+00:00', 'Z'))


def _event_row(r: sqlite3.Row) -> Dict[str, Any]:
    try:
        details = json.loads(r['payload']) if r['payload'] else {}
    except Exception:
        details = {'raw': r['payload']}
    return {
        'id': r['event_id'],
        'timestamp': _iso_z(r['ts']),
        'event_type': r['event_type'],
        'dog_id': r['app_dog_id'] or r['dog_id'] or 'unknown',
        'dog_name': r['name'] or '',
        'details': details,
        'mode': r['mode'],
        'session_id': r['session_id'],
    }


_BASE = ('''SELECT e.event_id, e.ts, e.event_type, e.payload, e.session_id,
                   e.dog_id, d.name, d.app_dog_id, s.mode
            FROM event e
            LEFT JOIN dog d ON e.dog_id = d.dog_id
            LEFT JOIN session s ON e.session_id = s.session_id
            WHERE 1=1 ''')


def get_dog_events(limit: int = 100, event_type: str = None,
                   dog_id: str = None, since: str = None) -> List[Dict[str, Any]]:
    q, params = _BASE, []
    if event_type:
        q += ' AND e.event_type = ?'
        params.append(event_type)
    if dog_id:
        q += (' AND e.dog_id IN (SELECT dog_id FROM dog WHERE dog_id=? '
              'OR app_dog_id=? OR lower(name)=lower(?))')
        params += [dog_id, dog_id, dog_id]
    if since:
        try:
            ms = int(datetime.fromisoformat(
                since.replace('Z', '+00:00')).timestamp() * 1000)
            q += ' AND e.ts >= ?'
            params.append(ms)
        except ValueError:
            pass
    q += ' ORDER BY e.ts DESC LIMIT ?'
    params.append(limit)
    with _conn() as c:
        return [_event_row(r) for r in c.execute(q, params)]


def get_dog_events_summary(hours: int = 24) -> Dict[str, Any]:
    since_ms = int((datetime.now(tz=timezone.utc)
                    - timedelta(hours=hours)).timestamp() * 1000)
    with _conn() as c:
        by_type = {r[0]: r[1] for r in c.execute(
            'SELECT event_type, COUNT(*) FROM event WHERE ts >= ? '
            'GROUP BY event_type', (since_ms,))}
        by_dog = {r[0]: r[1] for r in c.execute(
            'SELECT d.name, COUNT(*) FROM event e JOIN dog d ON e.dog_id=d.dog_id '
            'WHERE e.ts >= ? AND d.name IS NOT NULL GROUP BY d.name', (since_ms,))}
        total = c.execute('SELECT COUNT(*) FROM event WHERE ts >= ?',
                          (since_ms,)).fetchone()[0]
    return {'period_hours': hours, 'total_events': total,
            'by_type': by_type, 'by_dog': by_dog}


def get_dog_events_latest(dog_id: str = None) -> List[Dict[str, Any]]:
    q = _BASE
    params: list = []
    if dog_id:
        q += (' AND e.dog_id IN (SELECT dog_id FROM dog WHERE dog_id=? '
              'OR app_dog_id=? OR lower(name)=lower(?))')
        params += [dog_id, dog_id, dog_id]
    q += ' ORDER BY e.ts DESC LIMIT 1'
    with _conn() as c:
        return [_event_row(r) for r in c.execute(q, params)]


def get_sg_sessions_since(since_ts: float = 0.0, limit: int = 20) -> List[Dict[str, Any]]:
    """App offline catch-up: SG sessions (+ their intervention outcomes)
    started after since_ts. Legacy wire shape kept: unix-second floats,
    same field names; 'id' is now the session UUID (was an int)."""
    out = []
    with _conn() as c:
        for s in c.execute(
                "SELECT session_id, started_at, ended_at, outcome_json "
                "FROM session WHERE mode='sg' AND started_at > ? "
                "ORDER BY started_at DESC LIMIT ?",
                (int(since_ts * 1000), limit)):
            o = json.loads(s['outcome_json']) if s['outcome_json'] else {}
            sess = {
                'id': s['session_id'],
                'session_start': s['started_at'] / 1000.0,
                'session_end': s['ended_at'] / 1000.0 if s['ended_at'] else None,
                'total_barks': o.get('total_barks', 0),
                'interventions_count': o.get('interventions_triggered', 0),
                'successful_quiets': o.get('successful_quiets', 0),
                'treats_dispensed': o.get('treats_dispensed', 0),
                'max_escalation_level': o.get('max_escalation_level', 0),
                'interventions': [],
            }
            for e in c.execute(
                    "SELECT e.ts, e.payload, d.name, d.app_dog_id, e.dog_id "
                    "FROM event e LEFT JOIN dog d ON e.dog_id = d.dog_id "
                    "WHERE e.session_id=? AND e.event_type='sg_intervention' "
                    "AND json_extract(e.payload,'$.phase')='outcome' ORDER BY e.ts",
                    (s['session_id'],)):
                p = json.loads(e['payload']) if e['payload'] else {}
                sess['interventions'].append({
                    'timestamp': e['ts'] / 1000.0,
                    'escalation_level': p.get('escalation_level'),
                    'dog_id': e['app_dog_id'] or e['dog_id'],
                    'dog_name': e['name'] or '',
                    'quiet_achieved': bool(p.get('quiet_achieved')),
                    'treat_given': bool(p.get('treat_given')),
                    'music_played': bool(p.get('music_played')),
                })
            out.append(sess)
    return out
