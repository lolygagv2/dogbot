#!/usr/bin/env python3
"""
Bark Store - SQLite storage for bark detection history
Tracks barks by dog, emotion, loudness, and frequency
"""

import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / 'data' / 'treatbot.db'


class BarkStore:
    """SQLite storage for bark detection data"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._ensure_table()
        logger.info(f"BarkStore initialized with database: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        """Create barks table if it doesn't exist"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS barks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    dog_id TEXT,
                    dog_name TEXT,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    loudness_db REAL,
                    duration_ms INTEGER,
                    session_id TEXT
                )
            ''')

            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_barks_dog ON barks(dog_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_barks_emotion ON barks(emotion)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_barks_timestamp ON barks(timestamp)')

            conn.commit()
            logger.debug("Barks table ensured")
        finally:
            conn.close()

    def log_bark(self, emotion: str, confidence: float, loudness_db: float = None,
                 dog_id: str = None, dog_name: str = None, duration_ms: int = None,
                 session_id: str = None) -> int:
        """
        Log a bark event to the database

        Args:
            emotion: Detected emotion (alert, anxious, scared, etc.)
            confidence: Detection confidence (0-1)
            loudness_db: Loudness in decibels
            dog_id: ArUco marker ID (e.g., "aruco_315")
            dog_name: Dog name (e.g., "Elsa")
            duration_ms: Bark duration in milliseconds
            session_id: Session identifier for grouping

        Returns:
            ID of inserted row
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO barks (dog_id, dog_name, emotion, confidence, loudness_db, duration_ms, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (dog_id, dog_name, emotion, confidence, loudness_db, duration_ms, session_id))
            conn.commit()

            bark_id = cursor.lastrowid
            logger.debug(f"Logged bark: {dog_name or dog_id or 'unknown'} - {emotion} ({confidence:.2f})")
            return bark_id
        finally:
            conn.close()

    def get_barks_in_window(self, dog_id: str = None, seconds: int = 60) -> List[Dict]:
        """
        Get barks within a time window

        Args:
            dog_id: Filter by dog (optional)
            seconds: Time window in seconds

        Returns:
            List of bark records
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(seconds=seconds)

            if dog_id:
                cursor.execute('''
                    SELECT * FROM barks
                    WHERE timestamp >= ? AND dog_id = ?
                    ORDER BY timestamp DESC
                ''', (cutoff.isoformat(), dog_id))
            else:
                cursor.execute('''
                    SELECT * FROM barks
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (cutoff.isoformat(),))

            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_barks_per_minute(self, dog_id: str = None, minutes: int = 1) -> int:
        """
        Get bark count in the last N minutes

        Args:
            dog_id: Filter by dog (optional)
            minutes: Number of minutes to look back

        Returns:
            Bark count
        """
        barks = self.get_barks_in_window(dog_id, seconds=minutes * 60)
        return len(barks)

    def get_bark_history(self, dog_id: str = None, limit: int = 100) -> List[Dict]:
        """
        Get recent bark history

        Args:
            dog_id: Filter by dog (optional)
            limit: Maximum records to return

        Returns:
            List of bark records
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if dog_id:
                cursor.execute('''
                    SELECT * FROM barks
                    WHERE dog_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (dog_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM barks
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_bark_stats(self, dog_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get bark statistics for a time period

        Args:
            dog_id: Filter by dog (optional)
            hours: Hours to analyze

        Returns:
            Statistics dictionary
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cutoff = datetime.now() - timedelta(hours=hours)

            # Base query
            if dog_id:
                where_clause = "WHERE timestamp >= ? AND dog_id = ?"
                params = (cutoff.isoformat(), dog_id)
            else:
                where_clause = "WHERE timestamp >= ?"
                params = (cutoff.isoformat(),)

            # Total count
            cursor.execute(f'SELECT COUNT(*) as count FROM barks {where_clause}', params)
            total = cursor.fetchone()['count']

            # By emotion
            cursor.execute(f'''
                SELECT emotion, COUNT(*) as count
                FROM barks {where_clause}
                GROUP BY emotion
            ''', params)
            by_emotion = {row['emotion']: row['count'] for row in cursor.fetchall()}

            # By dog
            cursor.execute(f'''
                SELECT dog_name, dog_id, COUNT(*) as count
                FROM barks {where_clause}
                GROUP BY dog_id
            ''', params)
            by_dog = {
                row['dog_name'] or row['dog_id'] or 'unknown': row['count']
                for row in cursor.fetchall()
            }

            # Average loudness
            cursor.execute(f'''
                SELECT AVG(loudness_db) as avg_loud, MAX(loudness_db) as max_loud
                FROM barks {where_clause}
            ''', params)
            loud_row = cursor.fetchone()

            # Average confidence
            cursor.execute(f'''
                SELECT AVG(confidence) as avg_conf
                FROM barks {where_clause}
            ''', params)
            conf_row = cursor.fetchone()

            return {
                'total_barks': total,
                'by_emotion': by_emotion,
                'by_dog': by_dog,
                'avg_loudness_db': loud_row['avg_loud'],
                'max_loudness_db': loud_row['max_loud'],
                'avg_confidence': conf_row['avg_conf'],
                'hours_analyzed': hours
            }
        finally:
            conn.close()

    def get_frequency_events(self, dog_id: str, threshold: int = 3,
                              window_minutes: int = 1, hours: int = 24) -> List[Dict]:
        """
        Find time periods where bark frequency exceeded threshold

        Args:
            dog_id: Dog to analyze
            threshold: Barks per window to consider an "event"
            window_minutes: Window size in minutes
            hours: How far back to look

        Returns:
            List of frequency event periods
        """
        # This is a simplified version - for full implementation,
        # you'd want to use SQL window functions or analyze in Python
        barks = self.get_bark_history(dog_id, limit=1000)
        events = []

        # Group barks by minute
        by_minute = {}
        for bark in barks:
            ts = bark['timestamp']
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            minute_key = ts.strftime('%Y-%m-%d %H:%M')
            by_minute[minute_key] = by_minute.get(minute_key, 0) + 1

        # Find minutes that exceeded threshold
        for minute, count in by_minute.items():
            if count >= threshold:
                events.append({
                    'minute': minute,
                    'bark_count': count,
                    'threshold': threshold
                })

        return sorted(events, key=lambda x: x['minute'], reverse=True)


# Singleton instance
_bark_store_instance = None


def get_bark_store() -> BarkStore:
    """Get or create bark store singleton"""
    global _bark_store_instance
    if _bark_store_instance is None:
        _bark_store_instance = BarkStore()
    return _bark_store_instance
