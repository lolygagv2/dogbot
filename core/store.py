#!/usr/bin/env python3
"""
SQLite data persistence for TreatBot
Stores events, dog profiles, rewards, and telemetry
"""

import sqlite3
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import os


class TreatBotStore:
    """
    SQLite database for TreatBot data persistence
    Thread-safe storage for events, dogs, rewards, and telemetry
    """

    def __init__(self, db_path: str = "/home/morgan/dogbot/data/treatbot.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self.logger = logging.getLogger('TreatBotStore')

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                # Events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        type TEXT NOT NULL,
                        subtype TEXT NOT NULL,
                        source TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Dogs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS dogs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        profile_json TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_detections INTEGER DEFAULT 0,
                        total_rewards INTEGER DEFAULT 0
                    )
                ''')

                # Rewards table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rewards (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        dog_id TEXT,
                        behavior TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        treats_dispensed INTEGER DEFAULT 0,
                        mission_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dog_id) REFERENCES dogs (id)
                    )
                ''')

                # Telemetry table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        battery_pct REAL,
                        battery_voltage REAL,
                        temperature REAL,
                        mode TEXT,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Missions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS missions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        status TEXT NOT NULL,
                        rewards_given INTEGER DEFAULT 0,
                        target_rewards INTEGER DEFAULT 5,
                        config_json TEXT,
                        results_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Silent Guardian sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS silent_guardian_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_start REAL NOT NULL,
                        session_end REAL,
                        total_barks INTEGER DEFAULT 0,
                        interventions INTEGER DEFAULT 0,
                        successful_quiets INTEGER DEFAULT 0,
                        treats_dispensed INTEGER DEFAULT 0,
                        max_escalation_level INTEGER DEFAULT 0,
                        quiet_periods_json TEXT,
                        dog_bark_counts_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Silent Guardian interventions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sg_interventions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        timestamp REAL NOT NULL,
                        escalation_level INTEGER DEFAULT 0,
                        dog_id TEXT,
                        dog_name TEXT,
                        barks_triggering INTEGER DEFAULT 0,
                        quiet_achieved BOOLEAN DEFAULT 0,
                        quiet_duration REAL DEFAULT 0,
                        treat_given BOOLEAN DEFAULT 0,
                        music_played BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES silent_guardian_sessions(id)
                    )
                ''')

                # Coaching sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS coaching_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        dog_id TEXT,
                        dog_name TEXT,
                        trick_requested TEXT NOT NULL,
                        trick_completed BOOLEAN DEFAULT 0,
                        attention_duration REAL DEFAULT 0,
                        response_time REAL DEFAULT 0,
                        treat_dispensed BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events (type, subtype)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_rewards_timestamp ON rewards (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_rewards_dog ON rewards (dog_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sg_sessions_start ON silent_guardian_sessions (session_start)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sg_interventions_session ON sg_interventions (session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_coaching_timestamp ON coaching_sessions (timestamp)')

                conn.commit()
                self.logger.info("Database initialized successfully")

            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    def log_event(self, event_type: str, subtype: str, source: str, data: Dict[str, Any]) -> int:
        """
        Log an event to the database

        Args:
            event_type: Event type (vision, audio, motion, system, etc.)
            subtype: Event subtype (dog_detected, pose, bark, etc.)
            source: Event source (detector, audio, etc.)
            data: Event data

        Returns:
            int: Event ID
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (timestamp, type, subtype, source, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (time.time(), event_type, subtype, source, json.dumps(data)))

                event_id = cursor.lastrowid
                conn.commit()
                return event_id

            except Exception as e:
                self.logger.error(f"Failed to log event: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()

    def register_dog(self, dog_id: str, name: str, profile: Dict[str, Any]) -> bool:
        """Register a new dog or update existing dog profile"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO dogs (id, name, profile_json, last_seen)
                    VALUES (?, ?, ?, ?)
                ''', (dog_id, name, json.dumps(profile), datetime.now().isoformat()))

                conn.commit()
                self.logger.info(f"Dog registered: {name} ({dog_id})")
                return True

            except Exception as e:
                self.logger.error(f"Failed to register dog: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()

    def update_dog_seen(self, dog_id: str) -> bool:
        """Update last seen time for a dog"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE dogs
                    SET last_seen = ?, total_detections = total_detections + 1
                    WHERE id = ?
                ''', (datetime.now().isoformat(), dog_id))

                conn.commit()
                return cursor.rowcount > 0

            except Exception as e:
                self.logger.error(f"Failed to update dog seen: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()

    def log_reward(self, dog_id: Optional[str], behavior: str, confidence: float,
                   success: bool, treats_dispensed: int = 1, mission_name: str = "") -> int:
        """Log a reward event (non-blocking - skips if lock busy)"""
        # Use non-blocking lock to avoid blocking caller
        if not self._lock.acquire(timeout=0.5):
            self.logger.warning("log_reward skipped - lock busy")
            return -1
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO rewards (timestamp, dog_id, behavior, confidence, success,
                                       treats_dispensed, mission_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (time.time(), dog_id, behavior, confidence, success, treats_dispensed, mission_name))

                reward_id = cursor.lastrowid

                # Update dog's total rewards if successful
                if success and dog_id:
                    cursor.execute('''
                        UPDATE dogs
                        SET total_rewards = total_rewards + ?
                        WHERE id = ?
                    ''', (treats_dispensed, dog_id))

                conn.commit()
                return reward_id

            except Exception as e:
                self.logger.error(f"Failed to log reward: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()
        finally:
            self._lock.release()

    def log_telemetry(self, battery_pct: float = None, battery_voltage: float = None,
                     temperature: float = None, mode: str = None, cpu_usage: float = None,
                     memory_usage: float = None, disk_usage: float = None) -> bool:
        """Log system telemetry (non-blocking - skips if lock busy)"""
        # Use non-blocking lock to avoid blocking main loop heartbeat
        if not self._lock.acquire(timeout=0.1):
            self.logger.debug("Telemetry skipped - lock busy")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO telemetry (timestamp, battery_pct, battery_voltage, temperature,
                                         mode, cpu_usage, memory_usage, disk_usage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (time.time(), battery_pct, battery_voltage, temperature, mode,
                      cpu_usage, memory_usage, disk_usage))

                conn.commit()
                return True

            except Exception as e:
                self.logger.error(f"Failed to log telemetry: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()
        finally:
            self._lock.release()

    def start_mission(self, name: str, target_rewards: int = 5, config: Dict[str, Any] = None) -> int:
        """Start a new mission"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO missions (name, start_time, status, target_rewards, config_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, time.time(), 'active', target_rewards, json.dumps(config or {})))

                mission_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Mission started: {name} (ID: {mission_id})")
                return mission_id

            except Exception as e:
                self.logger.error(f"Failed to start mission: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()

    def end_mission(self, mission_id: int, status: str, results: Dict[str, Any] = None) -> bool:
        """End a mission"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE missions
                    SET end_time = ?, status = ?, results_json = ?
                    WHERE id = ?
                ''', (time.time(), status, json.dumps(results or {}), mission_id))

                conn.commit()
                self.logger.info(f"Mission ended: ID {mission_id}, Status: {status}")
                return cursor.rowcount > 0

            except Exception as e:
                self.logger.error(f"Failed to end mission: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()

    def get_recent_events(self, limit: int = 50, event_type: str = None) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                if event_type:
                    cursor.execute('''
                        SELECT * FROM events
                        WHERE type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (event_type, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM events
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))

                columns = [desc[0] for desc in cursor.description]
                events = []
                for row in cursor.fetchall():
                    event = dict(zip(columns, row))
                    event['payload'] = json.loads(event['payload_json'])
                    del event['payload_json']
                    events.append(event)

                return events

            except Exception as e:
                self.logger.error(f"Failed to get recent events: {e}")
                return []
            finally:
                conn.close()

    def get_dog_stats(self, dog_id: str = None) -> List[Dict[str, Any]]:
        """Get dog statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                if dog_id:
                    cursor.execute('SELECT * FROM dogs WHERE id = ?', (dog_id,))
                else:
                    cursor.execute('SELECT * FROM dogs ORDER BY last_seen DESC')

                columns = [desc[0] for desc in cursor.description]
                dogs = []
                for row in cursor.fetchall():
                    dog = dict(zip(columns, row))
                    dog['profile'] = json.loads(dog['profile_json'])
                    del dog['profile_json']
                    dogs.append(dog)

                return dogs

            except Exception as e:
                self.logger.error(f"Failed to get dog stats: {e}")
                return []
            finally:
                conn.close()

    def get_reward_history(self, dog_id: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """Get reward history"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                since_time = time.time() - (days * 24 * 3600)

                if dog_id:
                    cursor.execute('''
                        SELECT * FROM rewards
                        WHERE dog_id = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (dog_id, since_time))
                else:
                    cursor.execute('''
                        SELECT * FROM rewards
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (since_time,))

                columns = [desc[0] for desc in cursor.description]
                rewards = []
                for row in cursor.fetchall():
                    reward = dict(zip(columns, row))
                    rewards.append(reward)

                return rewards

            except Exception as e:
                self.logger.error(f"Failed to get reward history: {e}")
                return []
            finally:
                conn.close()

    def cleanup_old_data(self, days_to_keep: int = 30) -> Tuple[int, int]:
        """Clean up old events and telemetry data"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cutoff_time = time.time() - (days_to_keep * 24 * 3600)

                # Clean events
                cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff_time,))
                events_deleted = cursor.rowcount

                # Clean telemetry
                cursor.execute('DELETE FROM telemetry WHERE timestamp < ?', (cutoff_time,))
                telemetry_deleted = cursor.rowcount

                conn.commit()
                self.logger.info(f"Cleaned up {events_deleted} events, {telemetry_deleted} telemetry records")
                return events_deleted, telemetry_deleted

            except Exception as e:
                self.logger.error(f"Failed to cleanup old data: {e}")
                conn.rollback()
                return 0, 0
            finally:
                conn.close()

    def get_today_reward_count(self, dog_id: str = None) -> int:
        """Get today's reward count for a specific dog or all dogs"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                # Get today's start timestamp (midnight)
                from datetime import datetime, time as datetime_time
                today_start = datetime.combine(datetime.today(), datetime_time.min).timestamp()

                if dog_id:
                    cursor.execute('''
                        SELECT COUNT(*) FROM rewards
                        WHERE dog_id = ? AND timestamp >= ? AND success = 1
                    ''', (dog_id, today_start))
                else:
                    cursor.execute('''
                        SELECT COUNT(*) FROM rewards
                        WHERE timestamp >= ? AND success = 1
                    ''', (today_start,))

                return cursor.fetchone()[0]

            except Exception as e:
                self.logger.error(f"Failed to get today's reward count: {e}")
                return 0
            finally:
                conn.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                stats = {}

                # Count records in each table
                for table in ['events', 'dogs', 'rewards', 'telemetry', 'missions']:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]

                # Database file size
                stats['db_size_bytes'] = os.path.getsize(self.db_path)
                stats['db_size_mb'] = round(stats['db_size_bytes'] / (1024 * 1024), 2)

                # Recent activity
                cursor.execute('SELECT COUNT(*) FROM events WHERE timestamp > ?', (time.time() - 3600,))
                stats['events_last_hour'] = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM rewards WHERE timestamp > ?', (time.time() - 86400,))
                stats['rewards_last_day'] = cursor.fetchone()[0]

                return stats

            except Exception as e:
                self.logger.error(f"Failed to get database stats: {e}")
                return {}
            finally:
                conn.close()

    # ========== Silent Guardian Methods ==========

    def start_silent_guardian_session(self) -> int:
        """Start a new Silent Guardian session (non-blocking)"""
        if not self._lock.acquire(timeout=0.5):
            self.logger.warning("start_silent_guardian_session skipped - lock busy")
            return -1
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO silent_guardian_sessions (session_start)
                    VALUES (?)
                ''', (time.time(),))

                session_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Silent Guardian session started (ID: {session_id})")
                return session_id

            except Exception as e:
                self.logger.error(f"Failed to start SG session: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()
        finally:
            self._lock.release()

    def end_silent_guardian_session(self, session_id: int, total_barks: int = 0,
                                    interventions: int = 0, successful_quiets: int = 0,
                                    treats_dispensed: int = 0, max_escalation: int = 0,
                                    quiet_periods: List = None, dog_bark_counts: Dict = None) -> bool:
        """End a Silent Guardian session with summary data (non-blocking)"""
        if not self._lock.acquire(timeout=0.5):
            self.logger.warning("end_silent_guardian_session skipped - lock busy")
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE silent_guardian_sessions
                    SET session_end = ?, total_barks = ?, interventions = ?,
                        successful_quiets = ?, treats_dispensed = ?, max_escalation_level = ?,
                        quiet_periods_json = ?, dog_bark_counts_json = ?
                    WHERE id = ?
                ''', (time.time(), total_barks, interventions, successful_quiets,
                      treats_dispensed, max_escalation, json.dumps(quiet_periods or []),
                      json.dumps(dog_bark_counts or {}), session_id))

                conn.commit()
                self.logger.info(f"Silent Guardian session ended (ID: {session_id})")
                return True

            except Exception as e:
                self.logger.error(f"Failed to end SG session: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()
        finally:
            self._lock.release()

    def log_sg_intervention(self, session_id: int, escalation_level: int = 0,
                           dog_id: str = None, dog_name: str = None,
                           barks_triggering: int = 0, quiet_achieved: bool = False,
                           quiet_duration: float = 0, treat_given: bool = False,
                           music_played: bool = False) -> int:
        """Log a Silent Guardian intervention event (non-blocking)"""
        if not self._lock.acquire(timeout=0.5):
            self.logger.warning("log_sg_intervention skipped - lock busy")
            return -1
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sg_interventions
                    (session_id, timestamp, escalation_level, dog_id, dog_name,
                     barks_triggering, quiet_achieved, quiet_duration, treat_given, music_played)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, time.time(), escalation_level, dog_id, dog_name,
                      barks_triggering, quiet_achieved, quiet_duration, treat_given, music_played))

                intervention_id = cursor.lastrowid
                conn.commit()
                return intervention_id

            except Exception as e:
                self.logger.error(f"Failed to log SG intervention: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()
        finally:
            self._lock.release()

    def get_sg_session_stats(self, session_id: int = None, days: int = 7) -> Dict[str, Any]:
        """Get Silent Guardian session statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                if session_id:
                    # Get specific session
                    cursor.execute('''
                        SELECT * FROM silent_guardian_sessions WHERE id = ?
                    ''', (session_id,))
                    row = cursor.fetchone()
                    if row:
                        return {
                            'id': row[0], 'session_start': row[1], 'session_end': row[2],
                            'total_barks': row[3], 'interventions': row[4],
                            'successful_quiets': row[5], 'treats_dispensed': row[6],
                            'max_escalation_level': row[7]
                        }
                    return {}
                else:
                    # Get sessions from last N days
                    cutoff = time.time() - (days * 86400)
                    cursor.execute('''
                        SELECT COUNT(*), SUM(total_barks), SUM(interventions),
                               SUM(successful_quiets), SUM(treats_dispensed), MAX(max_escalation_level)
                        FROM silent_guardian_sessions
                        WHERE session_start > ?
                    ''', (cutoff,))
                    row = cursor.fetchone()
                    return {
                        'sessions': row[0] or 0,
                        'total_barks': row[1] or 0,
                        'total_interventions': row[2] or 0,
                        'total_successful_quiets': row[3] or 0,
                        'total_treats': row[4] or 0,
                        'max_escalation': row[5] or 0
                    }

            except Exception as e:
                self.logger.error(f"Failed to get SG stats: {e}")
                return {}
            finally:
                conn.close()

    # ========== Coaching Methods ==========

    def log_coaching_session(self, dog_id: str, dog_name: str, trick_requested: str,
                            trick_completed: bool = False, attention_duration: float = 0,
                            response_time: float = 0, treat_dispensed: bool = False) -> int:
        """Log a coaching interaction"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO coaching_sessions
                    (timestamp, dog_id, dog_name, trick_requested, trick_completed,
                     attention_duration, response_time, treat_dispensed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (time.time(), dog_id, dog_name, trick_requested, trick_completed,
                      attention_duration, response_time, treat_dispensed))

                session_id = cursor.lastrowid
                conn.commit()
                return session_id

            except Exception as e:
                self.logger.error(f"Failed to log coaching session: {e}")
                conn.rollback()
                return -1
            finally:
                conn.close()

    def get_coaching_stats(self, dog_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Get coaching statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cutoff = time.time() - (days * 86400)

                if dog_id:
                    cursor.execute('''
                        SELECT COUNT(*), SUM(trick_completed), SUM(treat_dispensed),
                               AVG(response_time)
                        FROM coaching_sessions
                        WHERE dog_id = ? AND timestamp > ?
                    ''', (dog_id, cutoff))
                else:
                    cursor.execute('''
                        SELECT COUNT(*), SUM(trick_completed), SUM(treat_dispensed),
                               AVG(response_time)
                        FROM coaching_sessions
                        WHERE timestamp > ?
                    ''', (cutoff,))

                row = cursor.fetchone()
                return {
                    'total_sessions': row[0] or 0,
                    'tricks_completed': row[1] or 0,
                    'treats_given': row[2] or 0,
                    'avg_response_time': row[3] or 0,
                    'success_rate': (row[1] / row[0] * 100) if row[0] > 0 else 0
                }

            except Exception as e:
                self.logger.error(f"Failed to get coaching stats: {e}")
                return {}
            finally:
                conn.close()


# Global store instance
_store_instance = None
_store_lock = threading.Lock()

def get_store() -> TreatBotStore:
    """Get the global store instance (singleton)"""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = TreatBotStore()
    return _store_instance


if __name__ == "__main__":
    # Test the store
    store = get_store()

    # Test logging events
    event_id = store.log_event('vision', 'dog_detected', 'detector', {
        'confidence': 0.95,
        'bbox': [100, 100, 200, 200]
    })
    print(f"Logged event ID: {event_id}")

    # Test registering a dog
    store.register_dog('dog_001', 'Buddy', {
        'breed': 'Golden Retriever',
        'age': 3,
        'training_level': 'intermediate'
    })

    # Test logging a reward
    reward_id = store.log_reward('dog_001', 'sit', 0.87, True, 1, 'test_mission')
    print(f"Logged reward ID: {reward_id}")

    # Test telemetry
    store.log_telemetry(battery_voltage=13.2, temperature=42.5, mode='detection')

    # Get recent events
    events = store.get_recent_events(10)
    print(f"Recent events: {len(events)}")

    # Get stats
    stats = store.get_database_stats()
    print("Database stats:", stats)