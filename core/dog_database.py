#!/usr/bin/env python3
"""
Database storage for dog profiles and behavior history
Persists dog identification and training progress
"""

import sqlite3
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

class DogDatabase:
    """Manages dog profiles and behavior history in SQLite"""

    def __init__(self, db_path: str = "data/dogbot.db"):
        """Initialize database connection"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Dog profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dogs (
                dog_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                marker_id INTEGER UNIQUE,
                profile_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            )
        """)

        # Behavior events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_id TEXT,
                behavior TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dog_id) REFERENCES dogs (dog_id)
            )
        """)

        # Rewards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_id TEXT,
                behavior TEXT,
                treat_count INTEGER DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dog_id) REFERENCES dogs (dog_id)
            )
        """)

        self.conn.commit()

    def register_dog(self, marker_id: int, name: str, profile: Optional[Dict] = None):
        """
        Register or update a dog in the database

        Args:
            marker_id: ArUco marker ID (e.g., 315, 832)
            name: Dog's name (e.g., "Elsa", "Bezik")
            profile: Optional profile data (breed, age, etc.)
        """
        dog_id = f"aruco_{marker_id}"
        profile_json = json.dumps(profile) if profile else "{}"

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO dogs (dog_id, name, marker_id, profile_json, last_seen)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (dog_id, name, marker_id, profile_json))

        self.conn.commit()

    def record_behavior(self, marker_id: int, behavior: str, confidence: float):
        """Record a behavior event for a dog"""
        dog_id = f"aruco_{marker_id}"

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO behavior_events (dog_id, behavior, confidence)
            VALUES (?, ?, ?)
        """, (dog_id, behavior, confidence))

        # Update last_seen
        cursor.execute("""
            UPDATE dogs SET last_seen = CURRENT_TIMESTAMP
            WHERE dog_id = ?
        """, (dog_id,))

        self.conn.commit()

    def record_reward(self, marker_id: int, behavior: str, treat_count: int = 1):
        """Record a reward given to a dog"""
        dog_id = f"aruco_{marker_id}"

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO rewards (dog_id, behavior, treat_count)
            VALUES (?, ?, ?)
        """, (dog_id, behavior, treat_count))

        self.conn.commit()

    def get_dog_stats(self, marker_id: int, days: int = 1) -> Dict:
        """
        Get statistics for a dog

        Args:
            marker_id: ArUco marker ID
            days: Number of days to look back (default: 1)

        Returns:
            Dict with dog statistics
        """
        dog_id = f"aruco_{marker_id}"
        since = datetime.now() - timedelta(days=days)

        cursor = self.conn.cursor()

        # Get dog info
        cursor.execute("""
            SELECT name, profile_json FROM dogs WHERE dog_id = ?
        """, (dog_id,))
        dog_info = cursor.fetchone()

        if not dog_info:
            return {}

        name, profile_json = dog_info
        profile = json.loads(profile_json)

        # Get behavior counts
        cursor.execute("""
            SELECT behavior, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM behavior_events
            WHERE dog_id = ? AND timestamp >= ?
            GROUP BY behavior
        """, (dog_id, since))
        behaviors = {row[0]: {'count': row[1], 'avg_confidence': row[2]}
                     for row in cursor.fetchall()}

        # Get reward counts
        cursor.execute("""
            SELECT behavior, SUM(treat_count) as total_treats
            FROM rewards
            WHERE dog_id = ? AND timestamp >= ?
            GROUP BY behavior
        """, (dog_id, since))
        rewards = {row[0]: row[1] for row in cursor.fetchall()}

        # Calculate total treats
        cursor.execute("""
            SELECT SUM(treat_count) as total
            FROM rewards
            WHERE dog_id = ? AND timestamp >= ?
        """, (dog_id, since))
        total_treats = cursor.fetchone()[0] or 0

        return {
            'name': name,
            'marker_id': marker_id,
            'profile': profile,
            'behaviors': behaviors,
            'rewards': rewards,
            'total_treats': total_treats,
            'period_days': days
        }

    def generate_progress_report(self, marker_id: int) -> str:
        """
        Generate a progress report for a dog

        Args:
            marker_id: ArUco marker ID

        Returns:
            Formatted progress report string
        """
        stats = self.get_dog_stats(marker_id, days=1)

        if not stats:
            return f"No data found for dog with marker {marker_id}"

        report = []
        report.append(f"📊 Daily Progress Report for {stats['name']}")
        report.append(f"{'=' * 40}")

        # Behavior summary
        report.append("\n🎯 Behaviors Detected:")
        for behavior, data in stats['behaviors'].items():
            count = data['count']
            confidence = data['avg_confidence'] * 100
            report.append(f"  • {behavior}: {count} times ({confidence:.0f}% avg confidence)")

        # Rewards summary
        report.append(f"\n🦴 Treats Earned: {stats['total_treats']}")
        for behavior, treats in stats['rewards'].items():
            report.append(f"  • {behavior}: {treats} treats")

        # Calculate accuracy
        if 'sit' in stats['behaviors'] and 'sit' in stats['rewards']:
            sit_count = stats['behaviors']['sit']['count']
            sit_rewards = stats['rewards']['sit']
            accuracy = (sit_rewards / sit_count * 100) if sit_count > 0 else 0
            report.append(f"\n📈 Sit Accuracy: {accuracy:.0f}%")

        report.append(f"\n{stats['name']} is doing great! Keep up the good work! 🐕")

        return '\n'.join(report)

    def close(self):
        """Close database connection"""
        self.conn.close()


# Initialize default dogs on first run
def init_default_dogs():
    """Initialize Elsa and Bezik in the database"""
    db = DogDatabase()

    # Register Elsa
    db.register_dog(
        marker_id=315,
        name="Elsa",
        profile={"breed": "small_furry", "color": "white", "weight_lbs": 7}
    )

    # Register Bezik
    db.register_dog(
        marker_id=832,
        name="Bezik",
        profile={"breed": "small_furry", "color": "white", "weight_lbs": 8}
    )

    print("✅ Dogs registered in database:")
    print("  • Elsa (ID: 315)")
    print("  • Bezik (ID: 832)")

    db.close()


if __name__ == "__main__":
    # Test the database
    init_default_dogs()

    # Test recording some events
    db = DogDatabase()

    # Record some test behaviors
    db.record_behavior(315, "sit", 0.85)  # Elsa sits
    db.record_behavior(832, "stand", 0.92)  # Bezik stands
    db.record_reward(315, "sit", 1)  # Elsa gets a treat

    # Generate reports
    print("\n" + db.generate_progress_report(315))
    print("\n" + db.generate_progress_report(832))

    db.close()