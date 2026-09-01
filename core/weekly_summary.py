#!/usr/bin/env python3
"""
Weekly Summary & Behavioral Analysis for WIM-Z
Generates comprehensive weekly reports with trends and insights

Features:
- Weekly bark statistics by dog, emotion, day
- Reward statistics by behavior, dog
- Silent Guardian effectiveness metrics
- Coaching session progress
- Week-over-week trend analysis (8 weeks)
- Export to markdown and CSV
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / 'data' / 'treatbot.db'

# Reports directory
REPORTS_DIR = Path(__file__).parent.parent / 'reports'


class WeeklySummary:
    """
    Generate weekly behavioral analysis reports

    Aggregates data from (refactor Phase 3):
    - wimz.db: bark events, SG sessions (session.outcome_json),
      sg_intervention events, training_attempt (coaching), dog identity
    - treatbot.db (legacy, until their writers cut over): rewards, missions
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.reports_dir = REPORTS_DIR

        # Ensure reports directory exists
        os.makedirs(self.reports_dir, exist_ok=True)

        logger.info(f"WeeklySummary initialized with database: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- wimz.db (spec store) access — refactor Phase 3 ----
    # Barks, SG sessions, and coaching now read the spec store (single
    # source, epoch-ms UTC, survives the legacy 24h prune). Rewards and
    # missions still read treatbot.db until their writers cut over.

    def _wimz_connection(self) -> sqlite3.Connection:
        from core.data import DATA_ROOT
        conn = sqlite3.connect(os.path.join(DATA_ROOT, 'wimz.db'))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ms(dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    # Real barks only: exclude 'notbark' veto rows kept as hard negatives.
    _BARK_WHERE = ("event_type='bark' "
                   "AND COALESCE(json_extract(payload,'$.class'),'bark') != 'notbark' "
                   "AND ts BETWEEN ? AND ?")

    def _wimz_bark_count(self, start: datetime, end: datetime,
                         dog_key: str = None, dog_name: str = None) -> int:
        """Bark count; dog filter matches local id, app id, or name."""
        conn = self._wimz_connection()
        try:
            q = f"SELECT COUNT(*) AS c FROM event e WHERE {self._BARK_WHERE}"
            params = [self._ms(start), self._ms(end)]
            if dog_key or dog_name:
                q += (" AND e.dog_id IN (SELECT dog_id FROM dog WHERE dog_id=? "
                      "OR app_dog_id=? OR lower(name)=lower(?))")
                params += [dog_key, dog_key, dog_name or dog_key]
            return conn.execute(q, params).fetchone()['c'] or 0
        finally:
            conn.close()

    def _wimz_coaching_agg(self, start: datetime, end: datetime,
                           dog_key: str = None, dog_name: str = None) -> Dict[str, Any]:
        """Coach attempts/completions from training_attempt (coach sessions
        only — SG quiet-attempts live in sg/monitor sessions)."""
        conn = self._wimz_connection()
        try:
            q = ('''SELECT COUNT(*) AS sessions,
                           SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END) AS completed,
                           SUM(ta.reward_dispensed) AS treats,
                           AVG(ta.latency_ms) AS avg_latency_ms
                    FROM training_attempt ta
                    JOIN session s ON ta.session_id = s.session_id
                    WHERE s.mode IN ('coach','training')
                      AND ta.cue_ts BETWEEN ? AND ?''')
            params = [self._ms(start), self._ms(end)]
            if dog_key or dog_name:
                q += (" AND ta.dog_id IN (SELECT dog_id FROM dog WHERE dog_id=? "
                      "OR app_dog_id=? OR lower(name)=lower(?))")
                params += [dog_key, dog_key, dog_name or dog_key]
            return dict(conn.execute(q, params).fetchone() or {})
        finally:
            conn.close()

    def _get_week_bounds(self, end_date: datetime = None) -> Tuple[datetime, datetime]:
        """
        Get start and end of week (Monday-Sunday)

        Args:
            end_date: End date (defaults to today)

        Returns:
            Tuple of (week_start, week_end) as datetime objects
        """
        if end_date is None:
            end_date = datetime.now()

        # Get Monday of the week
        days_since_monday = end_date.weekday()
        week_start = end_date - timedelta(days=days_since_monday)
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Get Sunday end of week
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

        return week_start, week_end

    def generate_weekly_report(self, end_date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive weekly summary

        Args:
            end_date: End date for the report week (defaults to current week)

        Returns:
            Dictionary with all weekly statistics
        """
        week_start, week_end = self._get_week_bounds(end_date)

        report = {
            'report_type': 'weekly_summary',
            'generated_at': datetime.now().isoformat(),
            'week_start': week_start.isoformat(),
            'week_end': week_end.isoformat(),
            'week_number': week_start.isocalendar()[1],
            'year': week_start.year,

            # Aggregate sections
            'bark_stats': self._get_bark_stats(week_start, week_end),
            'reward_stats': self._get_reward_stats(week_start, week_end),
            'silent_guardian': self._get_sg_stats(week_start, week_end),
            'coaching': self._get_coaching_stats(week_start, week_end),
            'missions': self._get_mission_stats(week_start, week_end),
            'dog_summary': self._get_dog_summary(week_start, week_end),
            'daily_breakdown': self._get_daily_breakdown(week_start, week_end),
        }

        # Add highlights/insights
        report['highlights'] = self._generate_highlights(report)

        logger.info(f"Generated weekly report for week {report['week_number']}, {report['year']}")
        return report

    def _get_bark_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get bark statistics for the week (wimz.db — Phase 3 cutover;
        also fixes the legacy UTC-text-vs-local-cutoff skew)"""
        conn = self._wimz_connection()
        try:
            cursor = conn.cursor()
            ms = (self._ms(start), self._ms(end))

            cursor.execute(f'''
                SELECT COUNT(*) as total,
                       AVG(json_extract(payload,'$.db')) as avg_loudness,
                       MAX(json_extract(payload,'$.db')) as max_loudness,
                       AVG(confidence) as avg_confidence
                FROM event e WHERE {self._BARK_WHERE}
            ''', ms)
            totals = dict(cursor.fetchone() or {})

            cursor.execute(f'''
                SELECT COALESCE(json_extract(payload,'$.emotion'),'unknown') as emotion,
                       COUNT(*) as count
                FROM event e WHERE {self._BARK_WHERE}
                GROUP BY emotion ORDER BY count DESC
            ''', ms)
            by_emotion = {row['emotion']: row['count'] for row in cursor.fetchall()}

            cursor.execute(f'''
                SELECT COALESCE(json_extract(payload,'$.bark_type'),'unclassified') as btype,
                       COUNT(*) as count
                FROM event e WHERE {self._BARK_WHERE}
                GROUP BY btype ORDER BY count DESC
            ''', ms)
            by_bark_type = {row['btype']: row['count'] for row in cursor.fetchall()}

            cursor.execute(f'''
                SELECT COALESCE(d.name, d.app_dog_id, 'unknown') as dog, COUNT(*) as count
                FROM event e LEFT JOIN dog d ON e.dog_id = d.dog_id
                WHERE {self._BARK_WHERE}
                GROUP BY dog ORDER BY count DESC
            ''', ms)
            by_dog = {row['dog']: row['count'] for row in cursor.fetchall()}

            cursor.execute(f'''
                SELECT strftime('%w', datetime(ts/1000.0,'unixepoch','localtime')) as day_num,
                       COUNT(*) as count
                FROM event e WHERE {self._BARK_WHERE}
                GROUP BY day_num
            ''', ms)
            by_day_raw = {int(row['day_num']): row['count'] for row in cursor.fetchall()}
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            by_day = {day_names[i]: by_day_raw.get(i, 0) for i in range(7)}

            return {
                'total_barks': totals.get('total', 0) or 0,
                'avg_loudness_db': round(totals.get('avg_loudness', 0) or 0, 1),
                'max_loudness_db': round(totals.get('max_loudness', 0) or 0, 1),
                'avg_confidence': round(totals.get('avg_confidence', 0) or 0, 2),
                'by_emotion': by_emotion,
                'by_bark_type': by_bark_type,
                'by_dog': by_dog,
                'by_day': by_day
            }
        finally:
            conn.close()

    def _get_reward_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get reward/treat statistics for the week"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Total rewards
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(treats_dispensed) as total_treats,
                       AVG(confidence) as avg_confidence
                FROM rewards
                WHERE timestamp BETWEEN ? AND ?
            ''', (start.timestamp(), end.timestamp()))
            totals = dict(cursor.fetchone() or {})

            # By behavior
            cursor.execute('''
                SELECT behavior, COUNT(*) as count, SUM(treats_dispensed) as treats
                FROM rewards
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY behavior
                ORDER BY count DESC
            ''', (start.timestamp(), end.timestamp()))
            by_behavior = {row['behavior']: {'count': row['count'], 'treats': row['treats'] or 0}
                          for row in cursor.fetchall()}

            # By dog
            cursor.execute('''
                SELECT COALESCE(dog_id, 'unknown') as dog,
                       COUNT(*) as count,
                       SUM(treats_dispensed) as treats
                FROM rewards
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY dog
                ORDER BY count DESC
            ''', (start.timestamp(), end.timestamp()))
            by_dog = {row['dog']: {'count': row['count'], 'treats': row['treats'] or 0}
                     for row in cursor.fetchall()}

            # Success rate
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                FROM rewards
                WHERE timestamp BETWEEN ? AND ?
            ''', (start.timestamp(), end.timestamp()))
            success = dict(cursor.fetchone() or {})
            total = success.get('total', 0) or 0
            successful = success.get('successful', 0) or 0
            success_rate = (successful / total * 100) if total > 0 else 0

            return {
                'total_rewards': totals.get('total', 0) or 0,
                'total_treats': totals.get('total_treats', 0) or 0,
                'avg_confidence': round(totals.get('avg_confidence', 0) or 0, 2),
                'success_rate': round(success_rate, 1),
                'by_behavior': by_behavior,
                'by_dog': by_dog
            }
        finally:
            conn.close()

    def _get_sg_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get Silent Guardian statistics for the week (wimz.db — Phase 3:
        sessions from session.outcome_json, interventions from
        sg_intervention events)"""
        conn = self._wimz_connection()
        try:
            cursor = conn.cursor()
            ms = (self._ms(start), self._ms(end))

            # Session totals from outcome_json (written at every SG session
            # close, including the 8h rollover). Open sessions count toward
            # `sessions` and duration; their outcome sums land at close.
            cursor.execute('''
                SELECT COUNT(*) as sessions,
                       SUM(json_extract(outcome_json,'$.total_barks')) as total_barks,
                       SUM(json_extract(outcome_json,'$.interventions_triggered')) as total_interventions,
                       SUM(json_extract(outcome_json,'$.successful_quiets')) as total_quiets,
                       SUM(json_extract(outcome_json,'$.treats_dispensed')) as total_treats,
                       MAX(json_extract(outcome_json,'$.max_escalation_level')) as max_escalation,
                       SUM(COALESCE(ended_at, ?) - started_at) / 1000.0 as total_duration
                FROM session
                WHERE mode = 'sg' AND started_at BETWEEN ? AND ?
            ''', (self._ms(end), *ms))
            totals = dict(cursor.fetchone() or {})

            # Calculate effectiveness
            interventions = totals.get('total_interventions', 0) or 0
            quiets = totals.get('total_quiets', 0) or 0
            effectiveness = (quiets / interventions * 100) if interventions > 0 else 0

            # Intervention outcomes by escalation level
            cursor.execute('''
                SELECT json_extract(payload,'$.escalation_level') as escalation_level,
                       COUNT(*) as count,
                       SUM(CASE WHEN json_extract(payload,'$.quiet_achieved') THEN 1 ELSE 0 END) as successful
                FROM event
                WHERE event_type = 'sg_intervention'
                  AND json_extract(payload,'$.phase') = 'outcome'
                  AND ts BETWEEN ? AND ?
                GROUP BY escalation_level
            ''', ms)
            by_escalation = {}
            for row in cursor.fetchall():
                level = row['escalation_level']
                by_escalation[f'level_{level}'] = {
                    'count': row['count'],
                    'avg_quiet_duration': 0,  # not tracked per-intervention in the spec store
                    'success_rate': round((row['successful'] / row['count'] * 100) if row['count'] > 0 else 0, 1)
                }

            return {
                'total_sessions': totals.get('sessions', 0) or 0,
                'total_barks': totals.get('total_barks', 0) or 0,
                'total_interventions': interventions,
                'successful_quiets': quiets,
                'effectiveness_rate': round(effectiveness, 1),
                'treats_dispensed': totals.get('total_treats', 0) or 0,
                'max_escalation_reached': totals.get('max_escalation', 0) or 0,
                'total_duration_hours': round((totals.get('total_duration', 0) or 0) / 3600, 1),
                'by_escalation_level': by_escalation
            }
        finally:
            conn.close()

    def _get_coaching_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get coaching statistics for the week (wimz.db training_attempt —
        Phase 3 cutover; coach/training sessions only, so SG quiet-attempts
        are excluded)"""
        conn = self._wimz_connection()
        try:
            cursor = conn.cursor()
            ms = (self._ms(start), self._ms(end))
            base = ('''FROM training_attempt ta
                       JOIN session s ON ta.session_id = s.session_id
                       WHERE s.mode IN ('coach','training')
                         AND ta.cue_ts BETWEEN ? AND ?''')

            cursor.execute(f'''
                SELECT COUNT(*) as total_sessions,
                       SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END) as completed,
                       SUM(CASE WHEN ta.reward_dispensed >= 1 THEN 1 ELSE 0 END) as treats,
                       AVG(ta.latency_ms) / 1000.0 as avg_response_time
                {base}
            ''', ms)
            totals = dict(cursor.fetchone() or {})

            total = totals.get('total_sessions', 0) or 0
            completed = totals.get('completed', 0) or 0
            success_rate = (completed / total * 100) if total > 0 else 0

            # By trick
            cursor.execute(f'''
                SELECT ta.trick_label as trick,
                       COUNT(*) as attempts,
                       SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END) as completed
                {base}
                GROUP BY ta.trick_label
                ORDER BY attempts DESC
            ''', ms)
            by_trick = {}
            for row in cursor.fetchall():
                attempts = row['attempts']
                completed = row['completed'] or 0
                by_trick[row['trick']] = {
                    'attempts': attempts,
                    'completed': completed,
                    'success_rate': round((completed / attempts * 100) if attempts > 0 else 0, 1)
                }

            # By dog
            cursor.execute(f'''
                SELECT COALESCE(d.name, d.app_dog_id, 'unknown') as dog,
                       COUNT(*) as sessions,
                       SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END) as completed
                {base.replace('FROM training_attempt ta',
                              'FROM training_attempt ta LEFT JOIN dog d ON ta.dog_id = d.dog_id')}
                GROUP BY dog
            ''', ms)
            by_dog = {}
            for row in cursor.fetchall():
                sessions = row['sessions']
                completed = row['completed'] or 0
                by_dog[row['dog']] = {
                    'sessions': sessions,
                    'completed': completed,
                    'success_rate': round((completed / sessions * 100) if sessions > 0 else 0, 1)
                }

            return {
                'total_sessions': total,
                'tricks_completed': totals.get('completed', 0) or 0,
                'success_rate': round(success_rate, 1),
                'treats_given': totals.get('treats', 0) or 0,
                'avg_response_time': round(totals.get('avg_response_time', 0) or 0, 1),
                'avg_attention_duration': 0,  # not tracked in the spec store
                'by_trick': by_trick,
                'by_dog': by_dog
            }
        finally:
            conn.close()

    def _get_mission_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get mission/program statistics for the week"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Mission totals
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                       SUM(CASE WHEN status = 'stopped' THEN 1 ELSE 0 END) as stopped,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                       SUM(rewards_given) as total_rewards,
                       AVG(end_time - start_time) as avg_duration
                FROM missions
                WHERE start_time BETWEEN ? AND ?
            ''', (start.timestamp(), end.timestamp()))
            totals = dict(cursor.fetchone() or {})

            total = totals.get('total', 0) or 0
            completed = totals.get('completed', 0) or 0
            success_rate = (completed / total * 100) if total > 0 else 0

            # By mission name
            cursor.execute('''
                SELECT name,
                       COUNT(*) as attempts,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                       SUM(rewards_given) as rewards,
                       AVG(end_time - start_time) as avg_duration
                FROM missions
                WHERE start_time BETWEEN ? AND ?
                GROUP BY name
                ORDER BY attempts DESC
            ''', (start.timestamp(), end.timestamp()))
            by_mission = {}
            for row in cursor.fetchall():
                attempts = row['attempts']
                mission_completed = row['completed'] or 0
                by_mission[row['name']] = {
                    'attempts': attempts,
                    'completed': mission_completed,
                    'success_rate': round((mission_completed / attempts * 100) if attempts > 0 else 0, 1),
                    'rewards': row['rewards'] or 0,
                    'avg_duration_sec': round(row['avg_duration'] or 0, 1)
                }

            # By day of week
            cursor.execute('''
                SELECT strftime('%w', datetime(start_time, 'unixepoch')) as day_num,
                       COUNT(*) as count
                FROM missions
                WHERE start_time BETWEEN ? AND ?
                GROUP BY day_num
            ''', (start.timestamp(), end.timestamp()))
            by_day_raw = {int(row['day_num']): row['count'] for row in cursor.fetchall()}
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            by_day = {day_names[i]: by_day_raw.get(i, 0) for i in range(7)}

            return {
                'total_missions': total,
                'completed': completed,
                'stopped': totals.get('stopped', 0) or 0,
                'failed': totals.get('failed', 0) or 0,
                'success_rate': round(success_rate, 1),
                'total_rewards': totals.get('total_rewards', 0) or 0,
                'avg_duration_sec': round(totals.get('avg_duration', 0) or 0, 1),
                'by_mission': by_mission,
                'by_day': by_day
            }
        finally:
            conn.close()

    def _wimz_dogs(self) -> List[Dict[str, Any]]:
        """Known dogs from the spec store (the legacy `dogs` table holds only
        test fixtures — real identity lives in wimz.db, Phase 3)."""
        conn = self._wimz_connection()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT dog_id, name, app_dog_id FROM dog "
                "WHERE name IS NOT NULL OR app_dog_id IS NOT NULL")]
        finally:
            conn.close()

    def _get_dog_summary(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get per-dog summary combining all metrics (wimz.db for identity,
        barks, coaching; legacy rewards until its writer cuts over)"""
        summaries = {}
        legacy = self._get_connection()
        try:
            for dog in self._wimz_dogs():
                display_name = dog['name'] or dog['app_dog_id']

                barks = self._wimz_bark_count(start, end,
                                              dog_key=dog['dog_id'],
                                              dog_name=dog['name'])

                # Rewards (legacy table keys on the app canonical id or name)
                row = legacy.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE (dog_id = ? OR dog_id = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog['app_dog_id'], dog['name'],
                      start.timestamp(), end.timestamp())).fetchone()
                rewards = dict(row or {})

                coaching = self._wimz_coaching_agg(start, end,
                                                   dog_key=dog['dog_id'],
                                                   dog_name=dog['name'])

                summaries[display_name] = {
                    'dog_id': dog['app_dog_id'] or dog['dog_id'],
                    'barks': barks,
                    'rewards': rewards.get('count', 0) or 0,
                    'treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': coaching.get('sessions', 0) or 0,
                    'tricks_completed': coaching.get('completed', 0) or 0
                }

            return summaries
        finally:
            legacy.close()

    def _get_daily_breakdown(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get day-by-day breakdown for the week"""
        breakdown = {}
        current = start

        while current <= end:
            day_start = current.replace(hour=0, minute=0, second=0)
            day_end = current.replace(hour=23, minute=59, second=59)
            day_name = current.strftime('%A')

            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Barks for this day (wimz.db)
                barks = self._wimz_bark_count(day_start, day_end)

                # Rewards for this day
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE timestamp BETWEEN ? AND ?
                ''', (day_start.timestamp(), day_end.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                breakdown[day_name] = {
                    'date': current.strftime('%Y-%m-%d'),
                    'barks': barks,
                    'rewards': rewards.get('count', 0) or 0,
                    'treats': rewards.get('treats', 0) or 0
                }
            finally:
                conn.close()

            current += timedelta(days=1)

        return breakdown

    def _generate_highlights(self, report: Dict) -> List[str]:
        """Generate insight highlights from report data"""
        highlights = []

        # Bark insights
        bark_stats = report.get('bark_stats', {})
        total_barks = bark_stats.get('total_barks', 0)
        if total_barks > 0:
            top_emotion = max(bark_stats.get('by_emotion', {}).items(),
                            key=lambda x: x[1], default=(None, 0))
            if top_emotion[0]:
                highlights.append(f"Most common bark emotion: {top_emotion[0]} ({top_emotion[1]} times)")

            top_barker = max(bark_stats.get('by_dog', {}).items(),
                           key=lambda x: x[1], default=(None, 0))
            if top_barker[0]:
                highlights.append(f"Most active barker: {top_barker[0]} ({top_barker[1]} barks)")

        # Reward insights
        reward_stats = report.get('reward_stats', {})
        total_treats = reward_stats.get('total_treats', 0)
        if total_treats > 0:
            highlights.append(f"Total treats earned: {total_treats}")

        # Coaching insights
        coaching = report.get('coaching', {})
        success_rate = coaching.get('success_rate', 0)
        if coaching.get('total_sessions', 0) > 0:
            highlights.append(f"Coaching success rate: {success_rate}%")

        # Silent Guardian insights
        sg = report.get('silent_guardian', {})
        if sg.get('total_sessions', 0) > 0:
            effectiveness = sg.get('effectiveness_rate', 0)
            highlights.append(f"Silent Guardian effectiveness: {effectiveness}%")

        return highlights

    def get_behavior_trends(self, weeks: int = 8) -> Dict[str, Any]:
        """
        Analyze behavior trends over multiple weeks

        Args:
            weeks: Number of weeks to analyze (default 8)

        Returns:
            Dictionary with week-over-week trend data
        """
        trends = {
            'weeks_analyzed': weeks,
            'generated_at': datetime.now().isoformat(),
            'weekly_data': [],
            'trends': {}
        }

        # Get data for each week
        end_date = datetime.now()
        for i in range(weeks):
            week_end = end_date - timedelta(weeks=i)
            week_start, week_end_bound = self._get_week_bounds(week_end)

            # Get simple stats for this week
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Barks (wimz.db)
                barks = self._wimz_bark_count(week_start, week_end_bound)

                # Rewards (legacy until its writer cuts over)
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE timestamp BETWEEN ? AND ?
                ''', (week_start.timestamp(), week_end_bound.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching (wimz.db)
                coaching = self._wimz_coaching_agg(week_start, week_end_bound)

                week_data = {
                    'week_start': week_start.isoformat(),
                    'week_number': week_start.isocalendar()[1],
                    'barks': barks,
                    'rewards': rewards.get('count', 0) or 0,
                    'treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': coaching.get('sessions', 0) or 0,
                    'tricks_completed': coaching.get('completed', 0) or 0
                }
                trends['weekly_data'].append(week_data)

            finally:
                conn.close()

        # Reverse so oldest is first
        trends['weekly_data'].reverse()

        # Calculate trends (compare current to average of previous weeks)
        if len(trends['weekly_data']) >= 2:
            current = trends['weekly_data'][-1]
            previous = trends['weekly_data'][:-1]

            for metric in ['barks', 'rewards', 'treats', 'coaching_sessions', 'tricks_completed']:
                current_val = current.get(metric, 0)
                prev_avg = sum(w.get(metric, 0) for w in previous) / len(previous) if previous else 0

                if prev_avg > 0:
                    change_pct = ((current_val - prev_avg) / prev_avg) * 100
                    direction = 'up' if change_pct > 5 else 'down' if change_pct < -5 else 'stable'
                else:
                    change_pct = 0
                    direction = 'stable' if current_val == 0 else 'up'

                trends['trends'][metric] = {
                    'current': current_val,
                    'previous_avg': round(prev_avg, 1),
                    'change_percent': round(change_pct, 1),
                    'direction': direction
                }

        return trends

    def get_dog_progress(self, dog_id: str, weeks: int = 8) -> Dict[str, Any]:
        """
        Get individual dog progress report

        Args:
            dog_id: Dog ID or name
            weeks: Number of weeks to analyze

        Returns:
            Dictionary with per-dog progress data
        """
        progress = {
            'dog_id': dog_id,
            'weeks_analyzed': weeks,
            'generated_at': datetime.now().isoformat(),
            'weekly_data': []
        }

        end_date = datetime.now()
        for i in range(weeks):
            week_end = end_date - timedelta(weeks=i)
            week_start, week_end_bound = self._get_week_bounds(week_end)

            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Barks for this dog (wimz.db; dog_id may be app UUID or name)
                barks = self._wimz_bark_count(week_start, week_end_bound,
                                              dog_key=dog_id, dog_name=dog_id)

                # Rewards for this dog (legacy until its writer cuts over)
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE dog_id = ? AND timestamp BETWEEN ? AND ?
                ''', (dog_id, week_start.timestamp(), week_end_bound.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching for this dog (wimz.db)
                coaching = self._wimz_coaching_agg(week_start, week_end_bound,
                                                   dog_key=dog_id, dog_name=dog_id)

                sessions = coaching.get('sessions', 0) or 0
                completed = coaching.get('completed', 0) or 0

                week_data = {
                    'week_start': week_start.isoformat(),
                    'week_number': week_start.isocalendar()[1],
                    'barks': barks,
                    'rewards': rewards.get('count', 0) or 0,
                    'treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': sessions,
                    'tricks_completed': completed,
                    'coaching_success_rate': round((completed / sessions * 100) if sessions > 0 else 0, 1)
                }
                progress['weekly_data'].append(week_data)

            finally:
                conn.close()

        # Reverse so oldest is first
        progress['weekly_data'].reverse()

        return progress

    def compare_dogs(self) -> Dict[str, Any]:
        """
        Cross-dog comparison analysis

        Returns:
            Dictionary with comparative metrics for all dogs
        """
        # Use last 4 weeks for comparison
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=4)

        comparison = {
            'period': '4 weeks',
            'generated_at': datetime.now().isoformat(),
            'dogs': {}
        }

        legacy = self._get_connection()
        wimz = self._wimz_connection()
        try:
            for dog in self._wimz_dogs():
                display_name = dog['name'] or dog['app_dog_id']

                # Barks (wimz.db)
                barks = dict(wimz.execute(f'''
                    SELECT COUNT(*) as count,
                           AVG(json_extract(payload,'$.db')) as avg_loud
                    FROM event e WHERE {self._BARK_WHERE} AND e.dog_id = ?
                ''', (self._ms(start_date), self._ms(end_date),
                      dog['dog_id'])).fetchone() or {})

                # Rewards (legacy table keys on the app canonical id or name)
                rewards = dict(legacy.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE (dog_id = ? OR dog_id = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog['app_dog_id'], dog['name'],
                      start_date.timestamp(), end_date.timestamp())).fetchone() or {})

                # Coaching (wimz.db)
                coaching = self._wimz_coaching_agg(start_date, end_date,
                                                   dog_key=dog['dog_id'],
                                                   dog_name=dog['name'])

                sessions = coaching.get('sessions', 0) or 0
                completed = coaching.get('completed', 0) or 0

                comparison['dogs'][display_name] = {
                    'dog_id': dog['app_dog_id'] or dog['dog_id'],
                    'total_barks': barks.get('count', 0) or 0,
                    'avg_bark_loudness': round(barks.get('avg_loud', 0) or 0, 1),
                    'total_rewards': rewards.get('count', 0) or 0,
                    'total_treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': sessions,
                    'tricks_completed': completed,
                    'coaching_success_rate': round((completed / sessions * 100) if sessions > 0 else 0, 1),
                    'avg_response_time': round((coaching.get('avg_latency_ms', 0) or 0) / 1000.0, 1)
                }

            return comparison
        finally:
            legacy.close()
            wimz.close()

    def export_report(self, report: Dict, format: str = 'markdown') -> str:
        """
        Export report to file

        Args:
            report: Report dictionary to export
            format: 'markdown' or 'csv'

        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        week_num = report.get('week_number', 0)
        year = report.get('year', datetime.now().year)

        if format == 'markdown':
            filename = f"weekly_report_{year}_w{week_num}_{timestamp}.md"
            filepath = self.reports_dir / filename
            content = self._format_markdown(report)
        elif format == 'csv':
            filename = f"weekly_report_{year}_w{week_num}_{timestamp}.csv"
            filepath = self.reports_dir / filename
            content = self._format_csv(report)
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(filepath, 'w') as f:
            f.write(content)

        logger.info(f"Exported report to {filepath}")
        return str(filepath)

    def _format_markdown(self, report: Dict) -> str:
        """Format report as markdown"""
        lines = [
            f"# WIM-Z Weekly Report",
            f"",
            f"**Week {report.get('week_number', 'N/A')}, {report.get('year', 'N/A')}**",
            f"",
            f"Generated: {report.get('generated_at', 'N/A')}",
            f"",
            f"Period: {report.get('week_start', 'N/A')[:10]} to {report.get('week_end', 'N/A')[:10]}",
            f"",
            f"---",
            f"",
            f"## Highlights",
            f""
        ]

        for highlight in report.get('highlights', []):
            lines.append(f"- {highlight}")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Bark Statistics",
            f"",
            f"- **Total Barks:** {report['bark_stats'].get('total_barks', 0)}",
            f"- **Average Loudness:** {report['bark_stats'].get('avg_loudness_db', 0)} dB",
            f"- **Max Loudness:** {report['bark_stats'].get('max_loudness_db', 0)} dB",
            f"",
            f"### By Emotion",
            f""
        ])

        for emotion, count in report['bark_stats'].get('by_emotion', {}).items():
            lines.append(f"- {emotion}: {count}")

        lines.extend([
            f"",
            f"### By Dog",
            f""
        ])

        for dog, count in report['bark_stats'].get('by_dog', {}).items():
            lines.append(f"- {dog}: {count}")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Reward Statistics",
            f"",
            f"- **Total Rewards:** {report['reward_stats'].get('total_rewards', 0)}",
            f"- **Total Treats:** {report['reward_stats'].get('total_treats', 0)}",
            f"- **Success Rate:** {report['reward_stats'].get('success_rate', 0)}%",
            f"",
            f"### By Behavior",
            f""
        ])

        for behavior, data in report['reward_stats'].get('by_behavior', {}).items():
            lines.append(f"- {behavior}: {data.get('count', 0)} rewards, {data.get('treats', 0)} treats")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Silent Guardian",
            f"",
            f"- **Total Sessions:** {report['silent_guardian'].get('total_sessions', 0)}",
            f"- **Total Duration:** {report['silent_guardian'].get('total_duration_hours', 0)} hours",
            f"- **Interventions:** {report['silent_guardian'].get('total_interventions', 0)}",
            f"- **Successful Quiets:** {report['silent_guardian'].get('successful_quiets', 0)}",
            f"- **Effectiveness:** {report['silent_guardian'].get('effectiveness_rate', 0)}%",
            f"- **Treats Given:** {report['silent_guardian'].get('treats_dispensed', 0)}",
            f"",
            f"---",
            f"",
            f"## Coaching Sessions",
            f"",
            f"- **Total Sessions:** {report['coaching'].get('total_sessions', 0)}",
            f"- **Tricks Completed:** {report['coaching'].get('tricks_completed', 0)}",
            f"- **Success Rate:** {report['coaching'].get('success_rate', 0)}%",
            f"- **Avg Response Time:** {report['coaching'].get('avg_response_time', 0)}s",
            f"",
            f"### By Trick",
            f""
        ])

        for trick, data in report['coaching'].get('by_trick', {}).items():
            lines.append(f"- {trick}: {data.get('attempts', 0)} attempts, {data.get('success_rate', 0)}% success")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Dog Summary",
            f""
        ])

        for dog, data in report.get('dog_summary', {}).items():
            lines.extend([
                f"### {dog}",
                f"- Barks: {data.get('barks', 0)}",
                f"- Rewards: {data.get('rewards', 0)}",
                f"- Treats: {data.get('treats', 0)}",
                f"- Coaching: {data.get('coaching_sessions', 0)} sessions, {data.get('tricks_completed', 0)} completed",
                f""
            ])

        lines.extend([
            f"---",
            f"",
            f"*Report generated by WIM-Z Weekly Summary*"
        ])

        return '\n'.join(lines)

    def _format_csv(self, report: Dict) -> str:
        """Format report as CSV (daily breakdown)"""
        lines = [
            "day,date,barks,rewards,treats"
        ]

        for day, data in report.get('daily_breakdown', {}).items():
            lines.append(f"{day},{data.get('date', '')},{data.get('barks', 0)},{data.get('rewards', 0)},{data.get('treats', 0)}")

        return '\n'.join(lines)


# Singleton instance
_weekly_summary_instance = None


def get_weekly_summary() -> WeeklySummary:
    """Get or create WeeklySummary instance (singleton)"""
    global _weekly_summary_instance
    if _weekly_summary_instance is None:
        _weekly_summary_instance = WeeklySummary()
    return _weekly_summary_instance


# Test function
def main():
    """Test weekly summary generation"""
    import pprint

    logging.basicConfig(level=logging.INFO)

    summary = WeeklySummary()

    print("\n=== WEEKLY REPORT ===\n")
    report = summary.generate_weekly_report()
    pprint.pprint(report)

    print("\n=== 8-WEEK TRENDS ===\n")
    trends = summary.get_behavior_trends(weeks=8)
    pprint.pprint(trends)

    print("\n=== DOG COMPARISON ===\n")
    comparison = summary.compare_dogs()
    pprint.pprint(comparison)

    # Export markdown
    filepath = summary.export_report(report, format='markdown')
    print(f"\n=== EXPORTED TO: {filepath} ===\n")


if __name__ == "__main__":
    main()
