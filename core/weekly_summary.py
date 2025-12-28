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

    Aggregates data from:
    - barks table (bark detection)
    - rewards table (treat dispensing)
    - silent_guardian_sessions (SG mode)
    - sg_interventions (bark interventions)
    - coaching_sessions (trick training)
    - dogs table (dog profiles)
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
            'dog_summary': self._get_dog_summary(week_start, week_end),
            'daily_breakdown': self._get_daily_breakdown(week_start, week_end),
        }

        # Add highlights/insights
        report['highlights'] = self._generate_highlights(report)

        logger.info(f"Generated weekly report for week {report['week_number']}, {report['year']}")
        return report

    def _get_bark_stats(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get bark statistics for the week"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Total barks
            cursor.execute('''
                SELECT COUNT(*) as total,
                       AVG(loudness_db) as avg_loudness,
                       MAX(loudness_db) as max_loudness,
                       AVG(confidence) as avg_confidence
                FROM barks
                WHERE timestamp BETWEEN ? AND ?
            ''', (start.isoformat(), end.isoformat()))
            totals = dict(cursor.fetchone() or {})

            # By emotion
            cursor.execute('''
                SELECT emotion, COUNT(*) as count
                FROM barks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY emotion
                ORDER BY count DESC
            ''', (start.isoformat(), end.isoformat()))
            by_emotion = {row['emotion']: row['count'] for row in cursor.fetchall()}

            # By dog
            cursor.execute('''
                SELECT COALESCE(dog_name, dog_id, 'unknown') as dog, COUNT(*) as count
                FROM barks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY COALESCE(dog_name, dog_id, 'unknown')
                ORDER BY count DESC
            ''', (start.isoformat(), end.isoformat()))
            by_dog = {row['dog']: row['count'] for row in cursor.fetchall()}

            # By day of week
            cursor.execute('''
                SELECT strftime('%w', timestamp) as day_num, COUNT(*) as count
                FROM barks
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY day_num
            ''', (start.isoformat(), end.isoformat()))
            by_day_raw = {int(row['day_num']): row['count'] for row in cursor.fetchall()}

            # Convert to day names
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            by_day = {day_names[i]: by_day_raw.get(i, 0) for i in range(7)}

            return {
                'total_barks': totals.get('total', 0) or 0,
                'avg_loudness_db': round(totals.get('avg_loudness', 0) or 0, 1),
                'max_loudness_db': round(totals.get('max_loudness', 0) or 0, 1),
                'avg_confidence': round(totals.get('avg_confidence', 0) or 0, 2),
                'by_emotion': by_emotion,
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
        """Get Silent Guardian statistics for the week"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Session totals
            cursor.execute('''
                SELECT COUNT(*) as sessions,
                       SUM(total_barks) as total_barks,
                       SUM(interventions) as total_interventions,
                       SUM(successful_quiets) as total_quiets,
                       SUM(treats_dispensed) as total_treats,
                       MAX(max_escalation_level) as max_escalation,
                       SUM(COALESCE(session_end, ?) - session_start) as total_duration
                FROM silent_guardian_sessions
                WHERE session_start BETWEEN ? AND ?
            ''', (end.timestamp(), start.timestamp(), end.timestamp()))
            totals = dict(cursor.fetchone() or {})

            # Calculate effectiveness
            interventions = totals.get('total_interventions', 0) or 0
            quiets = totals.get('total_quiets', 0) or 0
            effectiveness = (quiets / interventions * 100) if interventions > 0 else 0

            # Intervention details
            cursor.execute('''
                SELECT escalation_level, COUNT(*) as count,
                       AVG(quiet_duration) as avg_quiet,
                       SUM(CASE WHEN quiet_achieved = 1 THEN 1 ELSE 0 END) as successful
                FROM sg_interventions
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY escalation_level
            ''', (start.timestamp(), end.timestamp()))
            by_escalation = {}
            for row in cursor.fetchall():
                level = row['escalation_level']
                by_escalation[f'level_{level}'] = {
                    'count': row['count'],
                    'avg_quiet_duration': round(row['avg_quiet'] or 0, 1),
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
        """Get coaching session statistics for the week"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Session totals
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed,
                       SUM(CASE WHEN treat_dispensed = 1 THEN 1 ELSE 0 END) as treats,
                       AVG(response_time) as avg_response_time,
                       AVG(attention_duration) as avg_attention
                FROM coaching_sessions
                WHERE timestamp BETWEEN ? AND ?
            ''', (start.timestamp(), end.timestamp()))
            totals = dict(cursor.fetchone() or {})

            total = totals.get('total_sessions', 0) or 0
            completed = totals.get('completed', 0) or 0
            success_rate = (completed / total * 100) if total > 0 else 0

            # By trick
            cursor.execute('''
                SELECT trick_requested as trick,
                       COUNT(*) as attempts,
                       SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed
                FROM coaching_sessions
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY trick_requested
                ORDER BY attempts DESC
            ''', (start.timestamp(), end.timestamp()))
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
            cursor.execute('''
                SELECT COALESCE(dog_name, dog_id, 'unknown') as dog,
                       COUNT(*) as sessions,
                       SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed
                FROM coaching_sessions
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY COALESCE(dog_name, dog_id, 'unknown')
            ''', (start.timestamp(), end.timestamp()))
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
                'avg_attention_duration': round(totals.get('avg_attention', 0) or 0, 1),
                'by_trick': by_trick,
                'by_dog': by_dog
            }
        finally:
            conn.close()

    def _get_dog_summary(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get per-dog summary combining all metrics"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get all dogs
            cursor.execute('SELECT id, name FROM dogs')
            dogs = {row['id']: row['name'] for row in cursor.fetchall()}

            summaries = {}
            for dog_id, dog_name in dogs.items():
                display_name = dog_name or dog_id

                # Barks
                cursor.execute('''
                    SELECT COUNT(*) as count FROM barks
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_name, start.isoformat(), end.isoformat()))
                barks = cursor.fetchone()['count'] or 0

                # Rewards
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE dog_id = ? AND timestamp BETWEEN ? AND ?
                ''', (dog_id, start.timestamp(), end.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching
                cursor.execute('''
                    SELECT COUNT(*) as sessions,
                           SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed
                    FROM coaching_sessions
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_name, start.timestamp(), end.timestamp()))
                coaching = dict(cursor.fetchone() or {})

                summaries[display_name] = {
                    'dog_id': dog_id,
                    'barks': barks,
                    'rewards': rewards.get('count', 0) or 0,
                    'treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': coaching.get('sessions', 0) or 0,
                    'tricks_completed': coaching.get('completed', 0) or 0
                }

            return summaries
        finally:
            conn.close()

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

                # Barks for this day
                cursor.execute('''
                    SELECT COUNT(*) as count FROM barks
                    WHERE timestamp BETWEEN ? AND ?
                ''', (day_start.isoformat(), day_end.isoformat()))
                barks = cursor.fetchone()['count'] or 0

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

                # Barks
                cursor.execute('''
                    SELECT COUNT(*) as count FROM barks
                    WHERE timestamp BETWEEN ? AND ?
                ''', (week_start.isoformat(), week_end_bound.isoformat()))
                barks = cursor.fetchone()['count'] or 0

                # Rewards
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE timestamp BETWEEN ? AND ?
                ''', (week_start.timestamp(), week_end_bound.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching
                cursor.execute('''
                    SELECT COUNT(*) as sessions,
                           SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed
                    FROM coaching_sessions
                    WHERE timestamp BETWEEN ? AND ?
                ''', (week_start.timestamp(), week_end_bound.timestamp()))
                coaching = dict(cursor.fetchone() or {})

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

                # Barks for this dog
                cursor.execute('''
                    SELECT COUNT(*) as count FROM barks
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_id, week_start.isoformat(), week_end_bound.isoformat()))
                barks = cursor.fetchone()['count'] or 0

                # Rewards for this dog
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE dog_id = ? AND timestamp BETWEEN ? AND ?
                ''', (dog_id, week_start.timestamp(), week_end_bound.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching for this dog
                cursor.execute('''
                    SELECT COUNT(*) as sessions,
                           SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed
                    FROM coaching_sessions
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_id, week_start.timestamp(), week_end_bound.timestamp()))
                coaching = dict(cursor.fetchone() or {})

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

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get all dogs
            cursor.execute('SELECT id, name FROM dogs')
            dogs = {row['id']: row['name'] for row in cursor.fetchall()}

            comparison = {
                'period': '4 weeks',
                'generated_at': datetime.now().isoformat(),
                'dogs': {}
            }

            for dog_id, dog_name in dogs.items():
                display_name = dog_name or dog_id

                # Barks
                cursor.execute('''
                    SELECT COUNT(*) as count, AVG(loudness_db) as avg_loud FROM barks
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_name, start_date.isoformat(), end_date.isoformat()))
                barks = dict(cursor.fetchone() or {})

                # Rewards
                cursor.execute('''
                    SELECT COUNT(*) as count, SUM(treats_dispensed) as treats FROM rewards
                    WHERE dog_id = ? AND timestamp BETWEEN ? AND ?
                ''', (dog_id, start_date.timestamp(), end_date.timestamp()))
                rewards = dict(cursor.fetchone() or {})

                # Coaching
                cursor.execute('''
                    SELECT COUNT(*) as sessions,
                           SUM(CASE WHEN trick_completed = 1 THEN 1 ELSE 0 END) as completed,
                           AVG(response_time) as avg_response
                    FROM coaching_sessions
                    WHERE (dog_id = ? OR dog_name = ?) AND timestamp BETWEEN ? AND ?
                ''', (dog_id, dog_name, start_date.timestamp(), end_date.timestamp()))
                coaching = dict(cursor.fetchone() or {})

                sessions = coaching.get('sessions', 0) or 0
                completed = coaching.get('completed', 0) or 0

                comparison['dogs'][display_name] = {
                    'dog_id': dog_id,
                    'total_barks': barks.get('count', 0) or 0,
                    'avg_bark_loudness': round(barks.get('avg_loud', 0) or 0, 1),
                    'total_rewards': rewards.get('count', 0) or 0,
                    'total_treats': rewards.get('treats', 0) or 0,
                    'coaching_sessions': sessions,
                    'tricks_completed': completed,
                    'coaching_success_rate': round((completed / sessions * 100) if sessions > 0 else 0, 1),
                    'avg_response_time': round(coaching.get('avg_response', 0) or 0, 1)
                }

            return comparison
        finally:
            conn.close()

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
