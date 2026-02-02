#!/usr/bin/env python3
"""
AWS SNS Push Notification Service for WIM-Z

Sends SMS and push notifications to users for important events:
- Mission completed
- Bark alerts (when away)
- Low battery warnings
- Weekly summary ready

Setup:
1. Install boto3: pip install boto3
2. Configure AWS credentials:
   - Option A: aws configure (creates ~/.aws/credentials)
   - Option B: Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
3. For SMS: Ensure your AWS account is out of SMS sandbox OR verify phone numbers

Usage:
    from services.cloud.notification_service import get_notification_service
    notifier = get_notification_service()
    await notifier.send_sms("+15551234567", "Your dog completed sit training!")
"""

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

# Lazy import boto3 to avoid startup errors if not installed
boto3 = None

SUBSCRIBERS_FILE = "/home/morgan/dogbot/data/notification_subscribers.json"


class NotificationService:
    """
    AWS SNS notification service for SMS and push notifications
    """

    def __init__(self):
        self.logger = logging.getLogger('NotificationService')
        self._sns_client = None
        self._initialized = False
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._load_subscribers()

    def _ensure_boto3(self) -> bool:
        """Lazy load boto3"""
        global boto3
        if boto3 is None:
            try:
                import boto3 as _boto3
                boto3 = _boto3
                return True
            except ImportError:
                self.logger.error("boto3 not installed. Run: pip install boto3")
                return False
        return True

    def _get_client(self):
        """Get or create SNS client"""
        if self._sns_client is None:
            if not self._ensure_boto3():
                return None
            try:
                # Default to us-east-1 for SMS (best coverage)
                region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                self._sns_client = boto3.client('sns', region_name=region)
                self._initialized = True
                self.logger.info(f"SNS client initialized in region {region}")
            except Exception as e:
                self.logger.error(f"Failed to initialize SNS client: {e}")
                return None
        return self._sns_client

    def _load_subscribers(self):
        """Load subscriber list from disk"""
        try:
            if os.path.exists(SUBSCRIBERS_FILE):
                with open(SUBSCRIBERS_FILE, 'r') as f:
                    self._subscribers = json.load(f)
                self.logger.info(f"Loaded {len(self._subscribers)} notification subscribers")
        except Exception as e:
            self.logger.error(f"Failed to load subscribers: {e}")
            self._subscribers = {}

    def _save_subscribers(self):
        """Save subscriber list to disk"""
        try:
            Path(SUBSCRIBERS_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(SUBSCRIBERS_FILE, 'w') as f:
                json.dump(self._subscribers, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save subscribers: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Subscriber Management
    # ─────────────────────────────────────────────────────────────────

    def add_subscriber(self, user_id: str, phone_number: str,
                       notification_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Add or update a notification subscriber

        Args:
            user_id: Unique user identifier
            phone_number: Phone number in E.164 format (+15551234567)
            notification_types: List of notification types to receive
                               Options: mission_complete, bark_alert, low_battery, weekly_summary
                               Default: all types

        Returns:
            Dict with status and subscriber info
        """
        if not phone_number.startswith('+'):
            phone_number = '+1' + phone_number.replace('-', '').replace(' ', '')

        if notification_types is None:
            notification_types = ['mission_complete', 'bark_alert', 'low_battery', 'weekly_summary']

        self._subscribers[user_id] = {
            'phone_number': phone_number,
            'notification_types': notification_types,
            'enabled': True
        }
        self._save_subscribers()

        self.logger.info(f"Added subscriber {user_id}: {phone_number}")
        return {
            'status': 'success',
            'user_id': user_id,
            'phone_number': phone_number,
            'notification_types': notification_types
        }

    def remove_subscriber(self, user_id: str) -> Dict[str, Any]:
        """Remove a subscriber"""
        if user_id in self._subscribers:
            del self._subscribers[user_id]
            self._save_subscribers()
            return {'status': 'success', 'message': f'Removed subscriber {user_id}'}
        return {'status': 'error', 'message': f'Subscriber {user_id} not found'}

    def get_subscriber(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get subscriber info"""
        return self._subscribers.get(user_id)

    def list_subscribers(self) -> List[Dict[str, Any]]:
        """List all subscribers"""
        return [
            {'user_id': uid, **info}
            for uid, info in self._subscribers.items()
        ]

    def update_subscriber_preferences(self, user_id: str,
                                       notification_types: Optional[List[str]] = None,
                                       enabled: Optional[bool] = None) -> Dict[str, Any]:
        """Update subscriber notification preferences"""
        if user_id not in self._subscribers:
            return {'status': 'error', 'message': f'Subscriber {user_id} not found'}

        if notification_types is not None:
            self._subscribers[user_id]['notification_types'] = notification_types
        if enabled is not None:
            self._subscribers[user_id]['enabled'] = enabled

        self._save_subscribers()
        return {'status': 'success', 'subscriber': self._subscribers[user_id]}

    # ─────────────────────────────────────────────────────────────────
    # SMS Sending
    # ─────────────────────────────────────────────────────────────────

    async def send_sms(self, phone_number: str, message: str,
                       sender_id: str = "WIMZ") -> Dict[str, Any]:
        """
        Send an SMS message via AWS SNS

        Args:
            phone_number: Destination phone in E.164 format (+15551234567)
            message: Message text (max 160 chars for single SMS)
            sender_id: Sender ID shown on phone (may not work in all countries)

        Returns:
            Dict with status and message_id if successful
        """
        client = self._get_client()
        if client is None:
            return {'status': 'error', 'message': 'SNS client not available'}

        try:
            # Run boto3 call in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.publish(
                    PhoneNumber=phone_number,
                    Message=message,
                    MessageAttributes={
                        'AWS.SNS.SMS.SenderID': {
                            'DataType': 'String',
                            'StringValue': sender_id
                        },
                        'AWS.SNS.SMS.SMSType': {
                            'DataType': 'String',
                            'StringValue': 'Transactional'  # Higher delivery priority
                        }
                    }
                )
            )

            message_id = response.get('MessageId', 'unknown')
            self.logger.info(f"SMS sent to {phone_number}: {message_id}")
            return {
                'status': 'success',
                'message_id': message_id,
                'phone_number': phone_number
            }

        except Exception as e:
            self.logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'phone_number': phone_number
            }

    # ─────────────────────────────────────────────────────────────────
    # Event-based Notifications
    # ─────────────────────────────────────────────────────────────────

    async def notify_mission_complete(self, dog_name: str, mission_name: str,
                                       success: bool, stats: Optional[Dict] = None):
        """Send notification when a mission completes"""
        status = "completed successfully" if success else "ended"
        message = f"🐕 {dog_name} {status} {mission_name}!"

        if stats:
            if 'treats_given' in stats:
                message += f" ({stats['treats_given']} treats)"

        await self._broadcast('mission_complete', message)

    async def notify_bark_alert(self, dog_name: str, bark_count: int,
                                 duration_minutes: int):
        """Send notification for excessive barking"""
        message = f"🔔 {dog_name} has barked {bark_count} times in the last {duration_minutes} minutes"
        await self._broadcast('bark_alert', message)

    async def notify_low_battery(self, battery_percent: int):
        """Send notification for low battery"""
        message = f"🔋 WIM-Z battery is low ({battery_percent}%). Please charge soon."
        await self._broadcast('low_battery', message)

    async def notify_weekly_summary(self, summary: str):
        """Send weekly summary notification"""
        message = f"📊 Weekly WIM-Z Report Ready!\n{summary}"
        await self._broadcast('weekly_summary', message)

    async def _broadcast(self, notification_type: str, message: str):
        """Send notification to all subscribers of a given type"""
        results = []
        for user_id, info in self._subscribers.items():
            if not info.get('enabled', True):
                continue
            if notification_type not in info.get('notification_types', []):
                continue

            result = await self.send_sms(info['phone_number'], message)
            results.append({
                'user_id': user_id,
                **result
            })

        success_count = sum(1 for r in results if r.get('status') == 'success')
        self.logger.info(f"Broadcast '{notification_type}': {success_count}/{len(results)} delivered")
        return results

    # ─────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Check if notification service is healthy"""
        client = self._get_client()
        if client is None:
            return {
                'status': 'unhealthy',
                'message': 'SNS client not available (boto3 not installed or AWS credentials missing)'
            }

        try:
            # Try to get SMS attributes to verify credentials
            client.get_sms_attributes()
            return {
                'status': 'healthy',
                'subscribers': len(self._subscribers),
                'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': str(e)
            }


# Singleton instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get the singleton notification service instance"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
