#!/usr/bin/env python3
"""
Event publisher for dog detection with ArUco identification
Integrates with the event bus to publish dog-specific events
"""

import time
from typing import Dict, List, Optional
from core.bus import EventBus, VisionEvent, RewardEvent

class DogEventPublisher:
    """Publishes dog-specific events with identification"""

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus

    def publish_dog_detected(self, dog_name: str, dog_id: int, bbox: List[float],
                             confidence: float, behavior: Optional[str] = None):
        """
        Publish a dog detection event with identification

        Args:
            dog_name: Friendly name (Elsa/Bezik)
            dog_id: ArUco marker ID (315/832)
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            behavior: Optional behavior detected (sit/stand/lie)
        """
        event_data = {
            'dog_id': f"aruco_{dog_id}",
            'dog_name': dog_name,
            'bbox': bbox,
            'confidence': confidence,
            'behavior': behavior,
            'timestamp': time.time()
        }

        event = VisionEvent(
            subtype='dog_detected',
            data=event_data,
            source='ai_controller'
        )

        self.bus.publish(event)

    def publish_behavior_detected(self, dog_name: str, dog_id: int, behavior: str,
                                  confidence: float):
        """
        Publish a behavior detection event for a specific dog

        Args:
            dog_name: Friendly name (Elsa/Bezik)
            dog_id: ArUco marker ID (315/832)
            behavior: Detected behavior (sit/stand/lie/cross/spin)
            confidence: Behavior confidence
        """
        event_data = {
            'dog_id': f"aruco_{dog_id}",
            'dog_name': dog_name,
            'behavior': behavior,
            'confidence': confidence,
            'timestamp': time.time()
        }

        event = VisionEvent(
            subtype='behavior_detected',
            data=event_data,
            source='ai_controller'
        )

        self.bus.publish(event)

    def publish_reward_dispensed(self, dog_name: str, dog_id: int, behavior: str,
                                  treat_count: int = 1):
        """
        Publish a reward dispensed event for a specific dog

        Args:
            dog_name: Friendly name (Elsa/Bezik)
            dog_id: ArUco marker ID (315/832)
            behavior: Behavior that earned the reward
            treat_count: Number of treats dispensed
        """
        event_data = {
            'dog_id': f"aruco_{dog_id}",
            'dog_name': dog_name,
            'behavior': behavior,
            'treat_count': treat_count,
            'timestamp': time.time()
        }

        event = RewardEvent(
            subtype='treat_dispensed',
            data=event_data,
            source='reward_logic'
        )

        self.bus.publish(event)