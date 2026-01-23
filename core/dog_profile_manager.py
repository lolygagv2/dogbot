#!/usr/bin/env python3
"""
Dog Profile Manager - Handles dog identification via ARUCO and color matching

Identification Priority:
1. ARUCO marker (100% certain)
2. Color match with single dog (high confidence)
3. Color match with multiple dogs (uncertain - show generic)
4. No match (show "Unknown Dog")
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class IdentificationMethod(Enum):
    """How the dog was identified"""
    ARUCO = "aruco"           # ARUCO marker detected (100% confidence)
    COLOR_UNIQUE = "color"    # Color match, only one dog of that color (high confidence)
    COLOR_AMBIGUOUS = "color_ambiguous"  # Color match, multiple dogs same color (low confidence)
    PERSISTENCE = "persistence"  # Based on tracking persistence
    UNKNOWN = "unknown"       # Could not identify


class DogColor(Enum):
    """Basic dog coat colors for MVP identification"""
    BLACK = "black"
    WHITE = "white"
    BROWN = "brown"
    YELLOW = "yellow"    # Golden/cream/tan
    GRAY = "gray"
    MIXED = "mixed"      # Multi-colored
    UNKNOWN = "unknown"


@dataclass
class DogProfile:
    """Dog profile from cloud/app"""
    name: str
    aruco_id: Optional[int] = None
    color: DogColor = DogColor.UNKNOWN
    color_rgb: Optional[Tuple[int, int, int]] = None  # Specific RGB for matching
    household_id: str = ""
    photo_url: Optional[str] = None

    # Runtime state
    last_seen: float = 0.0
    identification_count: int = 0


@dataclass
class IdentificationResult:
    """Result of dog identification attempt"""
    dog_name: str
    method: IdentificationMethod
    confidence: float
    aruco_id: Optional[int] = None
    color_detected: Optional[DogColor] = None
    profile: Optional[DogProfile] = None


class DogProfileManager:
    """
    Manages dog profiles and identification logic.

    Fetches profiles from cloud, extracts colors, and identifies dogs
    using ARUCO markers or color matching.
    """

    # Color ranges in HSV for classification
    # Format: (lower_bound, upper_bound) as (H, S, V)
    COLOR_RANGES = {
        DogColor.BLACK: [
            ((0, 0, 0), (180, 255, 50)),      # Very dark
        ],
        DogColor.WHITE: [
            ((0, 0, 200), (180, 30, 255)),    # High value, low saturation
        ],
        DogColor.BROWN: [
            ((5, 50, 50), (20, 255, 200)),    # Brown range
            ((0, 50, 50), (10, 255, 180)),    # Reddish brown
        ],
        DogColor.YELLOW: [
            ((15, 40, 150), (35, 255, 255)),  # Golden/tan/cream
            ((20, 20, 180), (40, 150, 255)),  # Light cream
        ],
        DogColor.GRAY: [
            ((0, 0, 50), (180, 50, 180)),     # Gray range
        ],
    }

    def __init__(self):
        self.logger = logging.getLogger('DogProfileManager')
        self._profiles: Dict[str, DogProfile] = {}  # name -> profile
        self._aruco_map: Dict[int, str] = {}        # aruco_id -> name
        self._color_map: Dict[DogColor, List[str]] = {}  # color -> [names]
        self._lock = threading.RLock()

        # Color extraction settings
        self._sample_size = 50  # pixels to sample for color

        # Default profiles (can be overridden by cloud)
        self._load_default_profiles()

        self.logger.info("DogProfileManager initialized")

    def _load_default_profiles(self):
        """Load default dog profiles from config"""
        # Default profiles for WIM-Z household
        defaults = [
            DogProfile(
                name="bezik",
                aruco_id=832,
                color=DogColor.BLACK,
                household_id="default"
            ),
            DogProfile(
                name="elsa",
                aruco_id=1,
                color=DogColor.YELLOW,
                household_id="default"
            ),
        ]

        for profile in defaults:
            self.add_profile(profile)

        self.logger.info(f"Loaded {len(defaults)} default profiles")

    def add_profile(self, profile: DogProfile):
        """Add or update a dog profile"""
        with self._lock:
            name = profile.name.lower()
            self._profiles[name] = profile

            # Update ARUCO mapping
            if profile.aruco_id is not None:
                self._aruco_map[profile.aruco_id] = name

            # Update color mapping
            if profile.color != DogColor.UNKNOWN:
                if profile.color not in self._color_map:
                    self._color_map[profile.color] = []
                if name not in self._color_map[profile.color]:
                    self._color_map[profile.color].append(name)

            self.logger.debug(f"Added profile: {name} (aruco={profile.aruco_id}, color={profile.color.value})")

    def update_profiles_from_cloud(self, profiles_data: List[Dict[str, Any]]):
        """Update profiles from cloud/relay data

        Expected format:
        [
            {
                "name": "Bezik",
                "aruco_id": 832,
                "color": "black",
                "photo_url": "..."
            },
            ...
        ]
        """
        with self._lock:
            # Clear existing profiles
            self._profiles.clear()
            self._aruco_map.clear()
            self._color_map.clear()

            for data in profiles_data:
                try:
                    color_str = data.get('color', 'unknown').lower()
                    try:
                        color = DogColor(color_str)
                    except ValueError:
                        color = DogColor.UNKNOWN

                    profile = DogProfile(
                        name=data.get('name', 'Unknown'),
                        aruco_id=data.get('aruco_id'),
                        color=color,
                        household_id=data.get('household_id', ''),
                        photo_url=data.get('photo_url')
                    )
                    self.add_profile(profile)

                except Exception as e:
                    self.logger.warning(f"Failed to parse profile: {e}")

            self.logger.info(f"Updated {len(self._profiles)} profiles from cloud")

    def identify_dog(
        self,
        bbox: Optional[List[float]] = None,
        frame: Optional[np.ndarray] = None,
        aruco_id: Optional[int] = None,
        aruco_markers: Optional[List[Tuple[int, float, float]]] = None
    ) -> IdentificationResult:
        """
        Identify a dog using available information.

        Priority:
        1. ARUCO marker ID
        2. Color matching (if unique)
        3. Unknown

        Args:
            bbox: Bounding box [x1, y1, x2, y2] of detected dog
            frame: Camera frame for color extraction
            aruco_id: Directly detected ARUCO ID
            aruco_markers: List of (id, cx, cy) tuples to check in bbox

        Returns:
            IdentificationResult with dog name and method
        """
        with self._lock:
            # Priority 1: Direct ARUCO ID
            if aruco_id is not None and aruco_id in self._aruco_map:
                name = self._aruco_map[aruco_id]
                profile = self._profiles.get(name)
                return IdentificationResult(
                    dog_name=name.capitalize(),
                    method=IdentificationMethod.ARUCO,
                    confidence=1.0,
                    aruco_id=aruco_id,
                    profile=profile
                )

            # Priority 1b: Check ARUCO markers in bbox
            if aruco_markers and bbox:
                x1, y1, x2, y2 = bbox[:4]
                for marker_id, cx, cy in aruco_markers:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        if marker_id in self._aruco_map:
                            name = self._aruco_map[marker_id]
                            profile = self._profiles.get(name)
                            return IdentificationResult(
                                dog_name=name.capitalize(),
                                method=IdentificationMethod.ARUCO,
                                confidence=1.0,
                                aruco_id=marker_id,
                                profile=profile
                            )

            # Priority 2: Color-based identification
            if frame is not None and bbox is not None:
                color = self.extract_dominant_color(frame, bbox)

                if color != DogColor.UNKNOWN and color in self._color_map:
                    matching_dogs = self._color_map[color]

                    if len(matching_dogs) == 1:
                        # Unique color match - high confidence
                        name = matching_dogs[0]
                        profile = self._profiles.get(name)
                        return IdentificationResult(
                            dog_name=name.capitalize(),
                            method=IdentificationMethod.COLOR_UNIQUE,
                            confidence=0.8,
                            color_detected=color,
                            profile=profile
                        )
                    elif len(matching_dogs) > 1:
                        # Multiple dogs with same color - ambiguous
                        return IdentificationResult(
                            dog_name="Dog",
                            method=IdentificationMethod.COLOR_AMBIGUOUS,
                            confidence=0.3,
                            color_detected=color
                        )

            # Priority 3: Unknown
            return IdentificationResult(
                dog_name="Dog",
                method=IdentificationMethod.UNKNOWN,
                confidence=0.0
            )

    def extract_dominant_color(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> DogColor:
        """
        Extract dominant color from dog region.

        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Classified DogColor
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return DogColor.UNKNOWN

            # Crop dog region
            dog_region = frame[y1:y2, x1:x2]

            # Sample center region (avoid edges/background)
            ch, cw = dog_region.shape[:2]
            margin = 0.2
            cx1, cy1 = int(cw * margin), int(ch * margin)
            cx2, cy2 = int(cw * (1 - margin)), int(ch * (1 - margin))

            if cx2 <= cx1 or cy2 <= cy1:
                center_region = dog_region
            else:
                center_region = dog_region[cy1:cy2, cx1:cx2]

            # Convert to HSV
            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

            # Calculate average color
            avg_hsv = np.mean(hsv, axis=(0, 1))
            h, s, v = avg_hsv

            # Classify based on HSV values
            return self._classify_color_hsv(h, s, v)

        except Exception as e:
            self.logger.debug(f"Color extraction error: {e}")
            return DogColor.UNKNOWN

    def _classify_color_hsv(self, h: float, s: float, v: float) -> DogColor:
        """Classify color based on HSV values"""

        # Black: very low value
        if v < 50:
            return DogColor.BLACK

        # White: low saturation, high value
        if s < 30 and v > 200:
            return DogColor.WHITE

        # Gray: low saturation, medium value
        if s < 50 and 50 <= v <= 180:
            return DogColor.GRAY

        # Brown: orange-red hue with medium saturation
        if (0 <= h <= 20 or 160 <= h <= 180) and s > 50 and v < 200:
            return DogColor.BROWN

        # Yellow/Golden: yellow hue
        if 15 <= h <= 40 and s > 30:
            return DogColor.YELLOW

        return DogColor.MIXED

    def get_profile_by_aruco(self, aruco_id: int) -> Optional[DogProfile]:
        """Get profile by ARUCO ID"""
        with self._lock:
            name = self._aruco_map.get(aruco_id)
            if name:
                return self._profiles.get(name)
            return None

    def get_profile_by_name(self, name: str) -> Optional[DogProfile]:
        """Get profile by name"""
        with self._lock:
            return self._profiles.get(name.lower())

    def get_all_profiles(self) -> List[DogProfile]:
        """Get all profiles"""
        with self._lock:
            return list(self._profiles.values())

    def get_aruco_mapping(self) -> Dict[int, str]:
        """Get ARUCO ID to name mapping (for dog tracker)"""
        with self._lock:
            return dict(self._aruco_map)

    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        with self._lock:
            return {
                'profile_count': len(self._profiles),
                'profiles': [
                    {
                        'name': p.name,
                        'aruco_id': p.aruco_id,
                        'color': p.color.value
                    }
                    for p in self._profiles.values()
                ],
                'aruco_ids': list(self._aruco_map.keys()),
                'color_map': {
                    color.value: names
                    for color, names in self._color_map.items()
                }
            }


# Global singleton
_profile_manager: Optional[DogProfileManager] = None
_profile_lock = threading.Lock()


def get_dog_profile_manager() -> DogProfileManager:
    """Get the global DogProfileManager instance"""
    global _profile_manager
    with _profile_lock:
        if _profile_manager is None:
            _profile_manager = DogProfileManager()
    return _profile_manager
