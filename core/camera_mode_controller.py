#!/usr/bin/env python3
"""
Camera Mode Controller for TreatBot

Manages 4 camera operation modes:
1. Photography Mode - Max resolution, no AI, manual controls
2. AI Detection Mode - Single 640x640 frame, real-time inference
3. Vigilant Mode - Full frame tiling, comprehensive detection
4. Auto-switching based on vehicle state
"""

import enum
import threading
import time
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logger = logging.getLogger(__name__)

class CameraMode(enum.Enum):
    """Camera operation modes"""
    PHOTOGRAPHY = "photography"  # Max res, no AI
    AI_DETECTION = "ai_detection"  # 640x640 single frame
    VIGILANT = "vigilant"  # Full res with tiling
    IDLE = "idle"  # Camera off/standby

@dataclass
class TileRegion:
    """Defines a tile region for vigilant mode"""
    x: int
    y: int
    width: int
    height: int
    tile_id: int

@dataclass
class ModeConfig:
    """Configuration for each camera mode"""
    resolution: Tuple[int, int]
    fps: int
    ai_enabled: bool
    tiling_enabled: bool
    stream_enabled: bool
    manual_controls: bool

class CameraModeController:
    """Manages camera modes and automatic switching"""
    
    # Mode configurations
    MODE_CONFIGS = {
        CameraMode.PHOTOGRAPHY: ModeConfig(
            resolution=(4056, 3040),  # Max 4K resolution
            fps=10,
            ai_enabled=False,
            tiling_enabled=False,
            stream_enabled=True,
            manual_controls=True
        ),
        CameraMode.AI_DETECTION: ModeConfig(
            resolution=(640, 640),  # Direct 640x640 for YOLO
            fps=30,
            ai_enabled=True,
            tiling_enabled=False,
            stream_enabled=True,
            manual_controls=False
        ),
        CameraMode.VIGILANT: ModeConfig(
            resolution=(1920, 1080),  # Full HD for tiling
            fps=15,
            ai_enabled=True,
            tiling_enabled=True,
            stream_enabled=False,  # Only send detections
            manual_controls=False
        ),
        CameraMode.IDLE: ModeConfig(
            resolution=(640, 480),
            fps=1,
            ai_enabled=False,
            tiling_enabled=False,
            stream_enabled=False,
            manual_controls=False
        )
    }
    
    def __init__(self, ai_controller=None):
        """
        Initialize camera mode controller
        
        Args:
            ai_controller: AI3StageControllerFixed instance for inference
        """
        self.current_mode = CameraMode.IDLE
        self.ai_controller = ai_controller
        self.camera = None
        self.mode_lock = threading.Lock()
        self.frame_buffer = []
        self.is_running = False
        
        # Mode transition callbacks
        self.mode_callbacks = {}
        
        # Auto-switch triggers
        self.vehicle_moving = False
        self.manual_override = False
        
        # Tiling configuration for vigilant mode
        self.tile_size = 640
        self.tile_overlap = 0  # No overlap for speed
        
        # Photography mode controls
        self.manual_settings = {
            'iso': 100,
            'exposure': 10000,
            'gain': 1.0,
            'white_balance': 'auto'
        }
        
        self._init_camera()
    
    def _init_camera(self):
        """Initialize Picamera2 instance"""
        if not PICAMERA2_AVAILABLE:
            logger.error("Picamera2 not available")
            return
            
        try:
            self.camera = Picamera2()
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def set_mode(self, mode: CameraMode, force: bool = False) -> bool:
        """
        Switch to specified camera mode
        
        Args:
            mode: Target camera mode
            force: Override manual lock
            
        Returns:
            Success status
        """
        if self.manual_override and not force:
            logger.warning("Manual override active, use force=True to switch")
            return False
            
        with self.mode_lock:
            if self.current_mode == mode:
                return True
                
            # Stop current mode
            self._stop_current_mode()
            
            # Buffer last frames for smooth transition
            self._buffer_transition_frames()
            
            # Configure new mode
            success = self._configure_mode(mode)
            
            if success:
                old_mode = self.current_mode
                self.current_mode = mode
                logger.info(f"Mode switched: {old_mode.value} -> {mode.value}")
                
                # Trigger callbacks
                if mode in self.mode_callbacks:
                    self.mode_callbacks[mode]()
                    
                return True
            else:
                logger.error(f"Failed to switch to {mode.value}")
                return False
    
    def _configure_mode(self, mode: CameraMode) -> bool:
        """
        Configure camera for specific mode

        Args:
            mode: Target mode

        Returns:
            Success status
        """
        if not self.camera:
            return False

        config = self.MODE_CONFIGS[mode]

        try:
            # Stop camera if running
            try:
                if hasattr(self.camera, 'is_open') and self.camera.is_open:
                    self.camera.stop()
                    self.camera.close()
            except Exception:
                # Camera might not be started yet, ignore
                pass

            # Configure for new mode
            if mode != CameraMode.IDLE:
                camera_config = self.camera.create_still_configuration(
                    main={"size": config.resolution}
                )
                self.camera.configure(camera_config)
                self.camera.start()

                # Small delay for camera to stabilize
                time.sleep(0.5)

                # Apply manual controls for photography mode
                if mode == CameraMode.PHOTOGRAPHY and config.manual_controls:
                    self._apply_manual_controls()

            return True

        except Exception as e:
            logger.error(f"Mode configuration failed: {e}")
            return False
    
    def _apply_manual_controls(self):
        """Apply manual camera controls for photography mode"""
        if not self.camera or not self.camera.is_open:
            return
            
        try:
            controls = Controls(self.camera)
            controls.AnalogueGain = self.manual_settings['gain']
            controls.ExposureTime = self.manual_settings['exposure']
            # Note: ISO and white balance depend on camera capabilities
            self.camera.set_controls(controls)
            logger.info(f"Applied manual controls: {self.manual_settings}")
        except Exception as e:
            logger.warning(f"Could not apply all manual controls: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture frame based on current mode

        Returns:
            Frame array or None
        """
        if not self.camera:
            return None

        # Check if camera is properly started
        try:
            if not hasattr(self.camera, 'is_open') or not self.camera.is_open:
                return None
        except Exception:
            return None

        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def process_vigilant_mode(self, frame: np.ndarray) -> List[Dict]:
        """
        Process full frame in vigilant mode with tiling
        
        Args:
            frame: Full resolution frame
            
        Returns:
            List of detections across all tiles
        """
        if not self.ai_controller:
            return []
            
        height, width = frame.shape[:2]
        tiles = self._generate_tiles(width, height)
        all_detections = []
        
        for tile in tiles:
            # Extract tile from frame
            tile_img = frame[
                tile.y:tile.y + tile.height,
                tile.x:tile.x + tile.width
            ]
            
            # Resize to 640x640 if needed
            if tile_img.shape[:2] != (640, 640):
                tile_img = cv2.resize(tile_img, (640, 640))
            
            # Run inference
            detections = self.ai_controller.detect_dogs(tile_img)
            
            # Adjust coordinates back to full frame
            for det in detections:
                det['x1'] = det['x1'] + tile.x
                det['y1'] = det['y1'] + tile.y
                det['x2'] = det['x2'] + tile.x
                det['y2'] = det['y2'] + tile.y
                det['tile_id'] = tile.tile_id
                all_detections.append(det)
        
        # Merge overlapping detections if using overlap
        if self.tile_overlap > 0:
            all_detections = self._merge_overlapping_detections(all_detections)
            
        return all_detections
    
    def _generate_tiles(self, width: int, height: int) -> List[TileRegion]:
        """
        Generate tile regions for vigilant mode
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            List of tile regions
        """
        tiles = []
        tile_id = 0
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                tile_width = min(self.tile_size, width - x)
                tile_height = min(self.tile_size, height - y)
                
                tiles.append(TileRegion(
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                    tile_id=tile_id
                ))
                
                tile_id += 1
                x += self.tile_size - self.tile_overlap
            y += self.tile_size - self.tile_overlap
            
        return tiles
    
    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge detections from overlapping tiles
        
        Args:
            detections: All detections from tiles
            
        Returns:
            Merged detections
        """
        # Simple NMS-style merging
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        merged = []
        
        for det in detections:
            # Check if overlaps with existing
            is_duplicate = False
            for existing in merged:
                iou = self._calculate_iou(det, existing)
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                merged.append(det)
                
        return merged
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate IoU between two boxes
        
        Args:
            box1, box2: Detection dictionaries with x1,y1,x2,y2
            
        Returns:
            IoU score
        """
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def auto_switch_mode(self):
        """
        Automatically switch modes based on vehicle state
        """
        if self.manual_override:
            return
            
        if self.vehicle_moving:
            # Moving: Use AI Detection for speed
            self.set_mode(CameraMode.AI_DETECTION)
        else:
            # Stationary: Use Vigilant for coverage
            self.set_mode(CameraMode.VIGILANT)
    
    def set_vehicle_state(self, is_moving: bool):
        """
        Update vehicle movement state
        
        Args:
            is_moving: Whether vehicle is in motion
        """
        self.vehicle_moving = is_moving
        if not self.manual_override:
            self.auto_switch_mode()
    
    def _stop_current_mode(self):
        """Stop current camera mode"""
        # Mode-specific cleanup
        if self.current_mode == CameraMode.PHOTOGRAPHY:
            # Save any pending photos
            pass
        elif self.current_mode in [CameraMode.AI_DETECTION, CameraMode.VIGILANT]:
            # Stop AI inference
            pass
    
    def _buffer_transition_frames(self):
        """Buffer last few frames for smooth mode transition"""
        if self.camera and self.camera.is_open:
            try:
                # Capture 3 frames
                self.frame_buffer = []
                for _ in range(3):
                    frame = self.capture_frame()
                    if frame is not None:
                        self.frame_buffer.append(frame)
                        time.sleep(0.033)  # ~30fps
            except:
                pass
    
    def register_mode_callback(self, mode: CameraMode, callback: Callable):
        """
        Register callback for mode transitions
        
        Args:
            mode: Camera mode
            callback: Function to call on mode entry
        """
        self.mode_callbacks[mode] = callback
    
    def get_status(self) -> Dict:
        """
        Get current controller status
        
        Returns:
            Status dictionary
        """
        config = self.MODE_CONFIGS[self.current_mode]
        return {
            'current_mode': self.current_mode.value,
            'resolution': config.resolution,
            'fps': config.fps,
            'ai_enabled': config.ai_enabled,
            'tiling_enabled': config.tiling_enabled,
            'vehicle_moving': self.vehicle_moving,
            'manual_override': self.manual_override,
            'camera_ready': self.camera is not None
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            try:
                if self.camera.is_open:
                    self.camera.stop()
                    self.camera.close()
            except:
                pass
        self.is_running = False