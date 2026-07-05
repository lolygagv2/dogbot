#!/usr/bin/env python3
"""
Detection service wrapper for ai_controller_3stage_fixed.py
Publishes pose events to event bus
"""

import threading
import time
import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from core.bus import get_bus, publish_vision_event
from core.state import get_state, SystemMode
from core.store import get_store

# Dual-stream configuration (R-STREAM / work order §2B):
# The camera always produces TWO streams — a fixed 640x640 'lores' stream that
# feeds Hailo inference, and a 'main' stream for WebRTC/snapshots/recording.
# MODE_RESOLUTIONS now controls ONLY the main (stream) resolution; AI never
# sees a resolution change, and AI-mode switches need no camera reconfig.
AI_RESOLUTION = (640, 640)  # lores stream, fixed — Hailo model input
MODE_RESOLUTIONS = {
    SystemMode.IDLE: (1280, 720),            # 720p stream, instant mode switch
    SystemMode.MANUAL: (1920, 1080),          # No AI, full HD video
    SystemMode.SILENT_GUARDIAN: (1280, 720), # AI on lores, 720p stream
    SystemMode.COACH: (1280, 720),           # AI on lores, 720p stream
    SystemMode.PHOTOGRAPHY: (1920, 1080),     # High res photos
    SystemMode.EMERGENCY: (1280, 720),       # Safety mode
    SystemMode.SHUTDOWN: (1280, 720),        # Shutting down
}
DEFAULT_RESOLUTION = (1280, 720)
from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from config.config_loader import get_config

# Camera imports
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ArUco detection
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
DOG_MARKERS = {}  # Populated at runtime from dog profiles


def _load_dog_markers():
    """Load ArUco marker → dog name mappings from profiles and config"""
    global DOG_MARKERS
    try:
        from core.dog_profile_manager import get_dog_profile_manager
        pm = get_dog_profile_manager()
        for profile in pm.get_all_profiles():
            if profile.aruco_id is not None:
                DOG_MARKERS[profile.aruco_id] = profile.name.lower()
    except Exception:
        pass
    # Fallback to config.json if profiles empty
    if not DOG_MARKERS:
        try:
            import json
            with open('/home/morgan/dogbot/config/config.json') as f:
                cfg = json.load(f)
            for dog in cfg.get('dogs', []):
                mid = dog.get('marker_id')
                name = dog.get('id', '')
                if mid is not None:
                    DOG_MARKERS[int(mid)] = name.lower()
        except Exception:
            pass


class DetectorService:
    """
    Wrapper for AI3StageControllerFixed that publishes to event bus
    Handles mode-based pipeline switching
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.logger = logging.getLogger('DetectorService')

        # AI Controller
        self.ai = None
        self.ai_initialized = False

        # Camera
        self.camera = None
        self.camera_initialized = False

        # ArUco detector with tuned parameters to reduce false positives
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshConstant = 10
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.minMarkerPerimeterRate = 0.05  # Reject tiny markers (noise)
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.errorCorrectionRate = 0.4  # Stricter error correction (default 0.6)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self._aruco_consecutive = {}  # {marker_id: consecutive_frame_count} for validation
        self._aruco_min_frames = 3  # Require 3 consecutive frames before accepting

        # Dog identity persistence: once ARUCO identifies dog_0 as "elsa", keep that
        # mapping even when ARUCO isn't visible (intermittent detection)
        self._dog_identity_cache: Dict[int, str] = {}  # dog_index -> aruco_name
        self._dog_last_seen: Dict[int, float] = {}  # dog_index -> last_seen_time
        self._identity_persist_timeout = 5.0  # Keep identity for 5s after dog leaves frame

        # Load dog marker mappings from profiles/config
        _load_dog_markers()
        self.logger.info(f"[ARUCO] Dog markers loaded: {DOG_MARKERS}")

        # Detection state
        self.running = False
        self.detection_thread = None
        self._stop_event = threading.Event()

        # Camera pause state (for MANUAL mode)
        self._camera_paused = False
        self._ai_paused = False
        # _last_frame holds the MAIN (stream-resolution) frame for
        # snapshots/WebRTC/recording; AI consumes the lores frame returned
        # directly by _capture_frame()
        self._last_frame = None
        self._last_frame_time = 0
        self._frame_lock = threading.Lock()
        self._dual_stream = False  # set by _initialize_camera

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # --- R3: AI diagnostics ---
        self._heartbeat_time = time.time()
        self._heartbeat_detections = 0
        self._heartbeat_last_class = ""
        self._heartbeat_last_confidence = 0.0
        self._frames_processed_total = 0
        self._last_detection_info = {}  # Last detection for status endpoint

        # Failure counters (per 60s window)
        self._stats_window_start = time.time()
        self._cat_a_errors = 0    # Pipeline not running
        self._cat_b_misses = 0    # Below threshold
        self._cat_c_fails = 0     # Classification fails

        # Rate limiting for detection events (prevent log spam)
        self._last_detection_event_time = 0
        self._detection_event_interval = 0.2  # Minimum seconds between detection events

        # Camera retry logic
        self._camera_error_reason = None
        self._camera_retry_interval = 60  # seconds between retry attempts
        self._last_camera_retry_time = 0

        # Resolution tracking
        self._current_resolution = DEFAULT_RESOLUTION
        self._resolution_change_lock = threading.Lock()
        self._resolution_change_in_progress = False

        # Subscribe to mode changes for resolution switching
        self.state.subscribe('mode_change', self._on_mode_change)

    def initialize(self) -> bool:
        """Initialize AI controller and camera

        Returns True if camera works (for WebRTC), even if AI/Hailo fails.
        AI is optional - WebRTC streaming works without it.
        """
        try:
            # Initialize camera first - this is required for WebRTC
            self._initialize_camera()

            # Initialize AI controller (optional - Hailo may not be available)
            try:
                self.ai = AI3StageControllerFixed()
                success = self.ai.initialize()

                if success:
                    self.ai_initialized = True
                    self.logger.info("AI controller initialized (full pipeline)")
                else:
                    self.logger.warning("AI controller init failed - WebRTC will work, no AI processing")
            except Exception as e:
                self.logger.warning(f"AI controller not available: {e} - WebRTC will work, no AI processing")

            self.state.update_hardware(camera_initialized=self.camera_initialized)

            # Return True if camera works - AI is optional
            if self.camera_initialized:
                return True
            else:
                self.logger.error("Camera initialization failed - no video available")
                return False

        except Exception as e:
            self.logger.error(f"Detector initialization error: {e}")
            return False

    def _build_camera_config(self, stream_resolution: tuple):
        """Build the camera configuration for a given main-stream resolution.

        Dual-stream when supported: main = stream/snapshot feed at
        stream_resolution, lores = fixed AI_RESOLUTION feed for Hailo.
        Respects the single-stream fallback chosen at init.
        """
        if self._dual_stream:
            return self.camera.create_video_configuration(
                main={"size": stream_resolution, "format": "RGB888"},
                lores={"size": AI_RESOLUTION, "format": "RGB888"}
            )
        return self.camera.create_preview_configuration(
            main={"size": AI_RESOLUTION, "format": "RGB888"}
        )

    def _initialize_camera(self) -> bool:
        """Initialize camera for frame capture"""
        try:
            if PICAMERA_AVAILABLE:
                self.logger.info("Initializing Picamera2 for detection...")

                # Check if any cameras are available BEFORE creating Picamera2
                available_cameras = Picamera2.global_camera_info()
                if not available_cameras:
                    self.logger.error(
                        "No cameras detected! CSI cameras require a system reboot after "
                        "plugging in. Check ribbon cable connection and run: sudo reboot"
                    )
                    self._camera_error_reason = "no_cameras_detected"
                    return False

                self.logger.info(f"Detected cameras: {available_cameras}")

                # Reset any stale camera instances
                try:
                    import subprocess
                    # Kill any stale libcamera processes that might be holding the camera
                    subprocess.run(['pkill', '-f', 'libcamera'], timeout=2, capture_output=True)
                    time.sleep(0.5)
                except Exception:
                    pass

                self.camera = Picamera2()

                # Use mode-based resolution
                current_mode = self.state.get_mode()
                target_resolution = MODE_RESOLUTIONS.get(current_mode, DEFAULT_RESOLUTION)
                self._current_resolution = target_resolution
                self.logger.info(f"Camera resolution for {current_mode.value}: {target_resolution}")

                # Dual-stream: main feeds WebRTC/snapshots, lores feeds Hailo.
                # Falls back to single-stream if lores RGB888 is unsupported.
                try:
                    config = self.camera.create_video_configuration(
                        main={"size": target_resolution, "format": "RGB888"},
                        lores={"size": AI_RESOLUTION, "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    self._dual_stream = True
                except Exception as e:
                    self.logger.warning(
                        f"Dual-stream config failed ({e}), falling back to "
                        f"single {AI_RESOLUTION} stream"
                    )
                    config = self.camera.create_preview_configuration(
                        main={"size": AI_RESOLUTION, "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    self._dual_stream = False
                self.camera.start()

                # Wait for camera to stabilize
                time.sleep(0.5)

                # Apply saved calibration from robot profile
                self._apply_saved_calibration()

                # Verify camera is actually working
                try:
                    test_frame = self.camera.capture_array()
                    if test_frame is None:
                        self.logger.error("Camera test capture returned None")
                        return False
                    self.logger.info(f"Camera test capture successful: {test_frame.shape}")
                except Exception as e:
                    self.logger.error(f"Camera test capture failed: {e}")
                    return False

                self.camera_initialized = True
                self.logger.info("Picamera2 initialized for detection")
                return True
            else:
                self.logger.warning("Picamera2 not available, detection disabled")
                return False
        except IndexError as e:
            self.logger.error(
                f"Camera initialization error: {e} - No cameras found. "
                "CSI cameras require a reboot after connecting the ribbon cable."
            )
            self._camera_error_reason = "no_cameras_detected"
            return False
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            self._camera_error_reason = str(e)
            return False

    def _apply_saved_calibration(self):
        """Apply saved camera calibration from robot profile"""
        try:
            config = get_config()
            cam_config = config.camera

            controls = {}
            awb_modes = {
                "auto": 0, "incandescent": 1, "tungsten": 2, "fluorescent": 3,
                "indoor": 4, "daylight": 5, "cloudy": 6, "custom": 7
            }

            # Check for calibration settings in robot profile
            if hasattr(cam_config, 'awb_mode') and cam_config.awb_mode:
                mode = cam_config.awb_mode.lower()
                if mode in awb_modes:
                    controls["AwbMode"] = awb_modes[mode]

            if hasattr(cam_config, 'brightness') and cam_config.brightness is not None:
                controls["Brightness"] = float(cam_config.brightness)

            if hasattr(cam_config, 'contrast') and cam_config.contrast is not None:
                controls["Contrast"] = float(cam_config.contrast)

            if hasattr(cam_config, 'saturation') and cam_config.saturation is not None:
                controls["Saturation"] = float(cam_config.saturation)

            if hasattr(cam_config, 'sharpness') and cam_config.sharpness is not None:
                controls["Sharpness"] = float(cam_config.sharpness)

            if hasattr(cam_config, 'exposure_time') and cam_config.exposure_time:
                controls["ExposureTime"] = int(cam_config.exposure_time)

            if hasattr(cam_config, 'analogue_gain') and cam_config.analogue_gain:
                controls["AnalogueGain"] = float(cam_config.analogue_gain)

            if hasattr(cam_config, 'awb_enable') and cam_config.awb_enable is not None:
                controls["AwbEnable"] = bool(cam_config.awb_enable)

            if hasattr(cam_config, 'colour_gains') and cam_config.colour_gains:
                cg = cam_config.colour_gains
                controls["ColourGains"] = (float(cg[0]), float(cg[1]))

            if controls:
                self.camera.set_controls(controls)
                self.logger.info(f"Applied saved camera calibration: {controls}")
            else:
                self.logger.debug("No saved camera calibration found")

        except Exception as e:
            self.logger.warning(f"Could not apply saved calibration: {e}")

    def _capture_frame(self):
        """Capture a frame from the camera.

        Returns the lores (AI-resolution) frame for inference. The main
        (stream-resolution) frame is stored in _last_frame for
        snapshots/WebRTC/recording. Single-stream fallback returns/stores
        the same frame.
        """
        if not self.camera_initialized or self.camera is None:
            self.logger.debug("Camera not initialized or None")
            return None
        try:
            if self._dual_stream:
                # One request, both streams — no extra frame latency
                (frame, stream_frame), _md = self.camera.capture_arrays(["lores", "main"])
            else:
                frame = self.camera.capture_array()
                stream_frame = None  # single-stream: main frame doubles for both

            # Apply rotation from robot config (0, 90, 180, 270 degrees clockwise)
            config = get_config()
            rotation = config.camera.rotation
            rotate_code = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }.get(rotation)
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
                if stream_frame is not None:
                    stream_frame = cv2.rotate(stream_frame, rotate_code)

            # Store last main-stream frame for snapshots/WebRTC (thread-safe)
            with self._frame_lock:
                self._last_frame = stream_frame if stream_frame is not None else frame.copy()
                self._last_frame_time = time.time()

            return frame
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            # Try to recover camera on repeated errors
            self._camera_error_count = getattr(self, '_camera_error_count', 0) + 1
            if self._camera_error_count >= 5:
                self.logger.warning("Multiple camera errors, attempting to reinitialize...")
                self._reinitialize_camera()
                self._camera_error_count = 0
            return None

    def _reinitialize_camera(self):
        """Attempt to reinitialize camera after errors"""
        try:
            # Close existing camera
            if self.camera:
                try:
                    self.camera.stop()
                    self.camera.close()
                except Exception as e:
                    self.logger.warning(f"Error closing camera during reinit: {e}")
                self.camera = None
                self.camera_initialized = False

            # Wait for resources to be released
            time.sleep(0.5)

            # Reinitialize
            if self._initialize_camera():
                self.logger.info("Camera reinitialized successfully")
            else:
                self.logger.error("Camera reinitialization failed")
        except Exception as e:
            self.logger.error(f"Camera reinitialization error: {e}")

    def _release_camera(self) -> None:
        """Release camera for external use (e.g., rpicam-still in MANUAL mode)"""
        if self.camera is not None and self.camera_initialized:
            try:
                self.camera.stop()
                self.camera.close()
                self.camera = None
                self.camera_initialized = False
                self._camera_paused = True
                self.logger.debug("Camera released for MANUAL mode")
            except Exception as e:
                self.logger.error(f"Camera release error: {e}")

    def _reacquire_camera(self) -> bool:
        """Re-acquire camera after MANUAL mode"""
        if self._camera_paused:
            self.logger.debug("Re-acquiring camera from MANUAL mode...")
            result = self._initialize_camera()
            if result:
                self._camera_paused = False
                self.logger.debug("Camera re-acquired successfully")
            return result
        return self.camera_initialized

    def _on_mode_change(self, data: Dict[str, Any]) -> None:
        """Handle mode change for resolution switching"""
        try:
            new_mode_str = data.get('new_mode', '')
            previous_mode_str = data.get('previous_mode', '')

            # Convert string to SystemMode enum
            try:
                new_mode = SystemMode(new_mode_str)
            except ValueError:
                self.logger.warning(f"Unknown mode: {new_mode_str}")
                return

            # Get target resolution for new mode
            target_resolution = MODE_RESOLUTIONS.get(new_mode, DEFAULT_RESOLUTION)

            # Only change if resolution differs
            if target_resolution != self._current_resolution:
                self.logger.debug(
                    f"Mode change {previous_mode_str} -> {new_mode_str}: "
                    f"Resolution {self._current_resolution} -> {target_resolution}"
                )
                self._change_resolution(target_resolution)
            else:
                self.logger.debug(f"Mode change to {new_mode_str}, resolution unchanged at {target_resolution}")

        except Exception as e:
            self.logger.error(f"Mode change handler error: {e}")

    def _change_resolution(self, new_resolution: tuple) -> bool:
        """Safely change camera resolution

        Thread-safe method that:
        1. Stops camera capture
        2. Reconfigures with new resolution
        3. Restarts camera

        Returns True if successful, False otherwise.
        Falls back to previous resolution on failure.
        """
        with self._resolution_change_lock:
            if self._resolution_change_in_progress:
                self.logger.warning("Resolution change already in progress, skipping")
                return False

            self._resolution_change_in_progress = True

        old_resolution = self._current_resolution

        try:
            if not self.camera_initialized or self.camera is None:
                # Camera not initialized, just update target resolution
                self._current_resolution = new_resolution
                self.logger.debug(f"Resolution set to {new_resolution} (camera not active)")
                return True

            self.logger.debug(f"Changing resolution from {old_resolution} to {new_resolution}...")

            # Pause WebRTC video track before camera reconfig to prevent stale reads
            publish_vision_event('webrtc_pause_requested', {
                'reason': 'camera_reconfig',
                'old_resolution': old_resolution,
                'new_resolution': new_resolution
            })
            time.sleep(0.5)  # Allow WebRTC to pause

            # Stop camera
            try:
                self.camera.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping camera: {e}")

            # Reconfigure (main/stream size only — lores AI stream is fixed)
            try:
                config = self._build_camera_config(new_resolution)
                self.camera.configure(config)
            except Exception as e:
                self.logger.error(f"Error configuring camera at {new_resolution}: {e}")
                # Try to restore old resolution
                try:
                    config = self._build_camera_config(old_resolution)
                    self.camera.configure(config)
                    self.camera.start()
                    self.logger.warning(f"Restored previous resolution {old_resolution}")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore resolution: {restore_error}")
                    self.camera_initialized = False
                # Resume WebRTC even on failure
                publish_vision_event('webrtc_resume_requested', {'reason': 'camera_reconfig_failed'})
                return False

            # Restart camera
            try:
                self.camera.start()
                time.sleep(0.3)  # Brief stabilization

                # Verify with test capture
                test_frame = self.camera.capture_array()
                if test_frame is not None:
                    actual_shape = test_frame.shape[:2]  # (height, width)
                    expected_shape = (new_resolution[1], new_resolution[0])  # height, width
                    self.logger.debug(f"Resolution changed successfully: {actual_shape}")
                    self._current_resolution = new_resolution
                    return True
                else:
                    self.logger.error("Test capture returned None after resolution change")
                    return False

            except Exception as e:
                self.logger.error(f"Error restarting camera: {e}")
                self.camera_initialized = False
                return False

        finally:
            # Resume WebRTC video track after camera reconfig completes (success or failure)
            publish_vision_event('webrtc_resume_requested', {
                'reason': 'camera_reconfig_complete',
                'resolution': list(self._current_resolution)
            })
            with self._resolution_change_lock:
                self._resolution_change_in_progress = False

    def get_current_resolution(self) -> tuple:
        """Get current camera resolution"""
        return self._current_resolution

    def pause_ai(self):
        """Pause AI processing (keep camera running for frame capture)"""
        self._ai_paused = True
        self.logger.info("AI detection paused")

    def resume_ai(self):
        """Resume AI processing"""
        self._ai_paused = False
        self.logger.info("AI detection resumed")

    def set_resolution(self, resolution: tuple):
        """Change camera resolution at runtime"""
        if self.camera is None or not self.camera_initialized:
            self.logger.warning("Cannot set resolution - camera not initialized")
            return
        try:
            self.camera.stop()
            config = self._build_camera_config(resolution)
            self.camera.configure(config)
            self.camera.start()
            self._current_resolution = resolution
            self.logger.info(f"Camera resolution changed to {resolution}")
        except Exception as e:
            self.logger.error(f"Failed to set resolution: {e}")

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Get last captured frame for snapshots (thread-safe)"""
        with self._frame_lock:
            if self._last_frame is not None:
                return self._last_frame.copy()
        return None

    def get_last_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get last captured frame with its timestamp (thread-safe)

        Returns (frame, timestamp) tuple. If no frame, returns (None, 0)
        """
        with self._frame_lock:
            if self._last_frame is not None:
                return self._last_frame.copy(), self._last_frame_time
        return None, 0

    def get_last_frame_age(self) -> float:
        """Get age of last frame in seconds"""
        with self._frame_lock:
            if self._last_frame_time > 0:
                return time.time() - self._last_frame_time
        return float('inf')

    def _detect_aruco_markers(self, frame) -> List[Tuple[int, float, float]]:
        """Detect ArUco markers in frame with quality filtering.
        Requires markers to appear in consecutive frames to reduce false positives."""
        markers = []
        seen_this_frame = set()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    mid = int(marker_id)
                    corner = corners[i][0]

                    # Minimum marker size filter (perimeter in pixels)
                    perimeter = cv2.arcLength(corner.reshape(-1, 1, 2), True)
                    if perimeter < 60:  # Reject tiny detections (noise)
                        self.logger.debug(f"ARUCO_REJECTED: ID {mid} too small (perimeter={perimeter:.0f}px)")
                        continue

                    seen_this_frame.add(mid)
                    cx = float(np.mean(corner[:, 0]))
                    cy = float(np.mean(corner[:, 1]))

                    # Consecutive frame validation
                    self._aruco_consecutive[mid] = self._aruco_consecutive.get(mid, 0) + 1
                    if self._aruco_consecutive[mid] < self._aruco_min_frames:
                        continue  # Not confirmed yet

                    markers.append((mid, cx, cy))

                    dog_name = DOG_MARKERS.get(mid, 'unknown')
                    self.logger.info(f"[ARUCO] Detected marker ID: {mid}, mapped to dog: {dog_name} at ({cx:.0f}, {cy:.0f})")

            # Reset counters for markers NOT seen this frame
            for mid in list(self._aruco_consecutive.keys()):
                if mid not in seen_this_frame:
                    del self._aruco_consecutive[mid]

        except Exception as e:
            self.logger.error(f"ArUco detection error: {e}")

        return markers

    def is_initialized(self) -> bool:
        """Check if detector service is initialized (camera required, AI optional)"""
        return self.camera_initialized

    def start_detection(self) -> bool:
        """Start detection loop

        Starts frame capture loop. AI processing is optional - loop handles
        modes where AI isn't needed (IDLE/MANUAL just capture for WebRTC).
        """
        if not self.camera_initialized:
            self.logger.error("Camera not initialized - cannot start detection")
            return False

        if self.running:
            self.logger.warning("Detection already running")
            return True

        self.running = True
        self._stop_event.clear()

        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="DetectionService"
        )
        self.detection_thread.start()

        self.logger.info("Detection started")
        publish_vision_event('detection_started', {}, 'detector_service')
        return True

    def stop_detection(self) -> None:
        """Stop detection loop"""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        self.logger.info("Detection stopped")
        publish_vision_event('detection_stopped', {}, 'detector_service')

    def _detection_loop(self) -> None:
        """Main detection loop - runs in all operational modes"""
        self.logger.info("Detection loop started")
        self.last_fps_time = time.time()
        self.frame_count = 0
        self._loop_iteration = 0

        while not self._stop_event.is_set():
            try:
                self._loop_iteration += 1

                # Log diagnostic every 100 iterations (or first 5 for debugging)
                if self._loop_iteration <= 5 or self._loop_iteration % 100 == 0:
                    self.logger.debug(f"Detection loop alive: iter={self._loop_iteration}, camera_init={self.camera_initialized}, paused={self._camera_paused}, mode={self.state.get_mode()}")

                # Check mode to determine behavior
                # - SHUTDOWN/EMERGENCY: Stop everything
                # - MANUAL/IDLE: Capture frames for WebRTC, NO AI processing (high res)
                # - SILENT_GUARDIAN/COACH: Full AI processing (640x640)
                current_mode = self.state.get_mode()

                # Modes that completely pause detection
                if current_mode in [SystemMode.SHUTDOWN, SystemMode.EMERGENCY]:
                    time.sleep(0.1)
                    continue

                # Periodic camera retry if not initialized
                if not self.camera_initialized and not self._camera_paused:
                    now = time.time()
                    if now - self._last_camera_retry_time >= self._camera_retry_interval:
                        self._last_camera_retry_time = now
                        self.logger.info("Attempting periodic camera reinitialization...")
                        if self._initialize_camera():
                            self.logger.info("Camera reinitialized successfully on retry")
                            self.state.update_hardware(camera_initialized=True)
                        else:
                            self.logger.warning(
                                f"Camera retry failed. Next attempt in {self._camera_retry_interval}s. "
                                f"Reason: {self._camera_error_reason}"
                            )
                    time.sleep(0.5)
                    continue

                # Capture frame from camera (always, for WebRTC streaming)
                if self._loop_iteration <= 5:
                    self.logger.debug("About to capture frame...")
                frame = self._capture_frame()
                if frame is None:
                    self.logger.warning(f"No frame captured (camera_init={self.camera_initialized}), waiting...")
                    time.sleep(0.1)
                    continue

                if self._loop_iteration <= 5:
                    self.logger.debug(f"Frame captured: {frame.shape}")

                # Determine if AI processing should run
                # Full AI: COACH, MISSION (detection + pose + behavior)
                # Lightweight AI: SILENT_GUARDIAN (detection only, no behavior classification)
                # No AI: IDLE, MANUAL (just frame capture for WebRTC)
                full_ai_modes = [SystemMode.COACH, SystemMode.MISSION]
                # Night Sentry uses lightweight detection (dog detection only, no
                # behavior/pose) — same as Silent Guardian — so it can alert on
                # what the camera sees without the full pose pipeline cost.
                lightweight_ai_mode = current_mode in (SystemMode.SILENT_GUARDIAN, SystemMode.NIGHT_SENTRY)
                run_full_ai = current_mode in full_ai_modes

                dogs = []
                poses = []
                behaviors = []
                dog_assignments = {}
                aruco_markers = []

                if (run_full_ai or lightweight_ai_mode) and not self._ai_paused:
                    # Detect ArUco markers for dog identification
                    # Silent Guardian: skip ArUco (only needs bark detection, not dog ID)
                    if not lightweight_ai_mode:
                        aruco_markers = self._detect_aruco_markers(frame)

                    # Publish ArUco marker event for video recorder
                    if aruco_markers:
                        publish_vision_event('aruco_detected', {
                            'markers': aruco_markers,
                            'timestamp': time.time()
                        }, 'detector_service')

                    # Process frame with dog identification
                    # Silent Guardian: skip_behavior=True (no pose analysis, no geometric
                    # classifier, no temporal voting — frees CPU for emotion classifier)
                    result = self.ai.process_frame_with_dogs(
                        frame, aruco_markers, skip_behavior=lightweight_ai_mode
                    )
                    dogs = result.get('detections', [])
                    poses = result.get('poses', [])
                    behaviors = result.get('behaviors', [])
                    dog_assignments = result.get('dog_assignments', {})
                    dog_id_methods = result.get('dog_id_methods', {})
                else:
                    # No AI in IDLE/MANUAL modes - just capture frames for WebRTC
                    # Frame is already captured and stored via _capture_frame()
                    pass

                # Update FPS
                self.frame_count += 1
                self._frames_processed_total += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    self.logger.debug(f"Detection FPS: {self.current_fps:.1f}")

                # Track detections for heartbeat and failure categories
                if dogs:
                    self._heartbeat_detections += len(dogs)
                    if behaviors:
                        best = max(behaviors, key=lambda b: b.confidence)
                        self._heartbeat_last_class = best.behavior
                        self._heartbeat_last_confidence = best.confidence
                        self._last_detection_info = {
                            'class': best.behavior,
                            'confidence': round(best.confidence, 3),
                            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                        }
                        # Cat B: dog detected but behavior confidence below threshold
                        for b in behaviors:
                            if b.confidence < 0.5:  # Near-miss threshold
                                self._cat_b_misses += 1
                    else:
                        # Cat C: dog detected but behavior classifier produced nothing
                        if run_full_ai:
                            self._cat_c_fails += 1

                # R3.1: Heartbeat every 5 seconds
                if now - self._heartbeat_time >= 5.0:
                    if lightweight_ai_mode:
                        # SG mode: detection only, no behavior - don't show stale behavior label
                        self.logger.info(
                            f"[AI] Pipeline active (SG) | FPS: {self.current_fps:.1f} | "
                            f"Dogs last 5s: {self._heartbeat_detections}"
                        )
                    elif run_full_ai:
                        self.logger.info(
                            f"[AI] Pipeline active | FPS: {self.current_fps:.1f} | "
                            f"Detections last 5s: {self._heartbeat_detections} | "
                            f"Last: {self._heartbeat_last_class or 'none'} @ "
                            f"{self._heartbeat_last_confidence*100:.0f}%"
                        )
                    else:
                        self.logger.info(
                            f"[AI] Mode={current_mode.value} — inference paused | "
                            f"frame capture only | FPS: {self.current_fps:.1f}"
                        )
                    self._heartbeat_detections = 0
                    self._heartbeat_time = now

                # R3.4: 60-second failure stats
                if now - self._stats_window_start >= 60.0:
                    self.logger.info(
                        f"[AI] 60s stats — Pipeline errors: {self._cat_a_errors} | "
                        f"Below-threshold: {self._cat_b_misses} | "
                        f"Classification fails: {self._cat_c_fails}"
                    )
                    self._cat_a_errors = 0
                    self._cat_b_misses = 0
                    self._cat_c_fails = 0
                    self._stats_window_start = now

                # CRITICAL: Minimum loop interval to prevent Hailo driver exhaustion
                # Primary rate limiting is in AI controller, this is a safety backstop
                if lightweight_ai_mode:
                    # Silent Guardian: ~3 FPS — enough for dog presence/ArUco tracking,
                    # frees CPU cores for emotion classifier inference
                    time.sleep(0.300)
                elif run_full_ai:
                    # Full AI modes: ~20 FPS cap, leaves headroom for Hailo DMA
                    time.sleep(0.050)
                else:
                    # IDLE / MANUAL: no inference, frame loop only feeds WebRTC preview.
                    # ~5 FPS is smooth enough for app preview and cuts capture CPU ~75%.
                    time.sleep(0.200)

                # Process results
                if dogs:
                    self._process_detections(dogs, poses, behaviors, dog_assignments, dog_id_methods)
                else:
                    # No dogs detected - clear stale identities
                    self._clear_stale_identities(time.time(), set())
                    self.state.update_detection(
                        dogs_detected=0,
                        active_dog_id=None,
                        dog_name="",
                        id_method="",
                        current_behavior="",
                        behavior_confidence=0.0
                    )

                # Removed extra 10ms sleep — the 50ms sleep above is sufficient
                # and this was adding unnecessary latency to every detection cycle

            except Exception as e:
                self.logger.error(f"[AI] PIPELINE ERROR: {e}")
                self._cat_a_errors += 1
                time.sleep(0.1)

        self.logger.info("Detection loop stopped")

    def _process_detections(self, dogs, poses, behaviors, dog_assignments: Dict[int, str] = None, dog_id_methods: Dict[int, str] = None) -> None:
        """Process detection results and publish events

        Args:
            dogs: List of Detection dataclass objects
            poses: List of PoseKeypoints dataclass objects
            behaviors: List of BehaviorResult dataclass objects
            dog_assignments: Dict mapping detection index to dog name (from ArUco)
            dog_id_methods: Dict mapping detection index to identification method
        """
        if dog_assignments is None:
            dog_assignments = {}
        if dog_id_methods is None:
            dog_id_methods = {}

        # Rate limit detection events to prevent log spam
        current_time = time.time()

        # Update state with primary dog info
        num_dogs = len(dogs)
        # Get primary dog name - use cached identity if available
        aruco_name_this_frame = dog_assignments.get(0, "")
        if aruco_name_this_frame:
            # ARUCO detected this frame - will be cached in _get_stable_dog_identity
            primary_dog_name = aruco_name_this_frame
            primary_id_method = dog_id_methods.get(0, "aruco")
        elif 0 in self._dog_identity_cache:
            # Use cached identity
            primary_dog_name = self._dog_identity_cache[0]
            primary_id_method = "aruco_cached"
        else:
            primary_dog_name = ""
            primary_id_method = "unknown"

        display_name = primary_dog_name.capitalize() if primary_dog_name else "Dog"
        self.state.update_detection(
            dogs_detected=num_dogs,
            last_detection_time=current_time,
            dog_name=display_name,
            id_method=primary_id_method
        )
        should_publish_event = (current_time - self._last_detection_event_time) >= self._detection_event_interval

        # Track which dog indices are active this frame for stale identity cleanup
        active_indices = set(range(len(dogs)))

        # Process each dog (Detection dataclass)
        for i, dog in enumerate(dogs):
            # Get stable identity: dog_id is always "dog_0", dog_name is cached ARUCO name
            aruco_name_this_frame = dog_assignments.get(i)
            dog_id, dog_name = self._get_stable_dog_identity(i, aruco_name_this_frame, current_time)

            bbox = [dog.x1, dog.y1, dog.x2, dog.y2]
            center = dog.center

            # Publish dog detection event (rate limited to prevent spam)
            if should_publish_event:
                publish_vision_event('dog_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,  # Cached ARUCO name or None
                    'confidence': dog.confidence,
                    'bbox': bbox,
                    'center': list(center),
                    'timestamp': current_time
                }, 'detector_service')

                # Log to store (also rate limited)
                self.store.log_event('vision', 'dog_detected', 'detector_service', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'confidence': dog.confidence,
                    'bbox': bbox
                })

            # Update dog seen in store (always update, just don't spam logs)
            self.store.update_dog_seen(dog_id)

        # Clear identities for dogs that have left the frame
        self._clear_stale_identities(current_time, active_indices)

        # Update last event time if we published
        if should_publish_event and num_dogs > 0:
            self._last_detection_event_time = current_time

        # Process poses if available (PoseKeypoints dataclass) - rate limited
        if poses and should_publish_event:
            for i, pose in enumerate(poses):
                # Use stable identity (already cached from dog detection loop above)
                dog_id = f"dog_{i}"
                dog_name = self._dog_identity_cache.get(i)
                keypoints = pose.keypoints.tolist() if hasattr(pose.keypoints, 'tolist') else list(pose.keypoints)

                publish_vision_event('pose_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'keypoints': keypoints,
                    'confidence': pose.detection.confidence if pose.detection else 0.0,
                    'num_keypoints': len(keypoints),
                    'timestamp': current_time
                }, 'detector_service')

        # Process behaviors if available (BehaviorResult dataclass)
        if behaviors:
            for i, behavior in enumerate(behaviors):
                # Use stable identity (already cached from dog detection loop above)
                dog_id = f"dog_{i}"
                dog_name = self._dog_identity_cache.get(i)
                behavior_name = behavior.behavior
                confidence = behavior.confidence

                # Update state with primary dog's behavior
                if i == 0:
                    # Use proper id_method based on whether identity is from current frame or cache
                    if dog_id_methods.get(i):
                        id_method = dog_id_methods[i]
                    elif dog_name:
                        id_method = "aruco_cached"
                    else:
                        id_method = "unknown"
                    self.state.update_detection(
                        active_dog_id=dog_id,
                        dog_name=dog_name.capitalize() if dog_name else "",
                        id_method=id_method,
                        current_behavior=behavior_name,
                        behavior_confidence=confidence,
                        pose_stable=confidence > 0.7
                    )

                # Publish behavior event
                publish_vision_event('behavior_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'behavior': behavior_name,
                    'confidence': confidence,
                    'duration': getattr(behavior, 'duration', 0.0),
                    'timestamp': time.time()
                }, 'detector_service')

                # Update dog tracker with behavior for video overlay display
                # This bridges behavior data to video_track.py so it can show "sit 34%"
                # Works for both ArUco-identified dogs (by name) and unidentified (by index)
                if hasattr(self.ai, 'dog_tracker') and self.ai.dog_tracker:
                    self.ai.dog_tracker.update_dog_behavior(
                        dog_name, behavior_name, confidence, detection_idx=i
                    )

                # Log behavior to store
                self.store.log_event('vision', 'behavior_detected', 'detector_service', {
                    'dog_id': dog_id,
                    'behavior': behavior_name,
                    'confidence': confidence
                })

    def _get_stable_dog_identity(self, dog_index: int, aruco_name: Optional[str], current_time: float) -> Tuple[str, Optional[str]]:
        """Get stable dog identity with persistence across intermittent ARUCO detection.

        Args:
            dog_index: The spatial detection index (0, 1, 2, ...)
            aruco_name: Current frame's ARUCO-identified name (or None)
            current_time: Current timestamp

        Returns:
            Tuple of (dog_id, dog_name) where:
            - dog_id is always stable (dog_0, dog_1, etc.)
            - dog_name is the ARUCO name if known (current or cached), else None
        """
        dog_id = f"dog_{dog_index}"

        # Update last seen time for this dog index
        self._dog_last_seen[dog_index] = current_time

        # If ARUCO identifies the dog this frame, cache it
        # Filter out generic names like 'dog_0', 'dog_1' - those are NOT real ARUCO identifications
        is_real_aruco_name = (aruco_name and
                              aruco_name not in ['unknown', ''] and
                              not aruco_name.startswith('dog_'))
        if is_real_aruco_name:
            if dog_index not in self._dog_identity_cache or self._dog_identity_cache[dog_index] != aruco_name:
                self.logger.info(f"[IDENTITY] dog_{dog_index} identified as '{aruco_name}' - caching")
            self._dog_identity_cache[dog_index] = aruco_name
            # C3.1: Update state's ArUco dog for voice fallback chain
            try:
                from core.state import get_state
                get_state().update_aruco_dog(aruco_name)
            except Exception:
                pass
            return (dog_id, aruco_name)

        # No ARUCO this frame - check cache
        if dog_index in self._dog_identity_cache:
            cached_name = self._dog_identity_cache[dog_index]
            return (dog_id, cached_name)

        # No identification available
        return (dog_id, None)

    def _clear_stale_identities(self, current_time: float, active_indices: set):
        """Clear identity cache for dogs that have left the frame.

        Args:
            current_time: Current timestamp
            active_indices: Set of dog indices seen this frame
        """
        stale_indices = []
        for dog_index, last_seen in list(self._dog_last_seen.items()):
            # If dog wasn't seen this frame and timeout exceeded, clear it
            if dog_index not in active_indices:
                if current_time - last_seen > self._identity_persist_timeout:
                    stale_indices.append(dog_index)

        for idx in stale_indices:
            if idx in self._dog_identity_cache:
                name = self._dog_identity_cache.pop(idx)
                self.logger.info(f"[IDENTITY] Cleared cached identity for dog_{idx} (was '{name}')")
            self._dog_last_seen.pop(idx, None)

    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box"""
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        return [320, 240]  # Default center

    def get_status(self) -> Dict[str, Any]:
        """Get detector service status"""
        return {
            'initialized': self.ai_initialized,
            'running': self.running,
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'ai_status': self.ai.get_status() if self.ai else None
        }

    def get_camera_status(self) -> Dict[str, Any]:
        """Get camera-specific status for diagnostics"""
        # Check for available cameras
        available_cameras = []
        if PICAMERA_AVAILABLE:
            try:
                available_cameras = Picamera2.global_camera_info()
            except Exception as e:
                self.logger.warning(f"Could not get camera info: {e}")

        return {
            'initialized': self.camera_initialized,
            'paused': self._camera_paused,
            'available_cameras': available_cameras,
            'cameras_detected': len(available_cameras),
            'error_reason': self._camera_error_reason,
            'picamera_available': PICAMERA_AVAILABLE,
            'last_frame_age': self.get_last_frame_age() if self.camera_initialized else None,
            'current_resolution': self._current_resolution,
            'resolution_change_in_progress': self._resolution_change_in_progress
        }

    def request_camera_reinitialize(self) -> Dict[str, Any]:
        """Request camera reinitialization (public API method)"""
        self.logger.info("Camera reinitialize requested via API")
        self._camera_error_reason = None
        self._reinitialize_camera()
        return self.get_camera_status()

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_detection()
        # Clean up camera
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.logger.info("Camera closed")
            except Exception as e:
                self.logger.error(f"Camera cleanup error: {e}")
            self.camera = None
            self.camera_initialized = False
        if self.ai:
            self.ai.cleanup()
        self.logger.info("Detector service cleaned up")


# Global detector service instance
_detector_instance = None
_detector_lock = threading.Lock()

def get_detector_service() -> DetectorService:
    """Get the global detector service instance (singleton)"""
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = DetectorService()
    return _detector_instance