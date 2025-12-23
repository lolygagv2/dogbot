#!/usr/bin/env python3
"""
FastAPI server for TreatBot REST endpoints
Provides monitoring and control interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import logging
import threading
import cv2
import io
import time

# TreatBot imports
from config.settings import SystemSettings
from core.state import get_state, SystemMode
from core.store import get_store
from services.reward.dispenser import get_dispenser_service
from services.motion.motor import get_motor_service
from services.motion.pan_tilt import get_pantilt_service
from services.media.usb_audio import get_usb_audio_service
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from orchestrators.mode_fsm import get_mode_fsm
from orchestrators.mission_engine import get_mission_engine
from core.hardware.audio_controller import AudioController
from core.hardware.led_controller import LEDController, LEDMode
from config.settings import AudioFiles
import lgpio

# AI Detection imports
try:
    from core.ai_controller_3stage_fixed import AI3StageControllerFixed
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI detection not available")

# Direct GPIO control for blue LED to avoid conflicts
_gpio_handle = None
_gpio_lock = threading.Lock()
BLUE_LED_PIN = 25

def get_gpio_handle():
    global _gpio_handle
    with _gpio_lock:
        if _gpio_handle is None:
            try:
                _gpio_handle = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(_gpio_handle, BLUE_LED_PIN)
                logger.warning("🔵 Direct GPIO control initialized for blue LED")
            except Exception as e:
                logger.error(f"🔵 GPIO init failed: {e}")
                _gpio_handle = None
        return _gpio_handle

def blue_led_direct_control(state):
    """Direct GPIO control for blue LED"""
    try:
        handle = get_gpio_handle()
        if handle is not None:
            lgpio.gpio_write(handle, BLUE_LED_PIN, 1 if state else 0)
            logger.warning(f"🔵 Blue LED {'ON' if state else 'OFF'} via direct GPIO")
            return True
        return False
    except Exception as e:
        logger.error(f"🔵 Direct GPIO error: {e}")
        return False

# Direct NeoPixel control for API (separate from main LED service)
_neopixels = None
_led_controller = None
_neopixel_lock = threading.Lock()

def get_neopixels():
    """Get direct NeoPixel control for API"""
    global _neopixels
    with _neopixel_lock:
        if _neopixels is None:
            try:
                import board
                import neopixel
                _neopixels = neopixel.NeoPixel(
                    board.D10,  # GPIO 10 (Pin 19) - SPI MOSI
                    SystemSettings.NEOPIXEL_COUNT,
                    brightness=SystemSettings.NEOPIXEL_BRIGHTNESS,
                    auto_write=False,
                    pixel_order=neopixel.GRB
                )
                _neopixels.fill((0, 0, 0))
                _neopixels.show()
                logger.warning("🔗 Direct NeoPixel control initialized for API")
            except Exception as e:
                logger.error(f"🔗 Direct NeoPixel init failed: {e}")
                raise Exception(f"Cannot initialize NeoPixels: {e}")
        return _neopixels

class DirectNeoPixelController:
    def __init__(self):
        self.pixels = get_neopixels()
        self.current_mode = "off"
        self.animation_active = False
        self.animation_thread = None
        self.blue_is_on = False
        # Initialize blue LED GPIO
        self._blue_chip = None
        try:
            self._blue_chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._blue_chip, BLUE_LED_PIN, lgpio.SET_PULL_NONE)
            lgpio.gpio_write(self._blue_chip, BLUE_LED_PIN, 0)
            logger.info(f"🔵 Blue LED initialized on GPIO{BLUE_LED_PIN}")
        except Exception as e:
            logger.warning(f"🔵 Blue LED init skipped (may be claimed by main LED service): {e}")
            self._blue_chip = None

    def blue_on(self):
        """Turn blue LED on"""
        if not self._blue_chip:
            # Fallback to direct control
            return blue_led_direct_control(True)
        try:
            lgpio.gpio_write(self._blue_chip, BLUE_LED_PIN, 1)
            self.blue_is_on = True
            logger.warning("🔵 Blue LED: ON")
            return True
        except Exception as e:
            logger.error(f"🔵 Blue LED on error: {e}")
            return False

    def blue_off(self):
        """Turn blue LED off"""
        if not self._blue_chip:
            # Fallback to direct control
            return blue_led_direct_control(False)
        try:
            lgpio.gpio_write(self._blue_chip, BLUE_LED_PIN, 0)
            self.blue_is_on = False
            logger.warning("🔵 Blue LED: OFF")
            return True
        except Exception as e:
            logger.error(f"🔵 Blue LED off error: {e}")
            return False

    def stop_animation(self):
        """Stop any running animation forcefully - ONE AT A TIME"""
        self.animation_active = False
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.2)
        # Make sure it's really dead
        self.animation_thread = None
        logger.warning("🛑 Animation stopped")

    def set_mode(self, mode):
        """Set NeoPixel mode with safe patterns"""
        self.current_mode = mode.value if hasattr(mode, 'value') else mode
        logger.warning(f"🔗 Direct NeoPixel mode: {self.current_mode}")

        # Always stop animation first
        self.stop_animation()

        if self.current_mode == 'off':
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
        elif self.current_mode == 'idle':
            self.pixels.fill((30, 30, 30))
            self.pixels.show()
        elif self.current_mode == 'searching':
            self._start_safe_animation('pulse', (0, 0, 255))
        elif self.current_mode == 'dog_detected':
            self._start_safe_animation('sparkle', (0, 255, 0))
        elif self.current_mode == 'treat_launching':
            self._start_safe_animation('spin', (255, 255, 0))
        elif self.current_mode == 'error':
            self._start_safe_animation('pulse', (255, 0, 0))
        elif self.current_mode == 'charging':
            self._start_safe_animation('breathe', (255, 165, 0))
        elif self.current_mode == 'manual_rc':
            self._start_safe_animation('rainbow', None)
        elif self.current_mode == 'gradient_flow':
            self._start_safe_animation('gradient_flow', None)
        elif self.current_mode == 'chase':
            self._start_safe_animation('chase', (0, 255, 255))
        elif self.current_mode == 'fire':
            self._start_safe_animation('fire', None)
        else:
            self.pixels.fill((30, 30, 30))
            self.pixels.show()

        return True

    def _start_safe_animation(self, pattern_type, color):
        """Start a single safe animation thread"""
        import threading
        self.animation_active = True
        logger.warning(f"🎬 Starting animation: {pattern_type}")
        self.animation_thread = threading.Thread(
            target=self._safe_animation_loop,
            args=(pattern_type, color),
            daemon=True
        )
        self.animation_thread.start()
        logger.warning(f"🎬 Animation thread started: {self.animation_thread.is_alive()}")

    def _safe_animation_loop(self, pattern_type, color):
        """Single animation loop that handles all patterns safely"""
        import time, random, math

        step = 0
        while self.animation_active:
            try:
                if pattern_type == 'pulse':
                    brightness = abs(math.sin(step * 0.1)) * 0.8 + 0.2
                    dimmed = tuple(int(c * brightness) for c in color)
                    self.pixels.fill(dimmed)

                elif pattern_type == 'sparkle':
                    self.pixels.fill((0, 0, 0))
                    num_sparkles = max(3, len(self.pixels) // 25)
                    for _ in range(num_sparkles):
                        pixel = random.randint(0, len(self.pixels) - 1)
                        self.pixels[pixel] = color

                elif pattern_type == 'spin':
                    self.pixels.fill((0, 0, 0))
                    pos = step % len(self.pixels)
                    self.pixels[pos] = color

                elif pattern_type == 'breathe':
                    brightness = (math.sin(step * 0.05) + 1) * 0.5
                    dimmed = tuple(int(c * brightness) for c in color)
                    self.pixels.fill(dimmed)

                elif pattern_type == 'rainbow':
                    for i in range(len(self.pixels)):
                        hue = (step * 3 + i * 5) % 360
                        rgb = self._hsv_to_rgb(hue, 1.0, 0.3)
                        self.pixels[i] = rgb

                elif pattern_type == 'gradient_flow':
                    # Smooth flowing rainbow gradient
                    for i in range(len(self.pixels)):
                        hue = (step * 2 + (i * 360 / len(self.pixels))) % 360
                        self.pixels[i] = self._hsv_to_rgb(hue, 1.0, 0.7)

                elif pattern_type == 'chase':
                    # Comet with trailing fade
                    tail_length = 25
                    self.pixels.fill((0, 0, 0))
                    position = step % len(self.pixels)
                    for i in range(tail_length):
                        pixel_pos = (position - i) % len(self.pixels)
                        brightness = 1.0 - (i / tail_length)
                        self.pixels[pixel_pos] = tuple(int(c * brightness) for c in color)

                elif pattern_type == 'fire':
                    # Flickering fire effect
                    fire_colors = [(255, 0, 0), (255, 50, 0), (255, 100, 0), (255, 150, 0), (255, 200, 50)]
                    for i in range(len(self.pixels)):
                        flicker = random.random()
                        color_idx = int(flicker * (len(fire_colors) - 1))
                        brightness = 0.3 + flicker * 0.7
                        base = fire_colors[color_idx]
                        self.pixels[i] = tuple(int(c * brightness) for c in base)

                self.pixels.show()
                step += 1
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"🎬 Animation error: {e}")
                break
        logger.warning(f"🎬 Animation loop ended: {pattern_type}")

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h / 60.0
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q

        return (int(r * 255), int(g * 255), int(b * 255))

    def _start_animation(self, animation_func, *args):
        """Start animation in background thread"""
        import threading
        self.animation_active = True
        self.animation_thread = threading.Thread(target=animation_func, args=args)
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def _pulse_pattern(self, color):
        """Pulsing effect"""
        import time
        while self.animation_active:
            for brightness in range(0, 100, 10):
                if not self.animation_active: break
                factor = brightness / 100.0
                dimmed = tuple(int(c * factor) for c in color)
                self.pixels.fill(dimmed)
                self.pixels.show()
                time.sleep(0.05)
            for brightness in range(100, 0, -10):
                if not self.animation_active: break
                factor = brightness / 100.0
                dimmed = tuple(int(c * factor) for c in color)
                self.pixels.fill(dimmed)
                self.pixels.show()
                time.sleep(0.05)

    def _sparkle_pattern(self, color):
        """Random sparkle effect"""
        import time, random
        while self.animation_active:
            self.pixels.fill((0, 0, 0))
            for _ in range(5):
                if not self.animation_active: break
                pixel = random.randint(0, 74)
                self.pixels[pixel] = color
            self.pixels.show()
            time.sleep(0.2)

    def _spinning_pattern(self, color):
        """Spinning dot effect"""
        import time
        while self.animation_active:
            for i in range(len(self.pixels)):
                if not self.animation_active: break
                self.pixels.fill((0, 0, 0))
                self.pixels[i] = color
                self.pixels.show()
                time.sleep(0.02)  # Faster for longer strip

    def _breathing_pattern(self, color):
        """Slow breathing effect"""
        import time
        while self.animation_active:
            for brightness in range(0, 100, 5):
                if not self.animation_active: break
                factor = brightness / 100.0
                dimmed = tuple(int(c * factor) for c in color)
                self.pixels.fill(dimmed)
                self.pixels.show()
                time.sleep(0.1)
            for brightness in range(100, 0, -5):
                if not self.animation_active: break
                factor = brightness / 100.0
                dimmed = tuple(int(c * factor) for c in color)
                self.pixels.fill(dimmed)
                self.pixels.show()
                time.sleep(0.1)

    def _rainbow_pattern(self):
        """Rainbow color cycle"""
        import time
        hue = 0
        while self.animation_active:
            for i in range(len(self.pixels)):
                if not self.animation_active: break
                pixel_hue = (hue + i * 3) % 360  # Adjusted for longer strip
                color = self._hsv_to_rgb(pixel_hue, 1.0, 0.3)
                self.pixels[i] = color
            self.pixels.show()
            hue = (hue + 5) % 360  # Slower rotation for smoother effect
            time.sleep(0.05)

    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        h = h / 60.0
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q

        return (int(r * 255), int(g * 255), int(b * 255))

    def get_status(self):
        return {"mode": self.current_mode, "pixels_initialized": True, "animation_active": self.animation_active}

def get_led_controller():
    """Get LED controller - prefer main service's controller to avoid GPIO conflicts"""
    global _led_controller
    if _led_controller is None:
        # Try to use main LED service first (avoids GPIO conflicts)
        try:
            from services.media.led import get_led_service
            led_service = get_led_service()
            if led_service.led_initialized and led_service.led:
                logger.warning("🔗 Using main LED service controller")
                _led_controller = led_service.led
                return _led_controller
        except Exception as e:
            logger.warning(f"🔗 Main LED service not available: {e}")

        # Fallback to DirectNeoPixelController
        logger.warning("🏗️ Creating DirectNeoPixelController (fallback)")
        _led_controller = DirectNeoPixelController()
    return _led_controller

# Models
class ModeRequest(BaseModel):
    mode: str
    duration: Optional[float] = None

class MissionRequest(BaseModel):
    mission_name: str
    parameters: Optional[Dict[str, Any]] = None

class TreatRequest(BaseModel):
    dog_id: Optional[str] = None
    reason: str = "manual"
    count: int = 1

class SequenceRequest(BaseModel):
    sequence_name: str
    context: Optional[Dict[str, Any]] = None
    interrupt: bool = False

class MotorControlRequest(BaseModel):
    """Motor control for iPhone app"""
    left_speed: int  # -100 to 100
    right_speed: int  # -100 to 100
    duration: Optional[float] = None  # seconds, None = continuous

class PanTiltRequest(BaseModel):
    """Camera pan/tilt control"""
    pan: Optional[int] = None  # -90 to 270 degrees with extended PWM
    tilt: Optional[int] = None  # 0-180 degrees
    speed: Optional[int] = 5  # movement speed
    smooth: Optional[bool] = False  # use smooth movement

class JoystickRequest(BaseModel):
    """Virtual joystick input from app"""
    x: float  # -1.0 to 1.0 (left/right)
    y: float  # -1.0 to 1.0 (forward/back)

class EmergencyStopRequest(BaseModel):
    """Emergency stop all motors"""
    reason: str = "emergency_stop"

# USB Audio models
class AudioPlayRequest(BaseModel):
    filepath: str

class AudioVolumeRequest(BaseModel):
    volume: int  # 0-100

class AudioNumberRequest(BaseModel):
    number: int

class AudioSoundRequest(BaseModel):
    sound_name: str

# LED control models
class LEDColorRequest(BaseModel):
    color: str  # Color name or hex

class LEDBrightnessRequest(BaseModel):
    brightness: float  # 0.1 to 1.0

class LEDModeRequest(BaseModel):
    mode: str  # LEDMode values

class LEDAnimationRequest(BaseModel):
    animation: str  # spinning_dot, pulse_color, rainbow_cycle
    color: Optional[str] = "blue"
    delay: Optional[float] = 0.05
    steps: Optional[int] = 20

class LEDCustomColorRequest(BaseModel):
    red: int    # 0-255
    green: int  # 0-255
    blue: int   # 0-255

# FastAPI app
app = FastAPI(
    title="WIM-Z Robot API",
    description="REST API and WebSocket server for WIM-Z robot control",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web dashboard
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Global camera instance for video streaming
_camera = None
_camera_lock = threading.Lock()

# Global AI detection instance
_ai_controller = None
_ai_lock = threading.Lock()
ai_detection_enabled = False  # Control flag for AI overlays

def get_camera():
    """Get or initialize camera instance"""
    global _camera
    with _camera_lock:
        if _camera is None and CAMERA_AVAILABLE:
            try:
                _camera = Picamera2()
                _camera.configure(_camera.create_preview_configuration(main={"size": (640, 480)}))
                _camera.start()
                logger.info("Camera initialized for streaming")
            except Exception as e:
                logger.error(f"Failed to initialize camera: {e}")
                _camera = None
        return _camera

def cleanup_camera():
    """Clean up camera resources"""
    global _camera
    with _camera_lock:
        if _camera:
            try:
                _camera.stop()
                _camera.close()
                logger.info("Camera cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up camera: {e}")
            _camera = None

def get_ai_controller():
    """Get or initialize AI controller instance"""
    global _ai_controller
    with _ai_lock:
        if _ai_controller is None and AI_AVAILABLE:
            try:
                _ai_controller = AI3StageControllerFixed()
                if _ai_controller.initialize():
                    logger.info("AI Controller initialized for detection overlays")
                else:
                    logger.error("AI Controller failed to initialize")
                    _ai_controller = None
            except Exception as e:
                logger.error(f"Failed to initialize AI controller: {e}")
                _ai_controller = None
        return _ai_controller

def cleanup_ai():
    """Clean up AI controller resources"""
    global _ai_controller
    with _ai_lock:
        if _ai_controller:
            try:
                # AI controller doesn't have explicit cleanup method, just release reference
                logger.info("AI controller cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up AI controller: {e}")
            _ai_controller = None

def draw_detection_overlays(frame, detections, poses=None, behaviors=None):
    """Draw AI detection overlays on frame"""
    if not detections:
        return frame

    annotated = frame.copy()

    # Draw detections
    for i, det in enumerate(detections):
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"Dog {i+1}: {det.confidence:.2f}",
                   (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw poses if available
    if poses:
        for pose in poses:
            keypoints = pose.keypoints
            det = pose.detection
            scale_x = det.width / 640
            scale_y = det.height / 640

            for kpt_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    x_px = int(det.x1 + x * scale_x)
                    y_px = int(det.y1 + y * scale_y)
                    cv2.circle(annotated, (x_px, y_px), 3, (0, 0, 255), -1)

    # Draw behaviors if available
    if behaviors:
        for i, behavior in enumerate(behaviors):
            if behavior.behavior:
                y_offset = 30 + (i * 20)
                cv2.putText(annotated, f"Behavior: {behavior.behavior} ({behavior.confidence:.2f})",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Add detection count
    cv2.putText(annotated, f"Detections: {len(detections)}", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated

# Logging
logger = logging.getLogger('TreatBotAPI')

# Cleanup handler
# Import WebSocket server
from api.ws import get_websocket_server

# Camera for video streaming
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    logger.warning("Picamera2 not available - video streaming disabled")

import atexit

def cleanup_hardware():
    """Clean up hardware resources on exit"""
    global _led_controller
    if _led_controller:
        logger.info("Cleaning up LED controller...")
        _led_controller.cleanup()
        _led_controller = None
    cleanup_camera()
    cleanup_ai()

atexit.register(cleanup_hardware)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "WIM-Z Robot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "dashboard": "/dashboard",
            "video_feed": "/video/feed",
            "websocket": "/ws",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the web dashboard"""
    try:
        with open("api/static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/video/feed")
async def video_feed():
    """MJPEG video stream endpoint"""
    if not CAMERA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Camera not available")

    def generate_mjpeg():
        camera = get_camera()
        if camera is None:
            return

        frame_time = 1/30  # Target 30 FPS
        last_frame_time = time.time()

        try:
            while True:
                current_time = time.time()

                # Skip frames if we're falling behind to prevent latency buildup
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_time:
                    time.sleep(frame_time - time_since_last)

                # Capture frame
                frame = camera.capture_array()
                last_frame_time = time.time()

                # Process AI detection overlays if enabled
                if ai_detection_enabled:
                    ai = get_ai_controller()
                    if ai is not None:
                        try:
                            detections, poses, behaviors = ai.process_frame(frame)
                            frame = draw_detection_overlays(frame, detections, poses, behaviors)
                        except Exception as e:
                            logger.debug(f"AI processing error (continuing without overlay): {e}")

                # Convert to JPEG with optimized settings for speed
                _, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 50,  # Lower quality for speed
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Enable optimization
                ])
                frame_bytes = buffer.tobytes()

                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logger.error(f"Video streaming error: {e}")
            return

    return StreamingResponse(generate_mjpeg(),
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/ai/detection/status")
async def get_ai_detection_status():
    """Get AI detection overlay status"""
    global ai_detection_enabled
    return {
        "enabled": ai_detection_enabled,
        "ai_available": AI_AVAILABLE,
        "ai_initialized": _ai_controller is not None
    }

@app.post("/ai/detection/enable")
async def enable_ai_detection():
    """Enable AI detection overlays on video feed"""
    global ai_detection_enabled
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI detection not available")

    ai_detection_enabled = True
    # Initialize AI controller if not already done
    get_ai_controller()

    return {"status": "enabled", "message": "AI detection overlays enabled"}

@app.post("/ai/detection/disable")
async def disable_ai_detection():
    """Disable AI detection overlays on video feed"""
    global ai_detection_enabled
    ai_detection_enabled = False
    return {"status": "disabled", "message": "AI detection overlays disabled"}

@app.get("/ai/detection/latest")
async def get_latest_detection():
    """Get latest AI detection results"""
    global _ai_controller, ai_detection_enabled

    if not AI_AVAILABLE or not ai_detection_enabled:
        return {
            "enabled": ai_detection_enabled,
            "detections": [],
            "poses": [],
            "behaviors": [],
            "timestamp": time.time()
        }

    ai = get_ai_controller()
    if ai is None:
        return {
            "enabled": ai_detection_enabled,
            "detections": [],
            "poses": [],
            "behaviors": [],
            "timestamp": time.time(),
            "error": "AI controller not available"
        }

    # For now, return empty data - this would need integration with the AI processing pipeline
    # In a production system, this would maintain a recent detection cache
    return {
        "enabled": ai_detection_enabled,
        "detections": [],
        "poses": [],
        "behaviors": [],
        "timestamp": time.time(),
        "message": "Live detection data would be available here"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    ws_server = get_websocket_server()
    await ws_server.handle_websocket(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    state = get_state()

    return {
        "status": "healthy",
        "mode": state.get_mode().value,
        "emergency": state.is_emergency(),
        "uptime": state.get_mode_duration(),
        "timestamp": state.get_full_state()["timestamp"]
    }

# Mode endpoints
@app.get("/mode")
async def get_mode():
    """Get current system mode"""
    state = get_state()
    mode_fsm = get_mode_fsm()

    return {
        "current_mode": state.get_mode().value,
        "previous_mode": state.previous_mode.value,
        "time_in_mode": state.get_mode_duration(),
        "fsm_status": mode_fsm.get_status()
    }

@app.post("/mode/set")
async def set_mode(request: ModeRequest):
    """Set system mode"""
    try:
        mode = SystemMode(request.mode)
        mode_fsm = get_mode_fsm()

        if request.duration:
            success = mode_fsm.set_mode_override(mode, request.duration)
        else:
            success = mode_fsm.force_mode(mode, "API request")

        if success:
            return {"success": True, "mode": mode.value}
        else:
            raise HTTPException(status_code=400, detail="Mode change failed")

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

@app.post("/mode/clear_override")
async def clear_mode_override():
    """Clear mode override"""
    mode_fsm = get_mode_fsm()
    mode_fsm.clear_override("API request")
    return {"success": True}

# Mission endpoints
@app.get("/missions/status")
async def get_mission_status():
    """Get current mission status"""
    mission_engine = get_mission_engine()
    return mission_engine.get_mission_status()

@app.post("/missions/start")
async def start_mission(request: MissionRequest):
    """Start a mission"""
    mission_engine = get_mission_engine()

    try:
        success = mission_engine.start_mission(
            request.mission_name,
            dog_id=request.parameters.get('dog_id') if request.parameters else None
        )

        if success:
            return {
                "success": True,
                "message": f"Mission '{request.mission_name}' started successfully",
                "mission_name": request.mission_name
            }
        else:
            return {
                "success": False,
                "message": f"Failed to start mission '{request.mission_name}'",
                "mission_name": request.mission_name
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mission start error: {str(e)}")

@app.post("/missions/stop")
async def stop_mission():
    """Stop the current mission"""
    mission_engine = get_mission_engine()
    success = mission_engine.stop_mission("user_requested")

    return {
        "success": success,
        "message": "Mission stopped" if success else "No mission to stop"
    }

@app.post("/missions/pause")
async def pause_mission():
    """Pause the current mission"""
    mission_engine = get_mission_engine()
    success = mission_engine.pause_mission()

    return {
        "success": success,
        "message": "Mission paused" if success else "No mission to pause"
    }

@app.post("/missions/resume")
async def resume_mission():
    """Resume the current mission"""
    mission_engine = get_mission_engine()
    success = mission_engine.resume_mission()

    return {
        "success": success,
        "message": "Mission resumed" if success else "No mission to resume"
    }

@app.get("/missions/available")
async def get_available_missions():
    """Get list of available missions"""
    mission_engine = get_mission_engine()
    return {
        "missions": mission_engine.get_available_missions()
    }

# Treat endpoints
@app.post("/treat/dispense")
async def dispense_treat(request: TreatRequest):
    """Dispense treats"""
    try:
        dispenser = get_dispenser_service()

        if request.count == 1:
            success = dispenser.dispense_treat(
                dog_id=request.dog_id,
                reason=request.reason
            )
        else:
            dispensed = dispenser.dispense_multiple(
                count=request.count,
                dog_id=request.dog_id,
                reason=request.reason
            )
            success = dispensed > 0

        return {
            "success": success,
            "treats_dispensed": request.count if success else 0,
            "dog_id": request.dog_id,
            "reason": request.reason
        }

    except Exception as e:
        logger.error(f"Treat dispense error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/treat/status")
async def get_treat_status():
    """Get treat dispenser status"""
    dispenser = get_dispenser_service()
    return dispenser.get_status()

@app.post("/treat/force_reward")
async def force_reward(dog_id: str = "api_test"):
    """Force a reward for testing"""
    try:
        reward_logic = get_reward_logic()
        success = reward_logic.force_reward(dog_id, "api_test")

        return {
            "success": success,
            "dog_id": dog_id,
            "type": "forced_reward"
        }

    except Exception as e:
        logger.error(f"Force reward error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sequence endpoints
@app.post("/sequence/execute")
async def execute_sequence(request: SequenceRequest):
    """Execute a sequence"""
    try:
        sequence_engine = get_sequence_engine()

        sequence_id = sequence_engine.execute_sequence(
            request.sequence_name,
            request.context or {},
            request.interrupt
        )

        if sequence_id:
            return {
                "success": True,
                "sequence_id": sequence_id,
                "sequence_name": request.sequence_name
            }
        else:
            raise HTTPException(status_code=400, detail="Sequence execution failed")

    except Exception as e:
        logger.error(f"Sequence execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sequence/status")
async def get_sequence_status():
    """Get sequence engine status"""
    sequence_engine = get_sequence_engine()
    return sequence_engine.get_status()

@app.post("/sequence/stop/{sequence_id}")
async def stop_sequence(sequence_id: str):
    """Stop a specific sequence"""
    sequence_engine = get_sequence_engine()
    success = sequence_engine.stop_sequence(sequence_id)

    return {
        "success": success,
        "sequence_id": sequence_id
    }

@app.post("/sequence/stop_all")
async def stop_all_sequences():
    """Stop all sequences"""
    sequence_engine = get_sequence_engine()
    count = sequence_engine.stop_all_sequences()

    return {
        "success": True,
        "sequences_stopped": count
    }

# Bark Detection endpoints
@app.get("/bark/status")
async def get_bark_detection_status():
    """Get bark detection service status"""
    bark_detector = get_bark_detector_service()
    return bark_detector.get_status()

@app.post("/bark/enable")
async def enable_bark_detection():
    """Enable bark detection"""
    bark_detector = get_bark_detector_service()
    bark_detector.set_enabled(True)
    return {"success": True, "enabled": True}

@app.post("/bark/disable")
async def disable_bark_detection():
    """Disable bark detection"""
    bark_detector = get_bark_detector_service()
    bark_detector.set_enabled(False)
    return {"success": True, "enabled": False}

@app.post("/bark/config")
async def configure_bark_detection(
    confidence_threshold: float = None,
    reward_emotions: list = None,
    audio_gain: float = None
):
    """Configure bark detection parameters"""
    bark_detector = get_bark_detector_service()

    updates = {}
    if confidence_threshold is not None:
        bark_detector.set_confidence_threshold(confidence_threshold)
        updates["confidence_threshold"] = confidence_threshold

    if reward_emotions is not None:
        bark_detector.set_reward_emotions(reward_emotions)
        updates["reward_emotions"] = reward_emotions

    if audio_gain is not None:
        # Would need to restart audio buffer with new gain
        updates["audio_gain"] = audio_gain
        updates["note"] = "Audio gain change requires restart"

    return {
        "success": True,
        "updated": updates
    }

@app.post("/bark/reset_stats")
async def reset_bark_statistics():
    """Reset bark detection statistics"""
    bark_detector = get_bark_detector_service()
    bark_detector.reset_statistics()
    return {"success": True, "message": "Statistics reset"}

# Telemetry endpoints
@app.get("/telemetry")
async def get_telemetry():
    """Get system telemetry"""
    state = get_state()
    store = get_store()

    return {
        "system": state.get_status_summary(),
        "hardware": state.hardware.to_dict(),
        "detection": state.detection.to_dict(),
        "recent_events": store.get_recent_events(10),
        "database_stats": store.get_database_stats()
    }

@app.get("/telemetry/hardware")
async def get_hardware_status():
    """Get hardware status"""
    state = get_state()
    return state.hardware.to_dict()

@app.get("/telemetry/detection")
async def get_detection_status():
    """Get detection status"""
    state = get_state()
    return state.detection.to_dict()

# Event endpoints
@app.get("/events/recent")
async def get_recent_events(limit: int = 50, event_type: Optional[str] = None):
    """Get recent events"""
    store = get_store()
    events = store.get_recent_events(limit, event_type)

    return {
        "events": events,
        "count": len(events),
        "limit": limit,
        "event_type": event_type
    }

@app.get("/events/stats")
async def get_event_stats():
    """Get event statistics"""
    store = get_store()
    return store.get_database_stats()

# Dog endpoints
@app.get("/dogs")
async def get_dogs():
    """Get all dogs"""
    store = get_store()
    dogs = store.get_dog_stats()

    return {
        "dogs": dogs,
        "count": len(dogs)
    }

@app.get("/dogs/{dog_id}")
async def get_dog(dog_id: str):
    """Get specific dog info"""
    store = get_store()
    reward_logic = get_reward_logic()

    dogs = store.get_dog_stats(dog_id)
    if not dogs:
        raise HTTPException(status_code=404, detail="Dog not found")

    dog = dogs[0]

    # Add reward stats
    reward_stats = reward_logic.get_dog_reward_stats(dog_id)

    return {
        "dog": dog,
        "reward_stats": reward_stats
    }

@app.get("/dogs/{dog_id}/rewards")
async def get_dog_rewards(dog_id: str, days: int = 7):
    """Get dog's reward history"""
    store = get_store()
    rewards = store.get_reward_history(dog_id, days)

    return {
        "dog_id": dog_id,
        "rewards": rewards,
        "count": len(rewards),
        "days": days
    }

# Manual Control endpoints
class ManualDriveRequest(BaseModel):
    direction: str
    speed: Optional[int] = 50
    duration: Optional[float] = 1.0

class KeyboardControlRequest(BaseModel):
    key: str

@app.post("/manual/drive")
async def manual_drive(request: ManualDriveRequest):
    """Manual vehicle control (RC car style)"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        success = motor_service.manual_drive(
            direction=request.direction,
            speed=request.speed,
            duration=request.duration
        )

        return {
            "success": success,
            "direction": request.direction,
            "speed": request.speed,
            "duration": request.duration
        }

    except Exception as e:
        logger.error(f"Manual drive error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/keyboard")
async def keyboard_control(request: KeyboardControlRequest):
    """Process keyboard input for manual control"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        success = motor_service.keyboard_control(request.key)

        return {
            "success": success,
            "key": request.key,
            "action": "processed" if success else "ignored"
        }

    except Exception as e:
        logger.error(f"Keyboard control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/emergency_stop")
async def manual_emergency_stop():
    """Emergency stop for manual control"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        motor_service.emergency_stop()

        return {
            "success": True,
            "message": "Manual control emergency stop"
        }

    except Exception as e:
        logger.error(f"Manual emergency stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/manual/status")
async def get_manual_control_status():
    """Get manual control status"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        return motor_service.get_status()

    except Exception as e:
        logger.error(f"Manual status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/mode/{mode}")
async def set_manual_mode(mode: str):
    """Set manual control mode (manual/auto/disabled)"""
    try:
        from services.motion.motor import get_motor_service, MovementMode
        motor_service = get_motor_service()

        mode_map = {
            'manual': MovementMode.MANUAL,
            'auto': MovementMode.AUTO,
            'disabled': MovementMode.DISABLED
        }

        if mode not in mode_map:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

        success = motor_service.set_movement_mode(mode_map[mode])

        return {
            "success": success,
            "mode": mode
        }

    except Exception as e:
        logger.error(f"Set manual mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# iPhone App Motor Control endpoints
@app.post("/motor/control")
async def motor_control(request: MotorControlRequest):
    """Direct motor control for iPhone app"""
    try:
        motor_service = get_motor_service()

        # Set individual motor speeds
        success = motor_service.set_motor_speeds(
            left_speed=request.left_speed,
            right_speed=request.right_speed,
            duration=request.duration
        )

        return {
            "success": success,
            "left_speed": request.left_speed,
            "right_speed": request.right_speed,
            "duration": request.duration
        }
    except Exception as e:
        logger.error(f"Motor control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/motor/joystick")
async def joystick_control(request: JoystickRequest):
    """Virtual joystick control from iPhone app"""
    try:
        motor_service = get_motor_service()

        # Convert joystick input to motor speeds
        # Forward/back is y, left/right is x
        left_speed = int((request.y + request.x) * 100)
        right_speed = int((request.y - request.x) * 100)

        # Clamp to valid range
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        success = motor_service.set_motor_speeds(
            left_speed=left_speed,
            right_speed=right_speed
        )

        return {
            "success": success,
            "joystick": {"x": request.x, "y": request.y},
            "motors": {"left": left_speed, "right": right_speed}
        }
    except Exception as e:
        logger.error(f"Joystick control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/motor/profile")
async def set_motor_profile(request: dict):
    """Set motor speed profile (safe/sport/turbo)"""
    try:
        from config.motor_profiles import get_profile_manager
        profile_mgr = get_profile_manager()

        profile_name = request.get('profile', 'sport').lower()
        success = profile_mgr.set_profile(profile_name)

        if success:
            status = profile_mgr.get_status()
            logger.info(f"Motor profile set to: {profile_name} ({status['max_voltage']}V)")
            return {
                "success": True,
                "profile": profile_name,
                "status": status
            }
        else:
            return {
                "success": False,
                "error": f"Invalid profile: {profile_name}",
                "valid_profiles": ["safe", "sport", "turbo"]
            }
    except Exception as e:
        logger.error(f"Profile change error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/motor/profile")
async def get_motor_profile():
    """Get current motor speed profile status"""
    try:
        from config.motor_profiles import get_profile_manager
        profile_mgr = get_profile_manager()
        status = profile_mgr.get_status()

        return {
            "success": True,
            "status": status,
            "profiles": {
                "safe": "6V - Standard speed, maximum motor life",
                "sport": "7V - 15% faster, good balance",
                "turbo": "8V - 33% faster, moderate wear (Xbox RT trigger)"
            }
        }
    except Exception as e:
        logger.error(f"Profile status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/motor/turbo")
async def toggle_turbo(request: dict):
    """Toggle turbo boost mode"""
    try:
        from config.motor_profiles import get_profile_manager
        profile_mgr = get_profile_manager()

        enable = request.get('enable', False)
        if enable:
            profile_mgr.enable_turbo_boost()
        else:
            profile_mgr.disable_turbo_boost()

        status = profile_mgr.get_status()
        return {
            "success": True,
            "turbo_active": status['turbo_active'],
            "status": status
        }
    except Exception as e:
        logger.error(f"Turbo toggle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/motor/stop")
async def motor_stop(request: Optional[EmergencyStopRequest] = None):
    """Emergency stop all motors"""
    try:
        motor_service = get_motor_service()
        motor_service.emergency_stop()

        reason = request.reason if request else "emergency_stop"
        logger.warning(f"Motor emergency stop: {reason}")

        return {
            "success": True,
            "message": f"Motors stopped: {reason}"
        }
    except Exception as e:
        logger.error(f"Motor stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Camera Control endpoints
@app.post("/camera/photo")
async def capture_photo_imx500():
    """Capture photo using IMX500 PCIe camera via rpicam-still"""
    import subprocess
    from datetime import datetime
    import os

    try:
        # Create captures directory if needed
        os.makedirs("/home/morgan/dogbot/captures", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = f"/home/morgan/dogbot/captures/{filename}"

        # Use rpicam-still for IMX500 camera
        # --width 4056 --height 3040 : Full resolution
        # --quality 95 : JPEG quality
        # --timeout 1000 : 1 second timeout
        # --nopreview : No preview window
        # --immediate : Capture immediately
        cmd = [
            "rpicam-still",
            "--width", "4056",
            "--height", "3040",
            "--quality", "95",
            "--timeout", "1000",
            "--nopreview",
            "--immediate",
            "-o", filepath
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.returncode == 0 and os.path.exists(filepath):
            # Get file size for verification
            file_size = os.path.getsize(filepath)
            logger.info(f"IMX500 photo captured: {filepath} ({file_size} bytes)")

            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "resolution": "4056x3040",
                "size_bytes": file_size,
                "method": "rpicam-still",
                "camera": "IMX500 PCIe"
            }
        else:
            error_msg = result.stderr if result.stderr else "Unknown rpicam-still error"
            raise Exception(f"rpicam-still failed: {error_msg}")

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Camera capture timeout")
    except Exception as e:
        logger.error(f"Photo capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/photo_opencv")
async def capture_photo_opencv():
    """Capture a high-resolution photo"""
    import cv2
    from datetime import datetime
    import os

    try:
        # Try different camera devices and backends
        # Try /dev/video0 and /dev/video19 which aren't held by PipeWire
        camera_options = [
            (0, cv2.CAP_V4L2),  # video0 with V4L2 backend
            (19, cv2.CAP_V4L2),  # video19 with V4L2 backend
            ("/dev/video0", cv2.CAP_V4L2),  # Direct device path
            ("/dev/video19", cv2.CAP_V4L2),  # Direct device path
            (0, cv2.CAP_ANY),  # Let OpenCV choose
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_ANY)
        ]

        for cam_source, backend in camera_options:
            try:
                logger.info(f"Trying camera: {cam_source} with backend {backend}")
                cap = cv2.VideoCapture(cam_source, backend)

                if not cap.isOpened():
                    continue

                # Check if we actually got the camera
                test_ret, test_frame = cap.read()
                if not test_ret or test_frame is None:
                    cap.release()
                    continue

                # Try to set high resolution (but don't fail if it doesn't work)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                # Get actual resolution
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Camera resolution: {actual_width}x{actual_height}")

                # Warm up camera with a few reads
                for _ in range(3):
                    cap.read()

                # Capture frame
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    # Save photo with high quality
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"photo_{timestamp}.jpg"
                    filepath = f"/home/morgan/dogbot/captures/{filename}"

                    # Create captures directory if needed
                    os.makedirs("/home/morgan/dogbot/captures", exist_ok=True)

                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.info(f"Photo saved: {filepath}")

                    return {
                        "success": True,
                        "filename": filename,
                        "filepath": filepath,
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "camera_source": str(cam_source)
                    }
            except Exception as cam_error:
                logger.warning(f"Camera {cam_source} failed: {cam_error}")
                continue

        raise Exception("No camera available for capture")

    except Exception as e:
        logger.error(f"Photo capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Camera Pan/Tilt Control endpoints
@app.post("/camera/pantilt")
async def camera_pantilt(request: PanTiltRequest):
    """Control camera pan/tilt servos"""
    try:
        pantilt_service = get_pantilt_service()

        # Initialize if not already done
        if not pantilt_service.servo_initialized:
            if not pantilt_service.initialize():
                raise HTTPException(status_code=500, detail="Failed to initialize servos")

        # Move servos with optional smooth movement
        if pantilt_service.servo:
            if request.pan is not None:
                # Use smooth parameter if provided
                pantilt_service.servo.set_camera_pan(request.pan, smooth=request.smooth)
                pantilt_service.current_pan = request.pan

            if request.tilt is not None:
                # Use smooth parameter if provided
                pantilt_service.servo.set_camera_pitch(request.tilt, smooth=request.smooth)
                pantilt_service.current_tilt = request.tilt

            return {
                "success": True,
                "pan": request.pan,
                "tilt": request.tilt,
                "current_position": {
                    "pan": pantilt_service.current_pan,
                    "tilt": pantilt_service.current_tilt
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Servo controller not available")

    except Exception as e:
        logger.error(f"Pan/tilt control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/camera/position")
async def get_camera_position():
    """Get current camera pan/tilt position"""
    try:
        pantilt_service = get_pantilt_service()
        position = pantilt_service.get_position()

        return {
            "success": True,
            "position": position
        }
    except Exception as e:
        logger.error(f"Get camera position error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/center")
async def camera_center():
    """Center camera to default position"""
    try:
        pantilt_service = get_pantilt_service()
        pantilt_service.center()

        return {
            "success": True,
            "message": "Camera centered",
            "position": pantilt_service.get_position()
        }
    except Exception as e:
        logger.error(f"Camera center error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# USB Audio Control endpoints
@app.get("/audio/status")
async def get_audio_status():
    """Get USB audio status"""
    try:
        audio = get_audio_controller()
        status = audio.get_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Audio status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/files")
async def get_audio_files():
    """Get list of available audio files"""
    try:
        # Get all AudioFiles class attributes
        files = {}
        for attr_name in dir(AudioFiles):
            if not attr_name.startswith('_'):
                filepath = getattr(AudioFiles, attr_name)
                if isinstance(filepath, str) and filepath.startswith('/'):
                    files[attr_name.lower()] = {
                        "name": attr_name,
                        "path": filepath,
                        "category": filepath.split('/')[1] if '/' in filepath else "unknown"
                    }

        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Audio files listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

_audio_controller = None

def get_audio_controller():
    """Get singleton audio controller"""
    global _audio_controller
    if _audio_controller is None:
        _audio_controller = AudioController()
    return _audio_controller

@app.post("/audio/play/file")
async def play_audio_file(request: AudioPlayRequest):
    """Play audio file by path using USB audio service"""
    try:
        usb_audio_service = get_usb_audio_service()
        result = usb_audio_service.play_file(request.filepath)

        return {
            "success": result.get("success", False),
            "filepath": request.filepath,
            "message": result.get("message", f"Playing {request.filepath}" if result.get("success") else "Failed to play file")
        }
    except Exception as e:
        logger.error(f"Audio play file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/play/number")
async def play_audio_number(request: AudioNumberRequest):
    """Play audio file by number"""
    try:
        audio = get_audio_controller()
        success = audio.play_file_by_number(request.number)

        return {
            "success": success,
            "number": request.number,
            "message": f"Playing file #{request.number}" if success else "Failed to play file"
        }
    except Exception as e:
        logger.error(f"Audio play number error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/play/sound")
async def play_audio_sound(request: AudioSoundRequest):
    """Play audio by sound name (from AudioFiles)"""
    try:
        audio = get_audio_controller()
        success = audio.play_sound(request.sound_name)

        return {
            "success": success,
            "sound_name": request.sound_name,
            "message": f"Playing {request.sound_name}" if success else f"Unknown sound: {request.sound_name}"
        }
    except Exception as e:
        logger.error(f"Audio play sound error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/volume")
async def set_audio_volume(request: AudioVolumeRequest):
    """Set USB audio volume (0-100)"""
    try:
        if not 0 <= request.volume <= 100:
            raise HTTPException(status_code=400, detail="Volume must be between 0 and 100")

        audio = get_audio_controller()
        success = audio.set_volume(request.volume)

        return {
            "success": success,
            "volume": request.volume,
            "message": f"Volume set to {request.volume}" if success else "Failed to set volume"
        }
    except Exception as e:
        logger.error(f"Audio volume error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/pause")
async def pause_audio():
    """Pause/resume audio playback"""
    try:
        audio = get_audio_controller()
        success = audio.play_pause_toggle()

        return {
            "success": success,
            "message": "Audio pause/resume toggled" if success else "Failed to toggle pause"
        }
    except Exception as e:
        logger.error(f"Audio pause error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/next")
async def next_audio():
    """Play next track"""
    try:
        audio = get_audio_controller()
        success = audio.play_next()

        return {
            "success": success,
            "message": "Playing next track" if success else "Failed to play next"
        }
    except Exception as e:
        logger.error(f"Audio next error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/previous")
async def previous_audio():
    """Play previous track"""
    try:
        audio = get_audio_controller()
        success = audio.play_previous()

        return {
            "success": success,
            "message": "Playing previous track" if success else "Failed to play previous"
        }
    except Exception as e:
        logger.error(f"Audio previous error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy relay endpoints - USB audio only now (no relay switching needed)
@app.post("/audio/relay/pi")
async def switch_to_pi_audio():
    """Legacy endpoint - USB audio is always active"""
    return {
        "success": True,
        "audio_path": "USB Audio",
        "message": "USB audio is the only audio source"
    }

@app.get("/audio/relay/status")
async def get_relay_status():
    """Get audio status - USB audio only"""
    return {
        "success": True,
        "relay_status": {"mode": "usb_audio", "active": True}
    }

@app.post("/audio/test")
async def test_audio_system():
    """Test USB audio system"""
    try:
        audio = get_audio_controller()
        success = audio.test_relay_switching()

        return {
            "success": success,
            "message": "Audio system test completed" if success else "Audio system test failed"
        }
    except Exception as e:
        logger.error(f"Audio test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LED Control endpoints
@app.get("/leds/status")
async def get_led_status():
    """Get LED system status"""
    try:
        # Use LED service instead of dummy controller
        from services.media.led import get_led_service
        led_service = get_led_service()
        if led_service.led_initialized and led_service.led:
            status = led_service.led.get_status()
        else:
            status = {"error": "LED service not initialized"}
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"LED status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leds/colors")
async def get_available_colors():
    """Get list of available LED colors"""
    try:
        # Standard colors available in the system
        colors = {
            'off': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'dim_white': (30, 30, 30),
            'warm_white': (255, 180, 120)
        }

        return {
            "success": True,
            "colors": colors,
            "count": len(colors)
        }
    except Exception as e:
        logger.error(f"LED colors error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leds/modes")
async def get_led_modes():
    """Get available LED modes"""
    try:
        modes = [mode.value for mode in LEDMode]
        return {
            "success": True,
            "modes": modes,
            "count": len(modes)
        }
    except Exception as e:
        logger.error(f"LED modes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/color")
async def set_led_color(request: LEDColorRequest):
    """Set LED to solid color"""
    try:
        leds = get_led_controller()
        success = leds.set_solid_color(request.color)

        return {
            "success": success,
            "color": request.color,
            "message": f"LEDs set to {request.color}" if success else f"Failed to set color {request.color}"
        }
    except Exception as e:
        logger.error(f"LED color error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/custom_color")
async def set_led_custom_color(request: LEDCustomColorRequest):
    """Set LED to custom RGB color"""
    try:
        if not all(0 <= val <= 255 for val in [request.red, request.green, request.blue]):
            raise HTTPException(status_code=400, detail="RGB values must be between 0 and 255")

        leds = get_led_controller()
        color_tuple = (request.red, request.green, request.blue)
        success = leds.set_solid_color(color_tuple)

        return {
            "success": success,
            "color": f"rgb({request.red}, {request.green}, {request.blue})",
            "rgb": color_tuple,
            "message": f"LEDs set to custom color" if success else "Failed to set custom color"
        }
    except Exception as e:
        logger.error(f"LED custom color error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/brightness")
async def set_led_brightness(request: LEDBrightnessRequest):
    """Set LED brightness (0.1 to 1.0)"""
    try:
        if not 0.1 <= request.brightness <= 1.0:
            raise HTTPException(status_code=400, detail="Brightness must be between 0.1 and 1.0")

        leds = get_led_controller()
        success = leds.set_neopixel_brightness(request.brightness)

        return {
            "success": success,
            "brightness": request.brightness,
            "message": f"Brightness set to {request.brightness}" if success else "Failed to set brightness"
        }
    except Exception as e:
        logger.error(f"LED brightness error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/mode")
async def set_led_mode(request: LEDModeRequest):
    """Set LED mode (with animations)"""
    try:
        # Validate mode
        valid_modes = [mode.value for mode in LEDMode]
        if request.mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}")

        # Use direct LED controller ONLY (old service is broken)
        leds = get_led_controller()
        led_mode = LEDMode(request.mode)
        leds.set_mode(led_mode)

        return {
            "success": True,
            "mode": request.mode,
            "message": f"LED mode set to {request.mode}"
        }
    except Exception as e:
        logger.error(f"LED mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/animation")
async def start_led_animation(request: LEDAnimationRequest):
    """Start custom LED animation"""
    try:
        leds = get_led_controller()

        # Map animation names to methods
        animations = {
            'spinning_dot': leds.spinning_dot,
            'pulse_color': leds.pulse_color,
            'rainbow_cycle': leds.rainbow_cycle
        }

        if request.animation not in animations:
            raise HTTPException(status_code=400, detail=f"Invalid animation. Available: {list(animations.keys())}")

        # Start the animation with parameters
        if request.animation == 'spinning_dot':
            leds.start_animation(animations[request.animation], request.color, request.delay)
        elif request.animation == 'pulse_color':
            leds.start_animation(animations[request.animation], request.color, request.steps, request.delay)
        elif request.animation == 'rainbow_cycle':
            leds.start_animation(animations[request.animation], request.delay)

        return {
            "success": True,
            "animation": request.animation,
            "color": request.color,
            "delay": request.delay,
            "steps": request.steps if request.animation == 'pulse_color' else None,
            "message": f"Started {request.animation} animation"
        }
    except Exception as e:
        logger.error(f"LED animation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/stop")
async def stop_led_animation():
    """Stop all LED animations"""
    try:
        leds = get_led_controller()
        leds.stop_animation()

        return {
            "success": True,
            "message": "LED animations stopped"
        }
    except Exception as e:
        logger.error(f"LED stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/blue/on")
async def blue_led_on():
    """Turn blue LED on"""
    logger.warning("🔵 BLUE LED ON API CALLED!")
    try:
        leds = get_led_controller()
        success = leds.blue_on()
        logger.warning(f"🔵 Blue LED ON result: {success}")

        return {
            "success": success,
            "message": "Blue LED turned on" if success else "Failed to turn blue LED on"
        }
    except Exception as e:
        logger.error(f"Blue LED on error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/blue/off")
async def blue_led_off():
    """Turn blue LED off"""
    try:
        leds = get_led_controller()
        success = leds.blue_off()
        logger.warning(f"🔵 Blue LED OFF result: {success}")

        return {
            "success": success,
            "message": "Blue LED turned off" if success else "Failed to turn blue LED off"
        }
    except Exception as e:
        logger.error(f"Blue LED off error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/off")
async def turn_leds_off():
    """Turn all LEDs off"""
    try:
        leds = get_led_controller()
        leds.set_mode(LEDMode.OFF)

        return {
            "success": True,
            "message": "All LEDs turned off"
        }
    except Exception as e:
        logger.error(f"LEDs off error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Emergency endpoints
@app.post("/emergency/stop")
async def emergency_stop():
    """Trigger emergency stop"""
    state = get_state()
    state.set_emergency("API emergency stop")

    # Also stop manual control
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()
        motor_service.emergency_stop()
    except:
        pass

    return {
        "success": True,
        "message": "Emergency stop triggered"
    }

@app.post("/emergency/clear")
async def clear_emergency():
    """Clear emergency state"""
    state = get_state()
    state.clear_emergency()

    return {
        "success": True,
        "message": "Emergency cleared"
    }

# Bark Detection endpoints
@app.get("/bark/status")
async def get_bark_status():
    """Get bark detection status and statistics"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        return bark_detector.get_status()
    except Exception as e:
        logger.error(f"Failed to get bark detector status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/enable")
async def enable_bark_detection():
    """Enable bark detection"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_enabled(True)
        return {"success": True, "message": "Bark detection enabled"}
    except Exception as e:
        logger.error(f"Failed to enable bark detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/disable")
async def disable_bark_detection():
    """Disable bark detection"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_enabled(False)
        return {"success": True, "message": "Bark detection disabled"}
    except Exception as e:
        logger.error(f"Failed to disable bark detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BarkThresholdRequest(BaseModel):
    threshold: float  # 0.0 to 1.0

@app.post("/bark/threshold")
async def set_bark_threshold(request: BarkThresholdRequest):
    """Set bark detection confidence threshold"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_confidence_threshold(request.threshold)
        return {"success": True, "threshold": request.threshold}
    except Exception as e:
        logger.error(f"Failed to set bark threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BarkEmotionsRequest(BaseModel):
    emotions: List[str]  # List of emotion names

@app.post("/bark/reward_emotions")
async def set_reward_emotions(request: BarkEmotionsRequest):
    """Set which bark emotions trigger rewards"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_reward_emotions(request.emotions)
        return {"success": True, "reward_emotions": request.emotions}
    except Exception as e:
        logger.error(f"Failed to set reward emotions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/reset_stats")
async def reset_bark_statistics():
    """Reset bark detection statistics"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.reset_statistics()
        return {"success": True, "message": "Bark statistics reset"}
    except Exception as e:
        logger.error(f"Failed to reset bark statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time control (iPhone app)
@app.websocket("/ws/control")
async def websocket_control(websocket: WebSocket):
    """WebSocket endpoint for real-time robot control from iPhone app"""
    await websocket.accept()
    motor_service = None
    pantilt_service = None

    try:
        # Get services
        motor_service = get_motor_service()
        pantilt_service = get_pantilt_service()

        logger.info("WebSocket control connection established")

        while True:
            # Receive control commands
            data = await websocket.receive_json()

            command = data.get("command")

            if command == "motor":
                # Motor control
                left = data.get("left", 0)
                right = data.get("right", 0)
                motor_service.set_motor_speeds(left, right)

                await websocket.send_json({
                    "type": "motor_ack",
                    "left": left,
                    "right": right
                })

            elif command == "joystick":
                # Virtual joystick
                x = data.get("x", 0)
                y = data.get("y", 0)

                # Convert to motor speeds
                left = int((y + x) * 100)
                right = int((y - x) * 100)
                left = max(-100, min(100, left))
                right = max(-100, min(100, right))

                motor_service.set_motor_speeds(left, right)

                await websocket.send_json({
                    "type": "joystick_ack",
                    "motors": {"left": left, "right": right}
                })

            elif command == "camera":
                # Camera pan/tilt
                pan = data.get("pan")
                tilt = data.get("tilt")

                if pan is not None:
                    pantilt_service.set_pan(pan)
                if tilt is not None:
                    pantilt_service.set_tilt(tilt)

                await websocket.send_json({
                    "type": "camera_ack",
                    "position": pantilt_service.get_position()
                })

            elif command == "stop":
                # Emergency stop
                motor_service.emergency_stop()
                await websocket.send_json({
                    "type": "stop_ack",
                    "message": "Motors stopped"
                })

            elif command == "treat":
                # Dispense treat
                dispenser = get_dispenser_service()
                dispenser.dispense_treat(reason="websocket_command")

                await websocket.send_json({
                    "type": "treat_ack",
                    "message": "Treat dispensed"
                })

            elif command == "ping":
                # Keep-alive ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown command: {command}"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket control connection closed")
        if motor_service:
            motor_service.emergency_stop()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        await websocket.close()

# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    state = get_state()

    # Get status from all major components
    status = {
        "state": state.get_full_state(),
        "mode_fsm": get_mode_fsm().get_status(),
        "sequence_engine": get_sequence_engine().get_status(),
        "reward_logic": get_reward_logic().get_status()
    }

    # Add service status if available
    try:
        from services.perception.detector import get_detector_service
        from services.perception.bark_detector import get_bark_detector_service
        from services.motion.pan_tilt import get_pantilt_service
        from services.reward.dispenser import get_dispenser_service
        from services.media.usb_audio import get_usb_audio_service
        from services.media.led import get_led_service

        status["services"] = {
            "detector": get_detector_service().get_status(),
            "bark_detector": get_bark_detector_service().get_status(),
            "pantilt": get_pantilt_service().get_status(),
            "dispenser": get_dispenser_service().get_status(),
            "usb_audio": {"initialized": get_usb_audio_service().is_initialized, "volume": get_usb_audio_service().current_volume},
            "led": get_led_service().get_status()
        }

    except Exception as e:
        logger.warning(f"Could not get service status: {e}")
        status["services"] = {"error": str(e)}

    return status

# ============================================================================
# Audio Control Endpoints
# ============================================================================

@app.post("/audio/play")
async def play_audio(request: dict):
    """Play audio track via USB audio"""
    try:
        usb_audio_service = get_usb_audio_service()

        track = request.get("track")
        name = request.get("name", f"Track {track}")

        if track is None:
            raise HTTPException(status_code=400, detail="Track number required")

        # Play using USB audio - map track to appropriate file
        # For now, play a default test file
        success = usb_audio_service.play_file("0001.mp3", "talks")

        return {
            "success": success,
            "track": track,
            "name": name
        }
    except Exception as e:
        logger.error(f"Audio play error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/audio/play_file")
async def play_audio_file(request: dict):
    """Play audio file by path via USB audio"""
    try:
        usb_audio_service = get_usb_audio_service()

        file_path = request.get("path")
        name = request.get("name", file_path)

        if file_path is None:
            raise HTTPException(status_code=400, detail="File path required")

        # Parse file path to extract filename and folder
        # Expected format: /talks/0001.mp3 or /02/0020.mp3
        if file_path.startswith('/'):
            parts = file_path.strip('/').split('/')
            if len(parts) >= 2:
                folder = parts[0]
                filename = parts[1]
                success = usb_audio_service.play_file(filename, folder)
                if success:
                    logger.info(f"Playing audio file: {name} ({file_path})")
            else:
                success = False
        else:
            success = False

        return {
            "success": success,
            "path": file_path,
            "name": name
        }
    except Exception as e:
        logger.error(f"Audio play file error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/audio/stop")
async def stop_audio():
    """Stop audio playback"""
    try:
        usb_audio_service = get_usb_audio_service()
        success = usb_audio_service.stop()
        return {"success": success}
    except Exception as e:
        logger.error(f"Audio stop error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/audio/pause")
async def pause_audio():
    """Pause/resume audio playback"""
    try:
        usb_audio_service = get_usb_audio_service()
        # USB audio service doesn't have pause, use stop instead
        success = usb_audio_service.stop()
        return {"success": success}
    except Exception as e:
        logger.error(f"Audio pause error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/audio/status")
async def get_audio_status():
    """Get audio system status"""
    try:
        usb_audio_service = get_usb_audio_service()
        return {
            "initialized": usb_audio_service.is_initialized,
            "playing": usb_audio_service.is_busy(),
            "volume": usb_audio_service.current_volume
        }
    except Exception as e:
        logger.error(f"Audio status error: {e}")
        return {"error": str(e)}

# ============== AUDIO RECORDING ENDPOINTS ==============
# For Xbox controller "record new talk" feature

import subprocess
import glob as glob_module
from datetime import datetime

# Recording state
_recording_state = {
    "temp_wav": "/tmp/wimz_recording.wav",
    "temp_mp3": "/tmp/wimz_recording.mp3",
    "has_pending": False,
    "record_time": None,
    "in_progress": False
}

@app.post("/audio/record/start")
async def start_audio_recording():
    """
    Start recording a new talk audio clip.
    Records for 2 seconds, converts to MP3, plays it back.
    Call /audio/record/confirm to save or /audio/record/cancel to discard.
    """
    # Check if already recording - reject duplicate requests
    if _recording_state["in_progress"]:
        logger.warning("🎙️ Recording already in progress - ignoring duplicate request")
        return {"success": False, "error": "Recording already in progress"}

    try:
        import time
        from services.perception.bark_detector import get_bark_detector_service

        # Mark recording as in progress IMMEDIATELY
        _recording_state["in_progress"] = True

        usb_audio_service = get_usb_audio_service()

        # Step 1: Pause bark detector to free the microphone
        logger.info("🎙️ Pausing bark detector for recording...")
        try:
            bark_detector = get_bark_detector_service()
            bark_detector.set_enabled(False)
            time.sleep(0.5)  # Wait for mic to be released
        except Exception as e:
            logger.warning(f"Could not pause bark detector: {e}")

        # Step 2: Set LED to fire mode (visual indicator)
        try:
            leds = get_led_controller()
            leds.set_mode('fire')
        except Exception as e:
            logger.warning(f"Could not set fire LED mode: {e}")

        # Step 3: Play start beep and wait for it to finish
        logger.info("🔔 Playing start beep...")
        usb_audio_service.play_file("/home/morgan/dogbot/VOICEMP3/songs/door_scan.mp3")
        time.sleep(1.0)  # Wait for beep to finish completely

        # Step 4: Record 2 seconds of audio
        wav_path = _recording_state["temp_wav"]
        mp3_path = _recording_state["temp_mp3"]

        # Remove old temp files
        for f in [wav_path, mp3_path]:
            if os.path.exists(f):
                os.remove(f)

        logger.info("🎙️ RECORDING NOW - Speak for 2 seconds...")
        record_cmd = [
            'arecord', '-D', 'hw:2,0', '-f', 'S16_LE', '-r', '44100',
            '-c', '1', '-d', '2', wav_path
        ]
        result = subprocess.run(record_cmd, capture_output=True, timeout=5)

        # Re-enable bark detector regardless of recording success
        try:
            bark_detector = get_bark_detector_service()
            bark_detector.set_enabled(True)
            logger.info("🎙️ Bark detector re-enabled")
        except:
            pass

        if result.returncode != 0:
            logger.error(f"Recording failed: {result.stderr.decode()}")
            _recording_state["in_progress"] = False
            # Set LED back to normal
            try:
                leds = get_led_controller()
                leds.set_mode('manual_rc')
            except Exception as e:
                logger.warning(f"Could not reset LED mode: {e}")
            return {"success": False, "error": "Recording failed - mic busy"}

        # Step 5: Convert WAV to MP3 using ffmpeg
        logger.info("🔄 Converting to MP3...")
        convert_cmd = [
            'ffmpeg', '-y', '-i', wav_path, '-acodec', 'libmp3lame',
            '-b:a', '128k', mp3_path
        ]
        result = subprocess.run(convert_cmd, capture_output=True, timeout=10)

        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr.decode()}")
            _recording_state["in_progress"] = False
            return {"success": False, "error": "MP3 conversion failed"}

        # Step 6: Play back the recording so user can hear it
        logger.info("🔊 Playing back your recording...")
        usb_audio_service.play_file(mp3_path)
        time.sleep(2.5)  # Wait for 2 second recording to play back

        # Step 7: Mark as pending confirmation (and clear in_progress)
        _recording_state["in_progress"] = False
        _recording_state["has_pending"] = True
        _recording_state["record_time"] = time.time()

        # Set LED to chase mode (waiting for confirmation)
        try:
            leds = get_led_controller()
            leds.set_mode('chase')
        except Exception as e:
            logger.warning(f"Could not set chase LED mode: {e}")

        logger.info("⏳ Waiting for confirmation - press START again within 10s to save")

        return {
            "success": True,
            "message": "Recording complete. Press START again within 10s to save.",
            "duration": 2,
            "pending": True
        }

    except Exception as e:
        logger.error(f"Recording error: {e}")
        _recording_state["in_progress"] = False
        # Re-enable bark detector on error
        try:
            from services.perception.bark_detector import get_bark_detector_service
            get_bark_detector_service().set_enabled(True)
        except:
            pass
        return {"success": False, "error": str(e)}

@app.post("/audio/record/confirm")
async def confirm_audio_recording():
    """
    Confirm and save the pending recording to VOICEMP3/talks folder.
    Generates a unique filename based on timestamp.
    """
    try:
        import time

        if not _recording_state["has_pending"]:
            return {"success": False, "error": "No pending recording to save"}

        # Check if within 10 second window
        elapsed = time.time() - _recording_state["record_time"]
        if elapsed > 10:
            _recording_state["has_pending"] = False
            return {"success": False, "error": "Confirmation window expired (10s)"}

        mp3_path = _recording_state["temp_mp3"]
        if not os.path.exists(mp3_path):
            return {"success": False, "error": "Recording file not found"}

        # Generate unique filename
        talks_dir = "/home/morgan/dogbot/VOICEMP3/talks"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Find next available number for today
        existing = glob_module.glob(os.path.join(talks_dir, f"custom_{timestamp[:8]}_*.mp3"))
        next_num = len(existing) + 1

        new_filename = f"custom_{timestamp}.mp3"
        new_path = os.path.join(talks_dir, new_filename)

        # Copy the file
        import shutil
        shutil.copy2(mp3_path, new_path)

        # Clear pending state
        _recording_state["has_pending"] = False

        # Play "good dog" to confirm save
        usb_audio_service = get_usb_audio_service()
        usb_audio_service.play_file("/home/morgan/dogbot/VOICEMP3/talks/good_dog.mp3")

        # Set LED back to manual_rc
        try:
            leds = get_led_controller()
            leds.set_mode('manual_rc')
        except Exception as e:
            logger.warning(f"Could not reset LED mode: {e}")

        logger.info(f"✅ Recording saved: {new_filename}")

        return {
            "success": True,
            "message": f"Recording saved as {new_filename}",
            "filename": new_filename,
            "path": new_path
        }

    except Exception as e:
        logger.error(f"Confirm recording error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/audio/record/cancel")
async def cancel_audio_recording():
    """Cancel and discard the pending recording."""
    try:
        _recording_state["has_pending"] = False

        # Remove temp files
        for f in [_recording_state["temp_wav"], _recording_state["temp_mp3"]]:
            if os.path.exists(f):
                os.remove(f)

        logger.info("🗑️ Recording discarded")
        return {"success": True, "message": "Recording discarded"}

    except Exception as e:
        logger.error(f"Cancel recording error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/audio/record/status")
async def get_recording_status():
    """Get current recording state."""
    import time

    has_pending = _recording_state["has_pending"]
    time_remaining = 0

    if has_pending and _recording_state["record_time"]:
        elapsed = time.time() - _recording_state["record_time"]
        time_remaining = max(0, 10 - elapsed)
        if time_remaining == 0:
            _recording_state["has_pending"] = False
            has_pending = False

    return {
        "has_pending": has_pending,
        "in_progress": _recording_state["in_progress"],
        "time_remaining": round(time_remaining, 1)
    }

# ============== END AUDIO RECORDING ENDPOINTS ==============

def create_app():
    """Create FastAPI app (for external use)"""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server"""
    logger.info(f"Starting TreatBot API server on {host}:{port}")

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    run_server(debug=True)