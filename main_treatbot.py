#!/usr/bin/env python3
"""
Main TreatBot orchestrator - THE definitive entry point
Unified system that coordinates all subsystems using the event architecture
"""

import sys
import os
import time
import signal
import logging
import threading
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core infrastructure
from core.bus import get_bus
from core.state import get_state, SystemMode
from core.store import get_store
from core.safety import get_safety_monitor

# Services
from services.perception.detector import get_detector_service
from services.perception.bark_detector import get_bark_detector_service
from services.motion.pan_tilt import get_pantilt_service
from services.motion.motor import get_motor_service
from services.reward.dispenser import get_dispenser_service
from services.media.sfx import get_sfx_service
from services.media.led import get_led_service
from services.media.usb_audio import get_usb_audio_service
from services.power.battery_monitor import get_battery_monitor
from services.control.bluetooth_esc import BluetoothESCController
from services.control.xbox_controller import get_xbox_service
from services.cloud.relay_client import get_relay_client, configure_relay_from_yaml
from services.streaming.webrtc import get_webrtc_service, configure_webrtc_from_yaml
from api.server import run_server

# Orchestrators
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from orchestrators.mode_fsm import get_mode_fsm

# Mode handlers
from modes.silent_guardian import get_silent_guardian_mode
from orchestrators.coaching_engine import get_coaching_engine


class TreatBotMain:
    """
    Main TreatBot orchestrator

    Coordinates all subsystems:
    - Core infrastructure (bus, state, store, safety)
    - Hardware services (detection, motion, rewards, media)
    - Orchestration (sequences, rewards, modes)
    """

    def __init__(self):
        self.logger = self._setup_logging()
        self.logger.info("=" * 60)
        self.logger.info("🤖 TREATBOT MAIN ORCHESTRATOR STARTING")
        self.logger.info("=" * 60)

        # Core systems
        self.bus = None
        self.state = None
        self.store = None
        self.safety = None

        # Services
        self.detector = None
        self.pantilt = None
        self.motor = None
        self.dispenser = None
        self.sfx = None
        self.led = None
        self.usb_audio = None  # For voice announcements
        self.bluetooth_controller = None
        self.xbox_controller = None
        self.api_server = None
        self.relay_client = None
        self.webrtc_service = None

        # Mode audio mappings - voice announcements for mode changes
        self.mode_audio_files = {
            'idle': '/home/morgan/dogbot/VOICEMP3/wimz/IdleMode.mp3',
            'silent_guardian': '/home/morgan/dogbot/VOICEMP3/wimz/SilentGuardianMode.mp3',
            'coach': '/home/morgan/dogbot/VOICEMP3/wimz/CoachMode.mp3',
            'manual': '/home/morgan/dogbot/VOICEMP3/wimz/ManualMode.mp3',
            'mission': '/home/morgan/dogbot/VOICEMP3/wimz/MissionMode.mp3',
        }
        self.startup_audio = '/home/morgan/dogbot/VOICEMP3/wimz/WimZOnline.mp3'
        self.low_battery_audio = '/home/morgan/dogbot/VOICEMP3/wimz/Wimz_lowpower.mp3'
        self.low_battery_announced = False  # Prevent repeat announcements

        # Orchestrators
        self.sequence_engine = None
        self.reward_logic = None
        self.mode_fsm = None

        # Mode handlers
        self.silent_guardian_mode = None
        self.coaching_engine = None

        # Main state
        self.running = False
        self.initialization_successful = False
        self.shutdown_requested = False

        # Main loop
        self.main_thread = None
        self._stop_event = threading.Event()

        # Relay event throttling (5-second minimum between events of same type)
        self._relay_event_times: Dict[str, float] = {}
        self._relay_throttle_interval = 5.0  # seconds

        # Performance tracking
        self.start_time = time.time()
        self.loop_count = 0
        self.last_heartbeat = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging system with file rotation"""
        from logging.handlers import RotatingFileHandler

        # Ensure logs directory exists
        log_dir = '/home/morgan/dogbot/logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'treatbot.log')

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler with rotation (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Capture more detail to file
        file_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        return logging.getLogger('TreatBotMain')

    def initialize(self) -> bool:
        """Initialize all subsystems in correct order"""
        try:
            self.logger.info("🔧 Initializing core infrastructure...")
            if not self._initialize_core():
                return False

            self.logger.info("🔧 Initializing hardware services...")
            if not self._initialize_services():
                return False

            self.logger.info("🔧 Initializing orchestrators...")
            if not self._initialize_orchestrators():
                return False

            self.logger.info("🔧 Starting subsystems...")
            if not self._start_subsystems():
                return False

            self.logger.info("🔧 Running startup sequence...")
            self._run_startup_sequence()

            self.initialization_successful = True
            self.logger.info("✅ TreatBot initialization complete!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False

    def _initialize_core(self) -> bool:
        """Initialize core infrastructure"""
        try:
            # Event bus
            self.bus = get_bus()
            self.logger.info("✅ Event bus ready")

            # State manager
            self.state = get_state()
            self.state.set_mode(SystemMode.IDLE, "System starting")
            self.logger.info("✅ State manager ready")

            # Data store
            self.store = get_store()
            self.logger.info("✅ Data store ready")

            # Safety monitor
            self.safety = get_safety_monitor()
            self.safety.add_emergency_callback(self._emergency_shutdown)
            self.logger.info("✅ Safety monitor ready")

            return True

        except Exception as e:
            self.logger.error(f"Core initialization failed: {e}")
            return False

    def _initialize_services(self) -> bool:
        """Initialize hardware services"""
        services_status = {}

        # Detector service
        try:
            self.detector = get_detector_service()
            services_status['detector'] = self.detector.initialize()
        except Exception as e:
            self.logger.error(f"Detector service failed: {e}")
            services_status['detector'] = False

        # Bark detector service
        try:
            self.bark_detector = get_bark_detector_service()
            services_status['bark_detector'] = self.bark_detector.initialize()
        except Exception as e:
            self.logger.error(f"Bark detector service failed: {e}")
            services_status['bark_detector'] = False

        # Pan/tilt service
        try:
            self.pantilt = get_pantilt_service()
            services_status['pantilt'] = self.pantilt.initialize()
            # NOTE: When Xbox controller is connected, it will take over camera control
            # The pantilt service will be paused automatically during manual control
        except Exception as e:
            self.logger.error(f"Pan/tilt service failed: {e}")
            services_status['pantilt'] = False

        # Motor bus - Direct hardware control for WebRTC and Xbox controller
        # Uses singleton pattern so won't conflict if Xbox controller also starts it
        try:
            from core.motor_command_bus import get_motor_bus
            self.motor_bus = get_motor_bus()
            if self.motor_bus.start():
                services_status['motor'] = True
                self.logger.info("✅ Motor bus initialized (direct hardware control for WebRTC)")
            else:
                services_status['motor'] = False
                self.logger.warning("⚠️ Motor bus failed to start")
        except Exception as e:
            self.logger.error(f"Motor bus failed: {e}")
            services_status['motor'] = False
            self.motor_bus = None

        # Dispenser service
        try:
            self.dispenser = get_dispenser_service()
            services_status['dispenser'] = self.dispenser.initialize()
        except Exception as e:
            self.logger.error(f"Dispenser service failed: {e}")
            services_status['dispenser'] = False

        # Audio service
        try:
            self.sfx = get_sfx_service()
            if self.sfx:
                services_status['sfx'] = self.sfx.initialize()
                if services_status['sfx']:
                    self.logger.info("✅ SFX service initialized successfully")
                else:
                    self.logger.warning("⚠️ SFX service failed to initialize")
            else:
                services_status['sfx'] = False
                self.logger.error("Failed to get SFX service instance")
        except Exception as e:
            self.logger.error(f"Audio service failed: {e}")
            services_status['sfx'] = False

        # LED service
        try:
            self.led = get_led_service()
            if self.led:
                services_status['led'] = self.led.initialize()
                if services_status['led']:
                    self.logger.info("✅ LED service initialized successfully")
                else:
                    self.logger.warning("⚠️ LED service failed to initialize")
            else:
                services_status['led'] = False
                self.logger.error("Failed to get LED service instance")
        except Exception as e:
            self.logger.error(f"LED service failed: {e}")
            services_status['led'] = False

        # USB Audio service (voice announcements)
        try:
            self.usb_audio = get_usb_audio_service()
            services_status['usb_audio'] = self.usb_audio.initialized
            if services_status['usb_audio']:
                self.logger.info("✅ USB Audio service initialized (voice announcements ready)")
            else:
                self.logger.warning("⚠️ USB Audio service failed to initialize")
        except Exception as e:
            self.logger.error(f"USB Audio service failed: {e}")
            services_status['usb_audio'] = False

        # Battery monitor service (ADS1115 on I2C)
        try:
            self.battery_monitor = get_battery_monitor()
            services_status['battery'] = self.battery_monitor.initialize()
            if services_status['battery']:
                self.battery_monitor.start_monitoring()
                status = self.battery_monitor.get_status()
                self.logger.info(f"✅ Battery monitor: {status['voltage']:.1f}V ({status['percentage']}%)")
            else:
                self.logger.warning("⚠️ Battery monitor failed to initialize")
        except Exception as e:
            self.logger.error(f"Battery monitor failed: {e}")
            services_status['battery'] = False

        # Bluetooth controller service - DISABLED (conflicts with Xbox controller)
        try:
            # self.bluetooth_controller = BluetoothESCController()
            # services_status['bluetooth'] = self.bluetooth_controller.initialize()
            services_status['bluetooth'] = False  # Disabled
            self.logger.info("🎮 Bluetooth ESC controller disabled (Xbox controller active)")
        except Exception as e:
            self.logger.error(f"Bluetooth controller failed: {e}")
            services_status['bluetooth'] = False

        # Xbox controller service
        try:
            self.xbox_controller = get_xbox_service()
            services_status['xbox_controller'] = self.xbox_controller.start()
            if services_status['xbox_controller']:
                self.logger.info("🎮 Xbox controller service started")
        except Exception as e:
            self.logger.error(f"Xbox controller service failed: {e}")
            services_status['xbox_controller'] = False

        # API Server service
        try:
            self.api_server = self._start_api_server()
            services_status['api_server'] = True
            self.logger.info("🌐 API server started on port 8000")
        except Exception as e:
            self.logger.error(f"API server failed: {e}")
            services_status['api_server'] = False

        # WebRTC and Cloud Relay services
        try:
            self._initialize_cloud_services()
            services_status['cloud_relay'] = self.relay_client is not None and self.relay_client.config.enabled
            services_status['webrtc'] = self.webrtc_service is not None
        except Exception as e:
            self.logger.error(f"Cloud/WebRTC services failed: {e}")
            services_status['cloud_relay'] = False
            services_status['webrtc'] = False

        # Check critical services
        # Note: detector is non-critical - Silent Guardian mode uses bark detection only
        # Coach mode will log warning if detector unavailable
        critical_services = ['dispenser']
        for service in critical_services:
            if not services_status.get(service, False):
                self.logger.error(f"Critical service failed: {service}")
                return False

        # Log service status
        for service, status in services_status.items():
            status_msg = "✅" if status else "⚠️"
            self.logger.info(f"{status_msg} {service}: {'Ready' if status else 'Failed'}")

        return True

    def _start_api_server(self) -> threading.Thread:
        """Start API server in background thread"""
        def run_api():
            try:
                run_server(host="0.0.0.0", port=8000, debug=False)
            except Exception as e:
                self.logger.error(f"API server error: {e}")

        api_thread = threading.Thread(target=run_api, daemon=True, name="APIServer")
        api_thread.start()
        return api_thread

    def _initialize_cloud_services(self):
        """Initialize WebRTC and cloud relay services"""
        import yaml

        # Load config
        config_path = '/home/morgan/dogbot/config/robot_config.yaml'
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
            config_dict = {}

        # Initialize WebRTC service
        webrtc_config = configure_webrtc_from_yaml(config_dict)
        self.webrtc_service = get_webrtc_service(webrtc_config)
        self.logger.info("✅ WebRTC service initialized")

        # Initialize cloud relay
        relay_config = configure_relay_from_yaml(config_dict)
        if relay_config.enabled:
            self.relay_client = get_relay_client(relay_config)
            # Wire up WebRTC service to relay client
            self.relay_client.set_webrtc_service(self.webrtc_service)
            self.logger.info(f"✅ Cloud relay configured (url={relay_config.relay_url})")
        else:
            self.logger.info("☁️ Cloud relay disabled in config")

    def _initialize_orchestrators(self) -> bool:
        """Initialize orchestration layer"""
        try:
            # Sequence engine
            self.sequence_engine = get_sequence_engine()
            self.logger.info("✅ Sequence engine ready")

            # Reward logic
            self.reward_logic = get_reward_logic()
            self.logger.info("✅ Reward logic ready")

            # Mode FSM
            self.mode_fsm = get_mode_fsm()
            self.logger.info("✅ Mode FSM ready")

            # Silent Guardian mode handler
            self.silent_guardian_mode = get_silent_guardian_mode()
            self.logger.info("✅ Silent Guardian mode handler ready")

            # Coaching engine
            self.coaching_engine = get_coaching_engine()
            self.logger.info("✅ Coaching engine ready")

            return True

        except Exception as e:
            self.logger.error(f"Orchestrator initialization failed: {e}")
            return False

    def _start_subsystems(self) -> bool:
        """Start all subsystems"""
        try:
            # Start safety monitoring
            self.safety.start_monitoring(interval=5.0)

            # Start mode FSM
            self.mode_fsm.start_fsm()

            # Start pan/tilt tracking
            if self.pantilt and hasattr(self.pantilt, 'servo_initialized'):
                if self.pantilt.servo_initialized:
                    self.pantilt.start_tracking()

            # Start AI detection - runs in ALL operational modes
            # Vision is a core perception layer, not mode-specific
            if self.detector.ai_initialized:
                self.detector.start_detection()
                self.logger.info("🧠 AI detection started (runs in all operational modes)")
                # Subscribe to detection events for LED feedback
                self.bus.subscribe('vision', self._on_detection_for_feedback)

            # Start bark detection if enabled
            if self.bark_detector.enabled:
                self.bark_detector.start()
                # Subscribe to bark events for feedback
                self.bus.subscribe('audio', self._on_bark_for_feedback)
                self.logger.info("🎤 Bark detection started")

            # Subscribe to mode changes to manage mode handlers
            self.state.subscribe('mode_change', self._on_mode_change)

            # Subscribe to controller events for mode transitions
            self.bus.subscribe('system', self._on_system_event)

            # Start Bluetooth controller if available
            if self.bluetooth_controller and self.bluetooth_controller.is_connected:
                self.bluetooth_controller.start()
                self.logger.info("🎮 Bluetooth controller active - Press START to enter MANUAL mode")
                self.state.set_mode(SystemMode.MANUAL, "Bluetooth controller ready")

            # Start cloud relay client if enabled
            if self.relay_client and self.relay_client.config.enabled:
                self.relay_client.start()
                self.logger.info("☁️ Cloud relay client started")
                # Subscribe to events for relay forwarding
                self.bus.subscribe('vision', self._forward_event_to_relay)
                self.bus.subscribe('audio', self._forward_event_to_relay)
                self.bus.subscribe('safety', self._forward_event_to_relay)
                self.bus.subscribe('system', self._forward_event_to_relay)
                self.bus.subscribe('cloud', self._handle_cloud_command)
                self.logger.info("☁️ Event forwarding to relay enabled")
                self.logger.info("☁️ Cloud command handler enabled")

            self.logger.info("✅ All subsystems started")
            return True

        except Exception as e:
            self.logger.error(f"Subsystem startup failed: {e}")
            return False

    def _on_detection_for_feedback(self, event) -> None:
        """Provide visual feedback for detection events (COACH mode only)"""
        try:
            current_mode = self.state.get_mode()

            # Skip LED feedback in Silent Guardian - LEDs reserved for meaningful events
            # (interventions, rewards handled by silent_guardian.py itself)
            if current_mode == SystemMode.SILENT_GUARDIAN:
                return

            # Skip in IDLE mode - system should be non-reactive
            if current_mode == SystemMode.IDLE:
                return

            # Skip LED feedback when Xbox controller is active
            if current_mode == SystemMode.MANUAL:
                return
            if self.xbox_controller and self.xbox_controller.is_connected:
                return

            # Only show debug LED patterns in COACH mode
            if current_mode == SystemMode.COACH:
                if event.subtype == 'dog_detected':
                    if self.led:
                        self.led.set_pattern('dog_detected')
                    self.logger.info(f"🐕 Dog detected: {event.data.get('dog_id', 'unknown')}")

                elif event.subtype == 'dog_lost':
                    if self.led:
                        self.led.set_pattern('searching')
                    self.logger.info("👀 Dog lost, searching...")

                elif event.subtype == 'pose':
                    behavior = event.data.get('behavior')
                    if behavior:
                        self.logger.info(f"🎯 Behavior detected: {behavior}")
                        if self.led:
                            self.led.pulse_color('yellow')

        except Exception as e:
            self.logger.error(f"Detection feedback error: {e}")

    def _on_bark_for_feedback(self, event) -> None:
        """Provide feedback for bark detection events (COACH mode only)"""
        try:
            current_mode = self.state.get_mode()

            # Skip in Silent Guardian - LED feedback handled by silent_guardian.py
            if current_mode == SystemMode.SILENT_GUARDIAN:
                return

            # Skip in IDLE mode - system should be non-reactive
            if current_mode == SystemMode.IDLE:
                return

            # Skip LED feedback when Xbox controller is active
            if current_mode == SystemMode.MANUAL:
                return
            if self.xbox_controller and self.xbox_controller.is_connected:
                return

            if event.subtype == 'bark_detected':
                emotion = event.data.get('emotion', 'unknown')
                confidence = event.data.get('confidence', 0)
                self.logger.info(f"🐕 Bark detected: {emotion} (conf: {confidence:.2f})")

            elif event.subtype == 'bark_rewarded':
                emotion = event.data.get('emotion', 'unknown')
                self.logger.info(f"🎁 Bark reward triggered for: {emotion}")

                # Celebration feedback - only for rewards in COACH mode
                if self.led and self.led.led_initialized:
                    self.led.set_pattern('celebration', 3.0)

        except Exception as e:
            self.logger.error(f"Bark feedback error: {e}")

    def _forward_event_to_relay(self, event) -> None:
        """Forward relevant events to cloud relay for app communication

        Applies 5-second throttling to detection and bark events to prevent spam.
        Mode, battery, and alert events are always forwarded immediately.
        """
        if not self.relay_client or not self.relay_client.connected:
            return

        try:
            from core.bus import EventType
            event_type = None
            event_data = {}
            should_throttle = False  # Whether to apply 5-second throttle

            # Map event types to relay event types
            if event.type == EventType.VISION:
                if event.subtype == 'dog_detected':
                    event_type = 'detection'
                    event_data = {
                        'detected': True,
                        'dog_id': event.data.get('dog_id'),
                        'confidence': event.data.get('confidence', 0),
                    }
                    should_throttle = True
                elif event.subtype == 'dog_lost':
                    event_type = 'detection'
                    event_data = {'detected': False}
                    should_throttle = True
                elif event.subtype == 'pose':
                    event_type = 'detection'
                    event_data = {
                        'detected': True,
                        'behavior': event.data.get('behavior'),
                        'confidence': event.data.get('confidence', 0),
                    }
                    should_throttle = True
                elif event.subtype == 'aruco_detected':
                    event_type = 'detection'
                    event_data = {
                        'detected': True,
                        'aruco_id': event.data.get('marker_id'),
                        'dog_name': event.data.get('dog_name'),
                    }
                    should_throttle = True

            elif event.type == EventType.AUDIO:
                if event.subtype == 'bark_detected':
                    event_type = 'bark'
                    event_data = {
                        'emotion': event.data.get('emotion'),
                        'confidence': event.data.get('confidence', 0),
                    }
                    should_throttle = True

            elif event.type == EventType.SAFETY:
                if event.subtype == 'alert':
                    event_type = 'alert'
                    event_data = {
                        'alert_type': event.data.get('type'),
                        'message': event.data.get('message'),
                    }
                    # Alerts are not throttled - always send immediately

            elif event.type == EventType.SYSTEM:
                if event.subtype == 'mode_changed':
                    event_type = 'mode'
                    event_data = {
                        'mode': event.data.get('mode'),
                        'previous': event.data.get('previous'),
                    }
                    # Mode changes are not throttled
                elif event.subtype in ('battery_low', 'battery_critical', 'battery_charging', 'battery_status'):
                    # Battery events are published as SYSTEM events
                    # Include temperature and treats info for full telemetry
                    temperature = self.state.hardware.temperature if self.state else 0.0

                    # Get total treats dispensed today from dispenser
                    treats_today = 0
                    if self.dispenser:
                        try:
                            status = self.dispenser.get_status()
                            treats_today = status.get('total_dispensed', 0)
                        except Exception:
                            pass

                    event_type = 'battery'
                    event_data = {
                        'level': event.data.get('percentage', 0),
                        'charging': event.data.get('charging', False) or event.subtype == 'battery_charging',
                        'voltage': event.data.get('voltage'),
                        'temperature': temperature,
                        'treats_today': treats_today,
                    }
                    # Battery status is not throttled (already rate-limited at source)

            # Send to relay if we have a mapped event
            if event_type:
                # Apply throttling for detection and bark events
                if should_throttle:
                    current_time = time.time()
                    last_time = self._relay_event_times.get(event_type, 0)
                    if current_time - last_time < self._relay_throttle_interval:
                        # Throttled - skip this event
                        return
                    self._relay_event_times[event_type] = current_time

                self.relay_client.send_event(event_type, event_data)
                self.logger.debug(f"☁️ Forwarded {event_type} to relay: {event_data}")

        except Exception as e:
            self.logger.error(f"Relay forward error: {e}")

    def _handle_cloud_command(self, event) -> None:
        """Handle commands from cloud relay (app) by forwarding to local API

        Command formats (per API_CONTRACT_v1.1.md):
        - {"command": "treat"} → POST /treat/dispense
        - {"command": "led", "params": {"pattern": "rainbow"}} → POST /led/pattern
        - {"command": "servo", "params": {"pan": x}} → POST /servo/pan
        - {"command": "servo_center"} → POST /servo/center
        - {"command": "audio", "params": {"file": "..."}} → POST /audio/play
        - {"command": "mode", "params": {"mode": "..."}} → POST /mode/set

        Motor commands are ignored here - they go via WebRTC data channel for low latency.
        """
        if event.subtype != 'command':
            return

        try:
            import httpx

            # Log full event data for debugging
            self.logger.info(f"☁️ Event data: {event.data}")

            command = event.data.get('command')
            params = event.data.get('params', {})
            api_base = 'http://localhost:8000'

            self.logger.info(f"☁️ Processing command={command} params={params}")

            with httpx.Client(timeout=5.0) as client:
                if command == 'treat':
                    # Treat dispensing: POST /treat/dispense
                    resp = client.post(f'{api_base}/treat/dispense', json={
                        'reason': 'cloud_command'
                    })
                    self.logger.info(f"☁️ Treat dispensed -> {resp.status_code}")

                elif command == 'led':
                    # LED control per API contract
                    if params.get('off'):
                        resp = client.post(f'{api_base}/led/off')
                        self.logger.info(f"☁️ LED off -> {resp.status_code}")
                    elif 'pattern' in params:
                        resp = client.post(f'{api_base}/led/pattern', json={
                            'pattern': params['pattern']
                        })
                        self.logger.info(f"☁️ LED pattern={params['pattern']} -> {resp.status_code}")
                    elif 'color' in params:
                        color = params['color']
                        if isinstance(color, list) and len(color) == 3:
                            resp = client.post(f'{api_base}/led/color', json={
                                'r': color[0], 'g': color[1], 'b': color[2]
                            })
                            self.logger.info(f"☁️ LED color={color} -> {resp.status_code}")
                    elif 'r' in params and 'g' in params and 'b' in params:
                        resp = client.post(f'{api_base}/led/color', json={
                            'r': params['r'], 'g': params['g'], 'b': params['b']
                        })
                        self.logger.info(f"☁️ LED RGB -> {resp.status_code}")

                elif command == 'servo':
                    # Servo control: {"pan": x} or {"tilt": y} or {"center": true}
                    if params.get('center'):
                        resp = client.post(f'{api_base}/servo/center')
                        self.logger.info(f"☁️ Servo centered -> {resp.status_code}")
                    else:
                        pan_val = params.get('pan', 0)
                        tilt_val = params.get('tilt', 0)
                        # Ignore commands where both pan AND tilt are near zero (±0.5)
                        # This prevents camera from snapping back when joystick is released
                        # Only servo_center command should recenter (matches Xbox controller behavior)
                        if abs(pan_val) <= 0.5 and abs(tilt_val) <= 0.5:
                            self.logger.info("☁️ Servo ignoring near-zero (joystick released - camera holds position)")
                        else:
                            if 'pan' in params and abs(pan_val) > 0.5:
                                resp = client.post(f'{api_base}/servo/pan', json={
                                    'angle': float(pan_val)
                                })
                                self.logger.info(f"☁️ Servo pan={pan_val} -> {resp.status_code}")
                            if 'tilt' in params and abs(tilt_val) > 0.5:
                                resp = client.post(f'{api_base}/servo/tilt', json={
                                    'angle': float(tilt_val)
                                })
                                self.logger.info(f"☁️ Servo tilt={tilt_val} -> {resp.status_code}")

                elif command == 'servo_center':
                    # Dedicated servo center command
                    resp = client.post(f'{api_base}/servo/center')
                    self.logger.info(f"☁️ Servo centered -> {resp.status_code}")

                elif command == 'audio':
                    # Audio: {"file": "good_dog.mp3"} or {"stop": true}
                    if params.get('stop'):
                        resp = client.post(f'{api_base}/audio/stop')
                        self.logger.info(f"☁️ Audio stopped -> {resp.status_code}")
                    elif 'file' in params:
                        filename = params['file']
                        # Send just the filename to /audio/play endpoint
                        resp = client.post(f'{api_base}/audio/play', json={
                            'file': filename
                        })
                        self.logger.info(f"☁️ Audio play={filename} -> {resp.status_code}")

                elif command == 'mode':
                    # Mode: {"mode": "coach"} or {"mode": "idle"}
                    mode_name = params.get('mode', '').lower()
                    resp = client.post(f'{api_base}/mode/set', json={
                        'mode': mode_name
                    })
                    self.logger.info(f"☁️ Mode set={mode_name} -> {resp.status_code}")

                elif command == 'motor':
                    # Motor commands should go via WebRTC data channel for low latency
                    self.logger.debug(f"☁️ Motor command ignored (use WebRTC data channel)")

                elif command == 'stop':
                    # Emergency stop - stop motors
                    resp = client.post(f'{api_base}/motor/stop')
                    self.logger.info(f"☁️ Emergency stop -> {resp.status_code}")

                elif command == 'upload_voice':
                    # Voice upload: {"name": "sit", "dog_id": "1", "data": "<base64>"}
                    resp = client.post(f'{api_base}/voices/upload', json={
                        'name': params.get('name'),
                        'dog_id': params.get('dog_id'),
                        'data': params.get('data')
                    })
                    self.logger.info(f"☁️ Voice upload -> {resp.status_code}")

                elif command == 'list_voices':
                    # List voices: {"dog_id": "1"} (optional)
                    dog_id = params.get('dog_id')
                    if dog_id:
                        resp = client.get(f'{api_base}/voices/{dog_id}')
                    else:
                        resp = client.get(f'{api_base}/voices')
                    self.logger.info(f"☁️ List voices -> {resp.status_code}")

                elif command == 'delete_voice':
                    # Delete voice: {"name": "sit", "dog_id": "1"}
                    dog_id = params.get('dog_id')
                    name = params.get('name')
                    if dog_id and name:
                        resp = client.delete(f'{api_base}/voices/{dog_id}/{name}')
                        self.logger.info(f"☁️ Delete voice -> {resp.status_code}")

                elif command == 'play_command':
                    # Play voice command with custom voice support
                    resp = client.post(f'{api_base}/audio/play_command', json={
                        'command': params.get('command'),
                        'dog_id': params.get('dog_id')
                    })
                    self.logger.info(f"☁️ Play command -> {resp.status_code}")

                elif command == 'ptt_play':
                    # Push-to-talk: play audio from app
                    resp = client.post(f'{api_base}/ptt/play', json={
                        'data': params.get('data'),
                        'format': params.get('format', 'aac')
                    })
                    self.logger.info(f"☁️ PTT play -> {resp.status_code}")

                elif command == 'ptt_record':
                    # Push-to-talk: record from microphone
                    resp = client.post(f'{api_base}/ptt/record', json={
                        'duration': params.get('duration', 5),
                        'format': params.get('format', 'aac')
                    })
                    self.logger.info(f"☁️ PTT record -> {resp.status_code}")

                else:
                    self.logger.warning(f"☁️ Unknown cloud command: {command}")

        except httpx.TimeoutException:
            self.logger.error(f"☁️ Cloud command timeout: {command}")
        except httpx.ConnectError:
            self.logger.error(f"☁️ Cloud command failed - API not reachable")
        except Exception as e:
            self.logger.error(f"☁️ Cloud command error: {e}", exc_info=True)

    def _on_mode_change(self, data: Dict[str, Any]) -> None:
        """Handle mode changes to start/stop mode handlers"""
        try:
            previous_mode = data.get('previous_mode')  # String value
            new_mode = data.get('new_mode')  # String value

            self.logger.info(f"🔄 Mode change: {previous_mode} → {new_mode}")

            # Play voice announcement for mode change
            self._announce_mode(new_mode)

            # Stop Silent Guardian if leaving that mode
            if previous_mode == 'silent_guardian':
                if self.silent_guardian_mode and self.silent_guardian_mode.running:
                    self.silent_guardian_mode.stop()
                    self.logger.info("🛑 Silent Guardian mode stopped")

            # Stop Coach mode if leaving that mode
            if previous_mode == 'coach':
                if self.coaching_engine and self.coaching_engine.running:
                    self.coaching_engine.stop()
                    self.logger.info("🛑 Coach mode stopped")
                # NOTE: AI detection continues running in all modes (it's a perception layer)

            # Start Silent Guardian if entering that mode
            if new_mode == 'silent_guardian':
                if self.silent_guardian_mode:
                    self.silent_guardian_mode.start()
                    self.logger.info("🛡️ Silent Guardian mode started")

            # Start Coach mode if entering that mode
            if new_mode == 'coach':
                # NOTE: AI detection is already running (started at boot)
                if self.coaching_engine:
                    self.coaching_engine.start()
                    self.logger.info("🎓 Coach mode started")

        except Exception as e:
            self.logger.error(f"Mode change handler error: {e}")

    def _on_system_event(self, event) -> None:
        """Handle system events (controller connect/disconnect, etc.)"""
        try:
            if event.subtype == 'controller_disconnected':
                # Controller disconnected - return to IDLE mode
                current_mode = self.state.get_mode()
                if current_mode == SystemMode.MANUAL:
                    self.logger.info("🎮 Xbox controller disconnected - returning to IDLE")
                    self.state.set_mode(SystemMode.IDLE, "Controller disconnected")

            elif event.subtype == 'controller_connected':
                # Controller connected - switch to Manual mode
                current_mode = self.state.get_mode()
                if current_mode != SystemMode.MANUAL:
                    self.logger.info("🎮 Xbox controller connected - switching to Manual mode")
                    self.mode_fsm.force_mode(SystemMode.MANUAL, "Xbox controller connected")

        except Exception as e:
            self.logger.error(f"System event handler error: {e}")

    def _announce_mode(self, mode: str) -> None:
        """Play voice announcement for mode change (waits for current audio to finish)"""
        try:
            if not self.usb_audio or not self.usb_audio.initialized:
                self.logger.debug("USB audio not available for mode announcement")
                return

            audio_file = self.mode_audio_files.get(mode)
            if audio_file:
                import os
                if os.path.exists(audio_file):
                    # Wait for any current audio to finish (max 3 seconds)
                    # This prevents mode announcements from interrupting important audio
                    wait_count = 0
                    while self.usb_audio.is_busy() and wait_count < 30:
                        time.sleep(0.1)
                        wait_count += 1

                    if wait_count >= 30:
                        self.logger.debug(f"Skipping mode announcement - audio still playing after 3s")
                        return

                    result = self.usb_audio.play_file(audio_file)
                    if result.get('success'):
                        self.logger.info(f"🔊 Mode announcement: {mode}")
                    else:
                        self.logger.warning(f"Mode announcement failed: {result.get('error')}")
                else:
                    self.logger.warning(f"Mode audio file not found: {audio_file}")
            else:
                self.logger.debug(f"No audio file configured for mode: {mode}")
        except Exception as e:
            self.logger.error(f"Mode announcement error: {e}")

    def _run_startup_sequence(self) -> None:
        """Run startup sequence with voice announcement"""
        try:
            # Set LED to idle pattern
            if self.led:
                self.led.set_pattern('idle')

            # Play startup announcement
            # Wait 1.5 seconds for audio subsystem to fully initialize
            time.sleep(1.5)

            if self.usb_audio and self.usb_audio.initialized:
                import os
                if os.path.exists(self.startup_audio):
                    result = self.usb_audio.play_file(self.startup_audio)
                    if result.get('success'):
                        self.logger.info("🔊 WIM-Z Online announcement played")
                    else:
                        self.logger.warning(f"Startup audio failed: {result.get('error')}")
                else:
                    self.logger.warning(f"Startup audio file not found: {self.startup_audio}")
            else:
                self.logger.info("🤫 Silent startup (USB audio not available)")

        except Exception as e:
            self.logger.error(f"Startup sequence failed: {e}")

    def start(self) -> bool:
        """Start the main TreatBot system"""
        if not self.initialization_successful:
            self.logger.error("Cannot start - initialization incomplete")
            return False

        if self.running:
            self.logger.warning("TreatBot already running")
            return True

        self.running = True
        self._stop_event.clear()

        # Start main loop
        self.main_thread = threading.Thread(
            target=self._main_loop,
            daemon=False,
            name="TreatBotMain"
        )
        self.main_thread.start()

        # Set initial mode - IDLE is the boot mode
        # Mode changes to MANUAL on Xbox/app connect, SILENT_GUARDIAN via app command
        current_mode = self.state.get_mode()
        if current_mode == SystemMode.IDLE:
            self.logger.info("System starting in IDLE mode - waiting for Xbox or app connect")
        elif current_mode == SystemMode.MANUAL:
            self.logger.info("Xbox controller active - staying in MANUAL mode")

        self.logger.info("🚀 TreatBot main system started!")
        return True

    def _main_loop(self) -> None:
        """Main system loop"""
        self.logger.info("🔄 Main loop started")
        self.last_heartbeat = time.time()

        while not self._stop_event.wait(1.0):  # 1Hz main loop
            try:
                self.loop_count += 1
                current_time = time.time()

                # CRITICAL: Send heartbeat FIRST, before any potentially blocking calls
                # This ensures safety monitor knows we're alive even if other operations block
                self.safety.heartbeat()
                self.last_heartbeat = current_time

                # Update system telemetry (non-blocking, will skip if can't acquire lock)
                try:
                    self._update_telemetry()
                except Exception as e:
                    self.logger.debug(f"Telemetry update skipped: {e}")

                # Check for emergency conditions
                if self.state.is_emergency():
                    self.logger.error("Emergency detected in main loop")
                    break

                # Log periodic status
                if self.loop_count % 60 == 0:  # Every minute
                    self._log_system_status()

            except Exception as e:
                self.logger.error(f"Main loop error: {e}")

        self.logger.info("🔄 Main loop stopped")

    def _update_telemetry(self) -> None:
        """Update system telemetry"""
        try:
            # Get basic system info
            uptime = time.time() - self.start_time

            # Log telemetry to store
            self.store.log_telemetry(
                battery_voltage=self.state.hardware.battery_voltage,
                temperature=self.state.hardware.temperature,
                mode=self.state.mode.value
            )

            # Check for low battery warning
            self._check_battery_warning()

        except Exception as e:
            self.logger.error(f"Telemetry update failed: {e}")

    def _check_battery_warning(self) -> None:
        """Check battery level and play warning if low"""
        try:
            battery_voltage = self.state.hardware.battery_voltage
            # Only check if we have valid battery reading
            if battery_voltage <= 0:
                return

            # Low battery threshold (warning level from safety.py)
            LOW_BATTERY_THRESHOLD = 12.0

            if battery_voltage < LOW_BATTERY_THRESHOLD and not self.low_battery_announced:
                # Play low battery warning
                if self.usb_audio and self.usb_audio.initialized:
                    import os
                    if os.path.exists(self.low_battery_audio):
                        result = self.usb_audio.play_file(self.low_battery_audio)
                        if result.get('success'):
                            self.logger.warning(f"🔋 Low battery warning: {battery_voltage:.1f}V")
                            self.low_battery_announced = True
                        else:
                            self.logger.error(f"Battery warning audio failed: {result.get('error')}")
            elif battery_voltage >= LOW_BATTERY_THRESHOLD + 0.5:
                # Reset the announced flag if battery recovers (with hysteresis)
                self.low_battery_announced = False

        except Exception as e:
            self.logger.error(f"Battery check error: {e}")

    def _log_system_status(self) -> None:
        """Log periodic system status"""
        uptime = time.time() - self.start_time

        status = {
            'uptime': f"{uptime/3600:.1f}h",
            'mode': self.state.mode.value,
            'loop_count': self.loop_count,
            'safety_level': self.safety.current_level.value,
            'active_sequences': self.sequence_engine.get_status()['active_sequences']
        }

        self.logger.info(f"📊 System Status: {status}")

    def stop(self) -> None:
        """Stop the TreatBot system gracefully"""
        self.logger.info("🛑 Stopping TreatBot system...")

        self.running = False
        self._stop_event.set()

        # Wait for main loop to stop with shorter timeout
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
            if self.main_thread.is_alive():
                self.logger.warning("Main thread still running, forcing stop")
                # Force stop by setting emergency flag
                self.shutdown_requested = True

        # Execute shutdown sequence
        self._run_shutdown_sequence()

        # Stop all subsystems
        self._stop_subsystems()

        self.logger.info("🛑 TreatBot system stopped")

    def _run_shutdown_sequence(self) -> None:
        """Run shutdown sequence"""
        try:
            self.state.set_mode(SystemMode.SHUTDOWN, "System shutdown")

            sequence_id = self.sequence_engine.execute_sequence('shutdown')
            if sequence_id:
                # Wait for shutdown sequence to complete
                time.sleep(3.0)

        except Exception as e:
            self.logger.error(f"Shutdown sequence failed: {e}")

    def _stop_subsystems(self) -> None:
        """Stop all subsystems"""
        try:
            # Stop services
            if self.detector:
                self.detector.stop_detection()
            if self.bark_detector:
                self.bark_detector.stop()
            if self.pantilt:
                if self.pantilt:
                    self.pantilt.stop_tracking()
            if self.mode_fsm:
                self.mode_fsm.stop_fsm()
            if self.safety:
                self.safety.stop_monitoring()
            if self.xbox_controller:
                self.xbox_controller.stop()

            # Stop motor bus (direct hardware control)
            if hasattr(self, 'motor_bus') and self.motor_bus:
                self.logger.info("Stopping motor bus...")
                self.motor_bus.stop()

            # Stop mode handlers
            if self.silent_guardian_mode:
                self.silent_guardian_mode.stop()
            if self.coaching_engine:
                self.coaching_engine.stop()

            # Stop cloud relay and WebRTC
            if self.relay_client:
                self.relay_client.stop()
            if self.webrtc_service:
                import asyncio
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.webrtc_service.cleanup())
                    loop.close()
                except Exception as e:
                    self.logger.warning(f"WebRTC cleanup error: {e}")

            # Stop USB audio
            try:
                if self.sfx:
                    self.logger.info("Stopping USB audio...")
                    self.sfx.stop_sound()
            except Exception as e:
                self.logger.error(f"Error stopping audio: {e}")

            # Cleanup services
            services = [self.detector, self.bark_detector, self.pantilt, self.motor, self.dispenser, self.sfx, self.led, self.xbox_controller]

            # Stop API server (it's a daemon thread, will stop when main process stops)
            if self.api_server and self.api_server.is_alive():
                self.logger.info("API server will stop with main process")
            for service in services:
                if service and hasattr(service, 'cleanup'):
                    service.cleanup()

            # Cleanup orchestrators
            orchestrators = [self.sequence_engine, self.reward_logic, self.mode_fsm]
            for orchestrator in orchestrators:
                if orchestrator and hasattr(orchestrator, 'cleanup'):
                    orchestrator.cleanup()

        except Exception as e:
            self.logger.error(f"Subsystem cleanup failed: {e}")

    def _emergency_shutdown(self, reason: str, data: Dict[str, Any]) -> None:
        """Emergency shutdown callback"""
        self.logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        self.shutdown_requested = True

        # Immediate hardware stops
        try:
            if self.detector:
                self.detector.stop_detection()
            if self.pantilt:
                if self.pantilt:
                    self.pantilt.center_camera()
            if self.led:
                self.led.set_pattern('error', 10.0)

        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")

        # Stop main system
        self.stop()

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals"""
        self.logger.info(f"📡 Received signal {signum}")
        self.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = time.time() - self.start_time

        status = {
            'running': self.running,
            'initialization_successful': self.initialization_successful,
            'uptime': uptime,
            'loop_count': self.loop_count,
            'time_since_heartbeat': time.time() - self.last_heartbeat
        }

        # Add subsystem status
        if self.state:
            status['state'] = self.state.get_full_state()
        if self.safety:
            status['safety'] = self.safety.get_status()

        # Add service status
        services = {
            'detector': self.detector,
            'pantilt': self.pantilt,
            'motor': self.motor,
            'dispenser': self.dispenser,
            'sfx': self.sfx,
            'led': self.led
        }

        status['services'] = {}
        for name, service in services.items():
            if service and hasattr(service, 'get_status'):
                status['services'][name] = service.get_status()

        # Add orchestrator status
        orchestrators = {
            'sequence_engine': self.sequence_engine,
            'reward_logic': self.reward_logic,
            'mode_fsm': self.mode_fsm
        }

        status['orchestrators'] = {}
        for name, orchestrator in orchestrators.items():
            if orchestrator and hasattr(orchestrator, 'get_status'):
                status['orchestrators'][name] = orchestrator.get_status()

        return status

    def force_reward(self, dog_id: str = "manual_test") -> bool:
        """Force a reward for testing"""
        if self.reward_logic:
            return self.reward_logic.force_reward(dog_id, "manual")
        return False


def _setup_asyncio_exception_handler():
    """Setup global asyncio exception handler to prevent crashes from unhandled task exceptions"""
    import asyncio

    def handle_asyncio_exception(loop, context):
        """Handle unhandled asyncio exceptions without crashing"""
        exception = context.get('exception')
        message = context.get('message', 'Unknown async error')

        # Get a logger
        logger = logging.getLogger('AsyncioExceptionHandler')

        # Log the error with full context
        if exception:
            logger.error(f"Unhandled async exception: {message}, exception={type(exception).__name__}: {exception}")
            # Log specific known exceptions with helpful context
            if 'TransactionFailed' in str(type(exception).__name__):
                logger.warning("ICE/TURN transaction failed - WebRTC negotiation issue (non-fatal)")
            elif 'ConnectionRefused' in str(type(exception).__name__):
                logger.warning("Connection refused - network issue (non-fatal)")
        else:
            logger.error(f"Unhandled async error: {message}")

        # Don't crash - just log and continue

    # Apply to all event loops
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_asyncio_exception)
        logging.getLogger('AsyncioExceptionHandler').info("Global asyncio exception handler installed")
    except RuntimeError:
        # No event loop yet - will be set up when asyncio tasks start
        pass


def main():
    """Main entry point"""
    print("🤖 TreatBot - AI Dog Training Robot")
    print("=" * 50)

    # Setup global asyncio exception handler FIRST
    _setup_asyncio_exception_handler()

    # Create main orchestrator
    treatbot = TreatBotMain()

    try:
        # Initialize system
        if not treatbot.initialize():
            print("❌ Initialization failed")
            return 1

        # Start system
        if not treatbot.start():
            print("❌ Startup failed")
            return 1

        print("✅ TreatBot is running!")
        print("Press Ctrl+C to stop")

        # Keep main thread alive
        while treatbot.running:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

    finally:
        # Clean shutdown
        treatbot.stop()

    print("👋 TreatBot shutdown complete")
    return 0


if __name__ == "__main__":
    exit(main())