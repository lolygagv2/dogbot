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
from services.control.bluetooth_esc import BluetoothESCController
from services.control.xbox_controller import get_xbox_service
from api.server import run_server

# Orchestrators
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from orchestrators.mode_fsm import get_mode_fsm


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
        self.bluetooth_controller = None
        self.xbox_controller = None
        self.api_server = None

        # Orchestrators
        self.sequence_engine = None
        self.reward_logic = None
        self.mode_fsm = None

        # Main state
        self.running = False
        self.initialization_successful = False
        self.shutdown_requested = False

        # Main loop
        self.main_thread = None
        self._stop_event = threading.Event()

        # Performance tracking
        self.start_time = time.time()
        self.loop_count = 0
        self.last_heartbeat = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/var/log/treatbot.log', mode='a') if os.access('/var/log', os.W_OK) else logging.StreamHandler()
            ]
        )
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

        # Motor service - DISABLED (Xbox controller has direct motor control)
        try:
            # self.motor = get_motor_service()
            # services_status['motor'] = self.motor.initialize()
            services_status['motor'] = False  # Disabled to prevent GPIO conflicts
            self.logger.info("🚗 Main motor service disabled (Xbox controller has direct control)")
        except Exception as e:
            self.logger.error(f"Motor service failed: {e}")
            services_status['motor'] = False

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

        # Check critical services
        critical_services = ['detector', 'dispenser']
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

            # Start detection (will be controlled by mode FSM)
            if self.detector.ai_initialized:
                # Subscribe to detection events for LED feedback
                self.bus.subscribe('vision', self._on_detection_for_feedback)

            # Start bark detection if enabled
            if self.bark_detector.enabled:
                self.bark_detector.start()
                # Subscribe to bark events for feedback
                self.bus.subscribe('audio', self._on_bark_for_feedback)
                self.logger.info("🎤 Bark detection started")

            # Start Bluetooth controller if available
            if self.bluetooth_controller and self.bluetooth_controller.is_connected:
                self.bluetooth_controller.start()
                self.logger.info("🎮 Bluetooth controller active - Press START to enter MANUAL mode")
                self.state.set_mode(SystemMode.MANUAL, "Bluetooth controller ready")

            self.logger.info("✅ All subsystems started")
            return True

        except Exception as e:
            self.logger.error(f"Subsystem startup failed: {e}")
            return False

    def _on_detection_for_feedback(self, event) -> None:
        """Provide visual feedback for detection events"""
        try:
            if event.subtype == 'dog_detected':
                # Green LED for dog detected
                if self.led:
                    self.led.set_pattern('dog_detected')
                self.logger.info(f"🐕 Dog detected: {event.data.get('dog_id', 'unknown')}")

            elif event.subtype == 'dog_lost':
                # Blue LED for searching
                if self.led:
                    self.led.set_pattern('searching')
                self.logger.info("👀 Dog lost, searching...")

            elif event.subtype == 'pose':
                behavior = event.data.get('behavior')
                if behavior:
                    self.logger.info(f"🎯 Behavior detected: {behavior}")
                    # Pulse for behavior detection
                    if self.led:
                        self.led.pulse_color('yellow')

        except Exception as e:
            self.logger.error(f"Detection feedback error: {e}")

    def _on_bark_for_feedback(self, event) -> None:
        """Provide feedback for bark detection events"""
        try:
            if event.subtype == 'bark_detected':
                emotion = event.data.get('emotion', 'unknown')
                confidence = event.data.get('confidence', 0)

                # Log bark detection
                self.logger.info(f"🐕 Bark detected: {emotion} (conf: {confidence:.2f})")

                # Visual feedback - flash LED based on emotion
                if self.led and self.led.led_initialized:
                    if emotion in ['alert', 'attention']:
                        self.led.set_pattern('pulse_green', 2.0)
                    elif emotion in ['anxious', 'scared']:
                        self.led.set_pattern('spinning_dot', 2.0, color='yellow')
                    elif emotion == 'aggressive':
                        self.led.set_pattern('error', 2.0)
                    elif emotion == 'playful':
                        self.led.set_pattern('pulse_blue', 2.0)

            elif event.subtype == 'bark_rewarded':
                emotion = event.data.get('emotion', 'unknown')
                self.logger.info(f"🎁 Bark reward triggered for: {emotion}")

                # Celebration feedback
                if self.led and self.led.led_initialized:
                    self.led.set_pattern('celebration', 3.0)

        except Exception as e:
            self.logger.error(f"Bark feedback error: {e}")

    def _run_startup_sequence(self) -> None:
        """Run startup sequence"""
        try:
            # Execute startup sequence
            sequence_id = self.sequence_engine.execute_sequence('startup')
            if sequence_id:
                self.logger.info(f"🎬 Startup sequence launched: {sequence_id}")
            else:
                self.logger.warning("Startup sequence failed, using fallback")
                # Fallback startup
                if self.led:
                    self.led.set_pattern('idle')
                if self.sfx:
                    self.sfx.play_system_sound('startup')

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

        # Set initial mode - but don't override manual mode if Xbox controller is active
        current_mode = self.state.get_mode()
        if current_mode != SystemMode.MANUAL:
            self.state.set_mode(SystemMode.DETECTION, "System ready - starting detection")
        else:
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

                # Send heartbeat to safety monitor
                self.safety.heartbeat()
                self.last_heartbeat = current_time

                # Update system telemetry
                self._update_telemetry()

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

        except Exception as e:
            self.logger.error(f"Telemetry update failed: {e}")

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

            # Stop audio explicitly
            try:
                if self.sfx and hasattr(self.sfx, 'audio') and self.sfx.audio:
                    self.logger.info("Stopping DFPlayer audio...")
                    # Send stop commands to DFPlayer
                    if hasattr(self.sfx.audio, '_send_command'):
                        self.sfx.audio._send_command('AT+STOP')
                        self.sfx.audio._send_command('AT+VOL=0')
                        self.sfx.audio._send_command('AT+AMP=OFF')
                    # Close serial connection
                    if hasattr(self.sfx.audio, 'serial_connection') and self.sfx.audio.serial_connection:
                        self.sfx.audio.serial_connection.close()
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


def main():
    """Main entry point"""
    print("🤖 TreatBot - AI Dog Training Robot")
    print("=" * 50)

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