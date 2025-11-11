#!/usr/bin/env python3
"""
WebSocket server for real-time communication with web dashboard
Provides telemetry streaming, event notifications, and bidirectional control
"""

import asyncio
import json
import logging
import time
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from core.bus import get_bus
from core.state import get_state
from core.store import get_store
from orchestrators.mission_engine import get_mission_engine
from services.reward.dispenser import get_dispenser_service
from services.motion.motor import get_motor_service
from services.motion.pan_tilt import get_pantilt_service
from services.media.sfx import get_sfx_service
from services.media.led import get_led_service


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_to_all(self, message: Dict[str, Any]):
        """Send message to all connected clients"""
        if self.active_connections:
            message_text = json.dumps(message)
            disconnected = set()

            for connection in self.active_connections:
                try:
                    await connection.send_text(message_text)
                except Exception as e:
                    self.logger.error(f"Error sending message to client: {e}")
                    disconnected.add(connection)

            # Clean up disconnected clients
            for connection in disconnected:
                self.disconnect(connection)

    async def send_to_one(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending message to client: {e}")
            self.disconnect(websocket)


class TreatBotWebSocketServer:
    """
    WebSocket server for TreatBot web dashboard

    Handles:
    - Real-time telemetry streaming
    - Event notifications from event bus
    - Bidirectional control commands
    - Status updates
    """

    def __init__(self):
        self.manager = ConnectionManager()
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.mission_engine = get_mission_engine()
        self.logger = logging.getLogger(__name__)

        # Service references
        self.dispenser = None
        self.motor = None
        self.pantilt = None
        self.sfx = None
        self.led = None

        # Setup event handlers
        self._setup_event_handlers()

        # Telemetry task
        self.telemetry_task = None

    def _setup_event_handlers(self):
        """Subscribe to relevant events from the bus"""
        self.bus.subscribe("vision", self._on_vision_event)
        self.bus.subscribe("audio", self._on_audio_event)
        self.bus.subscribe("motion", self._on_motion_event)
        self.bus.subscribe("reward", self._on_reward_event)
        self.bus.subscribe("system", self._on_system_event)
        self.bus.subscribe("mission", self._on_mission_event)

    def _get_services(self):
        """Lazy load services to avoid circular imports"""
        if self.dispenser is None:
            try:
                self.dispenser = get_dispenser_service()
                self.motor = get_motor_service()
                self.pantilt = get_pantilt_service()
                self.sfx = get_sfx_service()
                self.led = get_led_service()

                # Auto-initialize services if not already done
                if self.motor and not self.motor.initialized:
                    self.logger.info("Auto-initializing motor service")
                    if self.motor.initialize():
                        # Ensure motor is in manual mode for web control
                        from services.motion.motor import MovementMode
                        self.motor.set_movement_mode(MovementMode.MANUAL)

                if self.pantilt and not self.pantilt.servo_initialized:
                    self.logger.info("Auto-initializing pan/tilt service")
                    self.pantilt.initialize()

            except Exception as e:
                self.logger.warning(f"Could not load services: {e}")

    async def handle_websocket(self, websocket: WebSocket):
        """Handle a WebSocket connection"""
        await self.manager.connect(websocket)

        # Start telemetry if this is the first connection
        if len(self.manager.active_connections) == 1:
            await self._start_telemetry()

        # Send initial status
        await self._send_initial_status(websocket)

        try:
            while True:
                # Listen for client messages
                data = await websocket.receive_text()
                await self._handle_client_message(websocket, data)

        except WebSocketDisconnect:
            self.logger.info("Client disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.manager.disconnect(websocket)

            # Stop telemetry if no more connections
            if len(self.manager.active_connections) == 0:
                await self._stop_telemetry()

    async def _send_initial_status(self, websocket: WebSocket):
        """Send initial system status to new client"""
        state = self.state.get_full_state()
        mission_status = self.mission_engine.get_mission_status()

        await self.manager.send_to_one(websocket, {
            "type": "initial_status",
            "data": {
                "system_state": state,
                "mission_status": mission_status,
                "timestamp": time.time()
            }
        })

    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            command = data.get("command")
            params = data.get("params", {})

            self.logger.info(f"Received command: {command}")

            response = await self._execute_command(command, params)

            await self.manager.send_to_one(websocket, {
                "type": "command_response",
                "command": command,
                "data": response,
                "timestamp": time.time()
            })

        except json.JSONDecodeError:
            await self.manager.send_to_one(websocket, {
                "type": "error",
                "message": "Invalid JSON",
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            })

    async def _execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command from the client"""
        self._get_services()

        if command == "motor_control":
            # {"left_speed": 50, "right_speed": -50, "duration": 2.0}
            left = params.get("left_speed", 0)
            right = params.get("right_speed", 0)
            duration = params.get("duration", None)

            if self.motor:
                # Convert individual wheel speeds to direction and speed
                if left == 0 and right == 0:
                    direction = "stop"
                    speed = 0
                elif left == right:
                    direction = "forward" if left > 0 else "backward"
                    speed = abs(left)
                elif left == -right:
                    direction = "left" if left < 0 else "right"
                    speed = abs(left)
                else:
                    # Mixed speeds - use average and best direction
                    avg_speed = (abs(left) + abs(right)) / 2
                    if abs(left + right) > abs(left - right):
                        direction = "forward" if (left + right) > 0 else "backward"
                    else:
                        direction = "left" if (left - right) < 0 else "right"
                    speed = int(avg_speed)

                success = self.motor.manual_drive(direction, speed, duration)
                return {"success": success, "left_speed": left, "right_speed": right, "direction": direction}
            return {"success": False, "error": "Motor service not available"}

        elif command == "joystick_control":
            # {"x": 0.5, "y": 0.8}
            x = params.get("x", 0.0)  # -1 to 1
            y = params.get("y", 0.0)  # -1 to 1

            if self.motor:
                # Convert joystick to direction and speed
                if abs(x) < 0.1 and abs(y) < 0.1:
                    # Deadzone - stop
                    direction = "stop"
                    speed = 0
                elif abs(y) > abs(x):
                    # Primarily forward/backward
                    direction = "forward" if y > 0 else "backward"
                    speed = int(abs(y) * 100)
                else:
                    # Primarily left/right
                    direction = "right" if x > 0 else "left"
                    speed = int(abs(x) * 100)

                # Ensure speed is within bounds
                speed = max(0, min(100, speed))

                success = self.motor.manual_drive(direction, speed, 0.5)

                # Calculate display speeds for joystick feedback
                left_speed = int((y + x) * 100)
                right_speed = int((y - x) * 100)
                left_speed = max(-100, min(100, left_speed))
                right_speed = max(-100, min(100, right_speed))

                return {"success": success, "left_speed": left_speed, "right_speed": right_speed, "direction": direction}
            return {"success": False, "error": "Motor service not available"}

        elif command == "pan_tilt":
            # {"pan": 90, "tilt": 120}
            pan = params.get("pan")
            tilt = params.get("tilt")

            if self.pantilt:
                success = self.pantilt.move_camera(pan=pan, tilt=tilt)
                return {"success": success, "pan": pan, "tilt": tilt}
            return {"success": False, "error": "Pan/tilt service not available"}

        elif command == "dispense_treat":
            # {"count": 1, "reason": "manual"}
            count = params.get("count", 1)
            reason = params.get("reason", "manual")

            if self.dispenser:
                success = self.dispenser.dispense_treat(count)
                return {"success": success, "count": count, "reason": reason}
            return {"success": False, "error": "Dispenser service not available"}

        elif command == "emergency_stop":
            if self.motor:
                self.motor.emergency_stop()
            return {"success": True, "message": "Emergency stop executed"}

        elif command == "play_sound":
            # {"sound_name": "good_dog"}
            sound_name = params.get("sound_name", "beep")

            if self.sfx:
                success = self.sfx.play_sound(sound_name)
                return {"success": success, "sound_name": sound_name}
            return {"success": False, "error": "SFX service not available"}

        elif command == "set_led_color":
            # {"color": "blue", "brightness": 0.5}
            color = params.get("color", "blue")
            brightness = params.get("brightness", 0.5)

            if self.led:
                success = self.led.set_color(color, brightness)
                return {"success": success, "color": color, "brightness": brightness}
            return {"success": False, "error": "LED service not available"}

        elif command == "start_mission":
            # {"mission_name": "train_sit_daily"}
            mission_name = params.get("mission_name")

            if mission_name:
                success = self.mission_engine.start_mission(mission_name)
                return {"success": success, "mission_name": mission_name}
            return {"success": False, "error": "Mission name required"}

        elif command == "stop_mission":
            success = self.mission_engine.stop_mission("user_requested")
            return {"success": success}

        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    async def _start_telemetry(self):
        """Start periodic telemetry broadcast"""
        if self.telemetry_task is None or self.telemetry_task.done():
            self.telemetry_task = asyncio.create_task(self._telemetry_loop())

    async def _stop_telemetry(self):
        """Stop telemetry broadcast"""
        if self.telemetry_task and not self.telemetry_task.done():
            self.telemetry_task.cancel()

    async def _telemetry_loop(self):
        """Periodic telemetry broadcast"""
        try:
            while True:
                telemetry = await self._collect_telemetry()
                await self.manager.send_to_all({
                    "type": "telemetry",
                    "data": telemetry,
                    "timestamp": time.time()
                })
                await asyncio.sleep(1.0)  # Send every second

        except asyncio.CancelledError:
            self.logger.info("Telemetry loop cancelled")
        except Exception as e:
            self.logger.error(f"Telemetry loop error: {e}")

    async def _collect_telemetry(self) -> Dict[str, Any]:
        """Collect current system telemetry"""
        state = self.state.get_full_state()
        mission_status = self.mission_engine.get_mission_status()

        # Get database stats
        db_stats = self.store.get_database_stats()

        telemetry = {
            "system_mode": state["mode"],
            "emergency": state["emergency"],
            "uptime": time.time() - state.get("start_time", time.time()),
            "mission": mission_status,
            "database": {
                "events_count": db_stats.get("events_count", 0),
                "rewards_today": db_stats.get("rewards_last_day", 0),
                "size_mb": db_stats.get("db_size_mb", 0)
            }
        }

        return telemetry

    # Event handlers
    async def _on_vision_event(self, event):
        """Handle vision events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "vision",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

    async def _on_audio_event(self, event):
        """Handle audio events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "audio",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

    async def _on_motion_event(self, event):
        """Handle motion events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "motion",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

    async def _on_reward_event(self, event):
        """Handle reward events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "reward",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

    async def _on_system_event(self, event):
        """Handle system events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "system",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

    async def _on_mission_event(self, event):
        """Handle mission events"""
        await self.manager.send_to_all({
            "type": "event",
            "category": "mission",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })


# Global WebSocket server instance
_websocket_server = None

def get_websocket_server():
    """Get the global WebSocket server instance"""
    global _websocket_server
    if _websocket_server is None:
        _websocket_server = TreatBotWebSocketServer()
    return _websocket_server