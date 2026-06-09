#!/usr/bin/env python3
"""
WebSocket server for real-time communication with web dashboard
Provides telemetry streaming, event notifications, and bidirectional control
"""

import asyncio
import json
import logging
import os
import threading
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
from services.media.photo_capture import get_photo_capture_service
from services.media.voice_manager import get_voice_manager
from services.media.push_to_talk import get_push_to_talk_service
from services.media.usb_audio import get_usb_audio_service


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger('TreatBotAPI')

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
        self.logger = logging.getLogger('TreatBotAPI')

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
        """Subscribe to relevant events from the bus.
        EventBus calls from threads, so wrap async handlers to schedule
        onto the asyncio event loop instead of creating unawaited coroutines."""
        import asyncio

        def _make_sync_wrapper(async_handler):
            """Wrap async handler for sync EventBus callback"""
            def wrapper(event):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(async_handler(event))
                    else:
                        loop.run_until_complete(async_handler(event))
                except RuntimeError:
                    # No event loop in this thread — create task from main loop
                    pass
            return wrapper

        self.bus.subscribe("vision", _make_sync_wrapper(self._on_vision_event))
        self.bus.subscribe("audio", _make_sync_wrapper(self._on_audio_event))
        self.bus.subscribe("motion", _make_sync_wrapper(self._on_motion_event))
        self.bus.subscribe("reward", _make_sync_wrapper(self._on_reward_event))
        self.bus.subscribe("system", _make_sync_wrapper(self._on_system_event))
        self.bus.subscribe("mission", _make_sync_wrapper(self._on_mission_event))

    def _get_services(self):
        """Lazy load services to avoid circular imports.
        Each service loaded independently so one failure doesn't block the rest."""
        if self.dispenser is not None and self.led is not None:
            return  # All critical services loaded

        for name, getter in [
            ("dispenser", get_dispenser_service),
            ("motor", get_motor_service),
            ("pantilt", get_pantilt_service),
            ("sfx", get_sfx_service),
            ("led", get_led_service),
        ]:
            if getattr(self, name) is None:
                try:
                    setattr(self, name, getter())
                except Exception as e:
                    self.logger.warning(f"Could not load {name}: {e}")

        # Auto-initialize services if not already done
        try:
            if self.motor and not self.motor.initialized:
                self.logger.info("Auto-initializing motor service")
                if self.motor.initialize():
                    from services.motion.motor import MovementMode
                    self.motor.set_movement_mode(MovementMode.MANUAL)

            if self.pantilt and not self.pantilt.servo_initialized:
                self.logger.info("Auto-initializing pan/tilt service")
                self.pantilt.initialize()
        except Exception as e:
            self.logger.warning(f"Service auto-init error: {e}")

    async def handle_websocket(self, websocket: WebSocket):
        """Handle a WebSocket connection (used by both /ws and /ws/local)"""
        await self.manager.connect(websocket)
        self.logger.info(f"WebSocket connected. Path: {websocket.url.path}")

        # Register this socket loop as a sink for controller-pairing events so
        # spontaneous updates (auto-reconnect, scan trickle) reach local-AP
        # clients. Once per process; the emitter fans out to all connections.
        self._register_controller_emitter()

        # Signal that an app client is connected (LED idle, publish event)
        self._on_app_connected()

        # Start telemetry if this is the first connection
        if len(self.manager.active_connections) == 1:
            await self._start_telemetry()

        # Send connected message (app expects this from /ws/local)
        await self.manager.send_to_one(websocket, {
            "type": "connected",
            "mode": "local",
            "current_mode": self.state.get_mode().value,
            "api_version": "1.0"
        })

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

            # Stop motors on disconnect (safety — same as Xbox controller)
            try:
                from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
                motor_bus = get_motor_bus()
                if motor_bus and motor_bus.running:
                    cmd = create_motor_command(0, 0, CommandSource.API)
                    motor_bus.send_command(cmd)
            except Exception:
                pass

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

    def _on_app_connected(self):
        """Signal that an app client connected via local WebSocket

        Stops the WiFi provisioning pulsing-blue animation and sets NeoPixels to idle.
        Also sets GPIO25 blue LED to solid on (not flashing).
        """
        self._get_services()
        # Retry LED init if it failed at boot (wifi provisioning may have released GPIO since)
        if self.led and not self.led.led_initialized:
            self.logger.info("Retrying LED initialization (may have been blocked at boot)")
            self.led.initialize()
        if self.led:
            self.led.set_pattern('idle')
        # Set blue LED solid on (stops the "flashing" look). Use LedService's
        # controller — the single GPIO 25 owner — not api.server's separate
        # lgpio claim, which loses the startup race and silently fails.
        try:
            led = self.led.led if self.led else None
            # blue_on() lazily (re)claims GPIO25, so don't pre-gate on blue_chip
            # — the on-connect blue tube self-heals past the boot-time race.
            if led is not None and led.blue_on():
                self.logger.info("App connected via local WS — LED idle, blue LED solid ON")
        except Exception as e:
            self.logger.warning(f"Blue LED on failed (non-critical): {e}")

    async def _handle_get_status(self, websocket: WebSocket):
        """Respond to get_status with robot online/paired status"""
        await self.manager.send_to_one(websocket, {
            "type": "status_response",
            "robot_online": True,
            "device_paired": True,
            "timestamp": time.time()
        })

    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)

            # Handle ping/pong (API Contract)
            if data.get("type") == "ping":
                await self.manager.send_to_one(websocket, {"type": "pong"})
                return

            # Handle auth (API Contract - for cloud mode, local mode ignores)
            if data.get("type") == "auth":
                await self.manager.send_to_one(websocket, {"type": "auth_result", "success": True})
                return

            # Handle get_status request
            if data.get("type") == "get_status":
                await self._handle_get_status(websocket)
                return

            # Ignore debug_log messages from app (just logging, no action needed)
            if data.get("type") == "debug_log":
                return

            # Handle webrtc_close (app closing video session)
            if data.get("type") == "webrtc_close":
                session_id = data.get("session_id", "local")
                try:
                    from services.streaming.webrtc import get_webrtc_service
                    webrtc = get_webrtc_service()
                    await webrtc._cleanup_connection(session_id)
                    self.logger.info(f"[LOCAL] WebRTC session {session_id} closed by app")
                except Exception as e:
                    self.logger.warning(f"WebRTC close error: {e}")
                return

            # Unwrap app command wrapper format:
            # {"type": "command", "device_id": "local_robot", "command": "motor", "data": {"left": 0.5}}
            # Flatten inner "data" to top level so downstream handlers find params directly
            if data.get("type") == "command" and "command" in data:
                inner_cmd = data["command"]
                inner_data = data.get("data") or {}
                data = {**inner_data, "type": inner_cmd, "command": inner_cmd}
                self.logger.info(f"[LOCAL] Unwrapped command: {inner_cmd} (keys: {list(data.keys())})")

            # Handle push-to-talk audio message (play from app)
            # Accept both "audio_message" (relay protocol) and "ptt_play" (app command)
            msg_type = data.get("type") or data.get("command")
            if msg_type in ("audio_message", "ptt_play"):
                await self._handle_audio_message(websocket, data)
                return

            # Handle audio request (record and send back)
            if msg_type == "audio_request":
                await self._handle_audio_request(websocket, data)
                return

            # Handle WebRTC signaling (for local mode — app connects to /ws directly)
            if msg_type == "webrtc_request":
                await self._handle_webrtc_request(websocket, data)
                return
            if msg_type == "webrtc_answer":
                await self._handle_webrtc_answer(websocket, data)
                return
            if msg_type == "webrtc_ice":
                await self._handle_webrtc_ice(websocket, data)
                return

            # Handle contract-style commands (API Contract format)
            if "command" in data and "params" not in data:
                # This is an API contract format command
                response = await self._execute_contract_command(websocket, data)
                return

            # Handle legacy format with command + params
            command = data.get("command")
            params = data.get("params", {})

            self.logger.info(f"Received command: {command} | raw keys: {list(data.keys())} | type: {data.get('type')}")

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

    async def _execute_contract_command(self, websocket: WebSocket, data: Dict[str, Any]):
        """Execute API contract format commands

        Sends a command_response back for every command so the app gets confirmation.
        """
        command = data.get("command")
        self._get_services()
        result = {"success": True, "command": command}

        try:
            if command == "motor":
                # {"command": "motor", "left": -1.0..1.0, "right": -1.0..1.0}
                # Use motor_bus directly (same path as Xbox controller) for low latency
                left = max(-1.0, min(1.0, float(data.get("left", 0.0))))
                right = max(-1.0, min(1.0, float(data.get("right", 0.0))))
                left_pct = int(left * 100)
                right_pct = int(right * 100)
                try:
                    from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
                    motor_bus = get_motor_bus()
                    if motor_bus and motor_bus.running:
                        cmd = create_motor_command(left_pct, right_pct, CommandSource.API)
                        motor_bus.send_command(cmd)
                    else:
                        result = {"success": False, "command": command, "error": "Motor bus not running"}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "servo":
                # {"command": "servo", "pan": 15.0, "tilt": -10.0}
                pan = data.get("pan")
                tilt = data.get("tilt")
                if self.pantilt:
                    pan_internal = int(90 + pan) if pan is not None else None
                    tilt_internal = int(90 + tilt) if tilt is not None else None
                    self.pantilt.move_camera(pan=pan_internal, tilt=tilt_internal)
                else:
                    result = {"success": False, "command": command, "error": "Pan/tilt service not available"}

            elif command in ("treat", "dispense_treat"):
                # {"command": "treat", "count": N, "reason": "...", "dog_id": "..."}
                if self.dispenser:
                    count = max(1, min(10, int(data.get("count", 1))))
                    reason = data.get("reason", "manual")
                    dog_id = data.get("dog_id")
                    if count == 1:
                        self.dispenser.dispense_treat(dog_id=dog_id, reason=reason)
                    else:
                        # Multi-dispense blocks ~1.5s per treat — run off the WS thread
                        threading.Thread(
                            target=self.dispenser.dispense_multiple,
                            kwargs={"count": count, "dog_id": dog_id, "reason": reason},
                            daemon=True, name="TreatMultiDispense",
                        ).start()
                else:
                    result = {"success": False, "command": command, "error": "Dispenser not available"}

            elif command == "led":
                # {"command": "led", "pattern": "celebration"}
                # Call LED service directly (not via self.led) to avoid lazy-load issues
                pattern = data.get("pattern", "idle")
                pattern_map = {
                    "breathing": "idle",
                    "rainbow": "rainbow",
                    "gradient_flow": "gradient_flow",
                    "celebration": "treat_launching",
                    "searching": "searching",
                    "alert": "error",
                    "idle": "idle",
                    "fire": "fire",
                    "chase": "chase",
                    "off": "off",
                    "blue": "solid_blue",
                    "red": "solid_red",
                    "green": "solid_green",
                    "white": "solid_white",
                    "blue_on": "blue_led_on",
                    "blue_off": "blue_led_off",
                    "ambient": "ambient",
                }
                mode = pattern_map.get(pattern, pattern)
                try:
                    led_svc = get_led_service()
                    led_svc.set_pattern(mode)
                    result["pattern"] = mode
                except Exception as e:
                    self.logger.error(f"LED command error: {e}")
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio":
                # {"command": "audio", "file": "good.mp3"}
                audio_file = data.get("file")
                if audio_file and self.sfx:
                    self.sfx.play_sound(audio_file)
                elif not self.sfx:
                    result = {"success": False, "command": command, "error": "SFX service not available"}

            elif command == "mode":
                # {"command": "mode", "mode": "training"}
                # Use set_mode_override() for parity with relay_client — protects
                # SG/Coach from FSM auto-revert (coach_timeout, etc.)
                mode = data.get("mode")
                if mode:
                    from orchestrators.mode_fsm import get_mode_fsm
                    from core.state import SystemMode
                    mode_fsm = get_mode_fsm()
                    # Map contract modes to internal modes (accept both aliases)
                    mode_map = {
                        "idle": SystemMode.IDLE,
                        "silent_guardian": SystemMode.SILENT_GUARDIAN,
                        "guardian": SystemMode.SILENT_GUARDIAN,  # alias
                        "coach": SystemMode.COACH,
                        "training": SystemMode.COACH,  # alias
                        "mission": SystemMode.MISSION,
                        "manual": SystemMode.MANUAL,
                        "docking": SystemMode.IDLE  # No docking mode yet
                    }
                    internal_mode = mode_map.get(mode)
                    if internal_mode:
                        mode_fsm.set_mode_override(internal_mode)
                        result["mode"] = mode

            elif command == "play_voice":
                # {"command": "play_voice", "voice_type": "sit", "dog_id": "1"}
                voice_type = data.get("voice_type")
                dog_id = data.get("dog_id")
                if voice_type:
                    try:
                        voice_manager = get_voice_manager()
                        voice_path = voice_manager.get_voice_path(dog_id or "default", voice_type)
                        if voice_path:
                            usb_audio = get_usb_audio_service()
                            usb_audio.play_file(voice_path)
                        elif self.sfx:
                            self.sfx.play_sound(voice_type)
                        else:
                            result = {"success": False, "command": command, "error": f"Voice not found: {voice_type}"}
                    except Exception as e:
                        self.logger.error(f"play_voice error: {e}")
                        result = {"success": False, "command": command, "error": str(e)}

            elif command == "call_dog":
                # {"command": "call_dog", "dog_id": "dog_123"} - plays 'come' command
                from services.media.voice_lookup import resolve_voice_file
                dog_id = data.get("dog_id")
                try:
                    audio_path = resolve_voice_file("come", dog_id_override=dog_id)
                    if audio_path:
                        usb_audio = get_usb_audio_service()
                        if usb_audio and usb_audio.is_initialized:
                            usb_audio.play_file(audio_path)
                            self.logger.info(f"call_dog: playing {audio_path}")
                        else:
                            result = {"success": False, "command": command, "error": "USB audio not initialized"}
                    else:
                        result = {"success": False, "command": command, "error": "Voice file not found for 'come'"}
                except Exception as e:
                    self.logger.error(f"call_dog error: {e}")
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "mood_led":
                # {"command": "mood_led", "action": "on"/"off"/"toggle"}
                # Controls the BLUE LED TUBE (GPIO25), NOT NeoPixels.
                # Route through LedService's controller — the SINGLE legitimate
                # owner of GPIO 25. The old api.server.blue_led_direct_control()
                # made its own lgpio.gpio_claim_output() on the same pin, lost
                # the startup race with LedController, and then failed silently
                # for the whole process — which is why mood_led did nothing over
                # the local/AP path while the relay path (already on LedService)
                # worked. (Same fix as relay_client._handle_mood_led.)
                action = data.get("action", "toggle").lower()
                try:
                    from services.media.led import get_led_service
                    led = get_led_service().led
                    if led is None:
                        raise RuntimeError("LED service unavailable")
                    # Don't pre-check blue_chip: blue_on/off lazily (re)claim GPIO25,
                    # self-healing the boot-time 'GPIO busy' race. ok reflects reality.
                    if action == "on":
                        ok = bool(led.blue_on())
                    elif action == "off":
                        ok = bool(led.blue_off())
                    else:
                        ok = bool(led.blue_off() if getattr(led, 'blue_is_on', False)
                                  else led.blue_on())
                    result["action"] = action
                    result["success"] = ok
                except Exception as e:
                    self.logger.error(f"mood_led error: {e}")
                    result = {"success": False, "command": command, "error": str(e)}

            elif command in ("led_color", "led_off"):
                # {"command": "led_color", "r": 255, "g": 0, "b": 0} or {"command": "led_off"}
                try:
                    led_svc = get_led_service()
                    if command == "led_off":
                        led_svc.set_pattern("off")
                    else:
                        color = data.get("color", "white")
                        color_map = {
                            "blue": "solid_blue", "red": "solid_red",
                            "green": "solid_green", "white": "solid_white",
                            "fire": "fire", "rainbow": "rainbow",
                            "chase": "chase", "off": "off",
                        }
                        led_svc.set_pattern(color_map.get(color, f"solid_{color}"))
                except Exception as e:
                    self.logger.error(f"led_color error: {e}")
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio_volume":
                # {"command": "audio_volume", "level": 50}
                level = data.get("level", 50)
                try:
                    from services.media.volume_manager import get_volume_manager
                    get_volume_manager().set_volume(int(level))
                    result["volume"] = int(level)
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio_next":
                try:
                    usb_audio = get_usb_audio_service()
                    usb_audio.play_next()
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio_prev":
                try:
                    usb_audio = get_usb_audio_service()
                    usb_audio.play_previous()
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio_toggle":
                try:
                    usb_audio = get_usb_audio_service()
                    usb_audio.toggle()
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "audio_stop":
                try:
                    usb_audio = get_usb_audio_service()
                    usb_audio.stop()
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "set_mode":
                # {"command": "set_mode", "mode": "manual"}
                # Use set_mode_override() to protect from FSM auto-transitions.
                # Without override, the FSM loop can revert SG/Coach back to IDLE
                # within seconds (e.g. coach_timeout with no dog detection).
                # This matches the relay_client behavior for parity.
                mode_name = data.get("mode", "").lower()
                if mode_name:
                    from orchestrators.mode_fsm import get_mode_fsm
                    from core.state import SystemMode
                    mode_fsm = get_mode_fsm()
                    mode_map = {
                        "idle": SystemMode.IDLE,
                        "silent_guardian": SystemMode.SILENT_GUARDIAN,
                        "guardian": SystemMode.SILENT_GUARDIAN,
                        "coach": SystemMode.COACH,
                        "training": SystemMode.COACH,
                        "mission": SystemMode.MISSION,
                        "manual": SystemMode.MANUAL,
                    }
                    internal_mode = mode_map.get(mode_name)
                    if internal_mode:
                        mode_fsm.set_mode_override(internal_mode)
                        result["mode"] = mode_name

            elif command == "force_trick":
                trick = data.get("trick")
                if trick:
                    BEHAVIOR_TO_TRICK = {'stand': 'come', 'lie': 'laydown', 'down': 'laydown'}
                    trick = BEHAVIOR_TO_TRICK.get(trick, trick)
                    # App (Build 106+) sends dog_id/dog_name resolved by priority
                    # (ArUco > selected profile). Pass through so TTS uses the real name.
                    dog_id = data.get("dog_id")
                    dog_name = data.get("dog_name")
                    from orchestrators.coaching_engine import get_coaching_engine
                    engine = get_coaching_engine()
                    if engine and engine.running:
                        engine.set_forced_trick(trick, dog_id=dog_id, dog_name=dog_name)

            elif command == "treat_counter_set":
                # {"command": "treat_counter_set", "count": 44}
                count = data.get("count")
                if count is not None:
                    try:
                        from services.reward.dispenser import get_dispenser_service
                        dispenser = get_dispenser_service()
                        dispenser.set_treat_count(int(count))
                        remaining = dispenser.treats_remaining
                        # Send immediate status so app display updates
                        await self._broadcast_contract_status()
                        result = {"success": True, "treats_remaining": remaining}
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                else:
                    result = {"success": False, "error": "count required"}

            elif command == "treat_counter_reset":
                # {"command": "treat_counter_reset"} — reset to 0
                try:
                    from services.reward.dispenser import get_dispenser_service
                    dispenser = get_dispenser_service()
                    dispenser.reset_treat_counter()
                    await self._broadcast_contract_status()
                    result = {"success": True, "treats_remaining": 0}
                except Exception as e:
                    result = {"success": False, "error": str(e)}

            elif command in ("treat_unjam", "carousel_rotate"):
                # Anti-jam wiggle sequence
                try:
                    from services.reward.dispenser import get_dispenser_service
                    import threading
                    dispenser = get_dispenser_service()
                    threading.Thread(target=dispenser.anti_jam_wiggle, daemon=True, name="TreatUnjam").start()
                    result = {"success": True, "message": "Unjam sequence started"}
                except Exception as e:
                    result = {"success": False, "error": str(e)}

            elif command == "reload_dogs":
                # App sends dog profiles on connect — also set current dog for voice playback
                self.logger.info(f"[RELOAD_DOGS] Received data: {data}")
                from core.state import get_state
                dog_data = data.get("data", data)
                dog_id = dog_data.get("dog_id") or dog_data.get("id")
                dog_name = dog_data.get("dog_name") or dog_data.get("name")
                self.logger.info(f"[RELOAD_DOGS] Parsed: dog_id={dog_id} dog_name={dog_name}")
                if dog_id:
                    state = get_state()
                    state.set_current_dog(dog_id, dog_name)
                    result["selected_dog"] = {"dog_id": dog_id, "dog_name": dog_name}
                    self.logger.info(f"[RELOAD_DOGS] Set current dog: {dog_name} ({dog_id})")
                result["reloaded"] = True

            elif command == "select_dog":
                # {"command": "select_dog", "data": {"dog_id": "dog_123", "dog_name": "Elsa"}}
                # C3.1: Cache selected dog for per-dog voice playback
                from core.state import get_state
                dog_data = data.get("data", data)  # Support both nested and flat format
                dog_id = dog_data.get("dog_id")
                dog_name = dog_data.get("dog_name", dog_id)
                if dog_id:
                    state = get_state()
                    state.set_current_dog(dog_id, dog_name)
                    result["selected_dog"] = {"dog_id": dog_id, "dog_name": dog_name}
                    self.logger.info(f"[SELECT_DOG] Selected: {dog_name} ({dog_id})")
                else:
                    result["success"] = False
                    result["error"] = "dog_id required"

            elif command == "take_photo":
                # {"command": "take_photo", "with_hud": true}
                with_hud = data.get("with_hud", True)
                await self._handle_take_photo(websocket, with_hud)
                return  # take_photo sends its own response

            elif command == "upload_voice":
                await self._handle_upload_voice(websocket, data)
                return

            elif command == "upload_song":
                await self._handle_upload_song(websocket, data)
                return

            elif command == "download_song":
                await self._handle_download_song(websocket, data)
                return

            elif command == "list_voices":
                await self._handle_list_voices(websocket, data)
                return

            elif command == "delete_voice":
                await self._handle_delete_voice(websocket, data)
                return

            elif command == "delete_dog":
                await self._handle_delete_dog(websocket, data)
                return

            elif command.startswith("controller_"):
                # Remote Bluetooth game-controller pairing over the local-AP
                # socket. Same ControllerManager brain as the relay path; the
                # spontaneous events come back via the emitter registered in
                # handle_websocket(). Params may be nested under "data" or flat.
                from services.control.controller_manager import get_controller_manager
                params = data.get("data")
                if not params:
                    params = {k: v for k, v in data.items()
                              if k not in ("type", "command")}
                ack = get_controller_manager().handle_command(command, params)
                await self.manager.send_to_one(
                    websocket, {"type": "command_ack", "command": command, **ack})
                return

            elif command in ("servo_center", "camera_center", "center"):
                # center() does NOT exist on the pan/tilt service — the only
                # centering method is center_camera() (calibrated center from
                # the per-unit yaml). Calling center() silently failed (caught
                # below), which is why the app's center button did nothing.
                try:
                    from services.motion.pan_tilt import get_pan_tilt_service
                    pan_tilt = get_pan_tilt_service()
                    pan_tilt.center_camera(reason="app_servo_center")
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "list_songs":
                try:
                    from services.media.voice_lookup import get_songs_folder
                    import os
                    dog_id = data.get("dog_id")
                    folder = get_songs_folder(dog_id)
                    songs = [f for f in os.listdir(folder) if f.endswith('.mp3')] if os.path.isdir(folder) else []
                    result = {"success": True, "songs": songs, "folder": folder}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "list_missions":
                try:
                    from orchestrators.mission_engine import get_mission_engine
                    engine = get_mission_engine()
                    missions = engine.get_available_missions()
                    result = {"success": True, "missions": missions}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "cancel_mission":
                try:
                    from orchestrators.mission_engine import get_mission_engine
                    engine = get_mission_engine()
                    success = engine.stop_mission("app_cancelled")
                    result = {"success": success, "reason": "app_cancelled"}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "mission_status":
                try:
                    from orchestrators.mission_engine import get_mission_engine
                    engine = get_mission_engine()
                    status = engine.get_mission_status()
                    result = {"success": True, **status}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            elif command == "play_command":
                # {"command": "play_command", "voice_type": "sit", "dog_id": "dog_123"}
                from services.media.voice_lookup import resolve_voice_file
                voice_type = data.get("voice_type") or data.get("command_type")
                dog_id = data.get("dog_id")
                try:
                    audio_path = resolve_voice_file(voice_type, dog_id_override=dog_id)
                    if audio_path:
                        usb_audio = get_usb_audio_service()
                        if usb_audio and usb_audio.is_initialized:
                            usb_audio.play_file(audio_path)
                            result = {"success": True, "played": audio_path}
                        else:
                            result = {"success": False, "command": command, "error": "USB audio not initialized"}
                    else:
                        result = {"success": False, "command": command, "error": f"Voice not found: {voice_type}"}
                except Exception as e:
                    result = {"success": False, "command": command, "error": str(e)}

            else:
                self.logger.warning(f"Unknown contract command: {command}")
                result = {"success": False, "command": command, "error": f"Unknown command: {command}"}

            # Send response for basic commands (photo/voice/song handlers send their own)
            await self.manager.send_to_one(websocket, {
                "type": "command_response",
                "data": result,
                "timestamp": time.time()
            })

        except Exception as e:
            self.logger.error(f"Contract command error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "command_response",
                "data": {"success": False, "command": command, "error": str(e)},
                "timestamp": time.time()
            })

    async def _handle_take_photo(self, websocket: WebSocket, with_hud: bool = True):
        """Handle take_photo command - capture and send photo via WebSocket"""
        try:
            from services.perception.detector import get_detector_service

            photo_service = get_photo_capture_service()
            detector = get_detector_service()

            # Capture photo with HUD
            result = photo_service.capture_photo(
                with_hud=with_hud,
                detector=detector,
                mission_engine=self.mission_engine
            )

            if result.get("success"):
                # Send photo via WebSocket
                await self.manager.send_to_one(websocket, {
                    "type": "photo",
                    "data": result.get("data"),
                    "timestamp": result.get("timestamp"),
                    "filename": result.get("filename"),
                    "with_hud": with_hud
                })
                self.logger.info(f"Photo sent: {result.get('filename')}")
            else:
                await self.manager.send_to_one(websocket, {
                    "type": "error",
                    "message": f"Photo capture failed: {result.get('error')}",
                    "timestamp": time.time()
                })

        except Exception as e:
            self.logger.error(f"Take photo error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            })

    async def _handle_upload_voice(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle upload_voice command - save custom voice recording"""
        try:
            voice_manager = get_voice_manager()

            command_name = data.get("name")
            dog_id = data.get("dog_id")
            audio_data = data.get("data")  # Base64 encoded

            if not command_name or not dog_id or not audio_data:
                await self.manager.send_to_one(websocket, {
                    "type": "voice_upload_result",
                    "success": False,
                    "error": "Missing required fields: name, dog_id, data"
                })
                return

            # Save the voice file
            result = voice_manager.save_voice_base64(dog_id, command_name, audio_data)

            await self.manager.send_to_one(websocket, {
                "type": "voice_upload_result",
                **result
            })

            if result.get("success"):
                self.logger.info(f"Voice uploaded: dog={dog_id}, command={command_name}")

        except Exception as e:
            self.logger.error(f"Upload voice error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "voice_upload_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_upload_song(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle upload_song command - save song to default songs folder"""
        import base64
        import re

        filename = data.get("filename", "")
        audio_data = data.get("data", "")

        try:
            # Validate filename
            if not filename or not re.match(r'^[\w\-. ]+\.(mp3|wav|ogg)$', filename, re.IGNORECASE):
                await self.manager.send_to_one(websocket, {
                    "type": "upload_error",
                    "filename": filename,
                    "success": False,
                    "error": "Invalid filename. Use alphanumeric characters, hyphens, underscores, spaces. Must end in .mp3, .wav, or .ogg"
                })
                return

            if not audio_data:
                await self.manager.send_to_one(websocket, {
                    "type": "upload_error",
                    "filename": filename,
                    "success": False,
                    "error": "Missing audio data"
                })
                return

            # Decode base64 audio
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                await self.manager.send_to_one(websocket, {
                    "type": "upload_error",
                    "filename": filename,
                    "success": False,
                    "error": f"Invalid base64 data: {e}"
                })
                return

            # Save to default songs folder
            songs_dir = "/home/morgan/dogbot/VOICEMP3/songs/default"
            os.makedirs(songs_dir, exist_ok=True)
            filepath = os.path.join(songs_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(audio_bytes)

            # Refresh playlist
            usb_audio = get_usb_audio_service()
            usb_audio.refresh_playlist()

            file_size = len(audio_bytes)
            self.logger.info(f"Song uploaded via WebSocket: {filename} ({file_size} bytes)")

            await self.manager.send_to_one(websocket, {
                "type": "upload_complete",
                "filename": filename,
                "success": True,
                "size_bytes": file_size,
                "path": f"default/{filename}"
            })

        except IOError as e:
            error_msg = "Disk full" if "No space" in str(e) else str(e)
            self.logger.error(f"Song upload IO error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "upload_error",
                "filename": filename,
                "success": False,
                "error": error_msg
            })
        except Exception as e:
            self.logger.error(f"Song upload error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "upload_error",
                "filename": filename,
                "success": False,
                "error": str(e)
            })

    async def _handle_download_song(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle download_song command - download MP3 from URL instead of base64 transfer

        Downloads file directly on robot side to avoid large WebSocket payloads.
        Command: {"command": "download_song", "url": "https://...", "filename": "my_song.mp3"}
        Response: {"type": "download_complete", "filename": "...", "success": true, "size_bytes": ...}
        """
        import re
        import httpx

        url = data.get("url", "")
        filename = data.get("filename", "")

        try:
            # Validate URL
            if not url or not url.startswith(('http://', 'https://')):
                await self.manager.send_to_one(websocket, {
                    "type": "download_error",
                    "filename": filename,
                    "success": False,
                    "error": "Invalid URL. Must start with http:// or https://"
                })
                return

            # Validate filename
            if not filename or not re.match(r'^[\w\-. ]+\.(mp3|wav|ogg)$', filename, re.IGNORECASE):
                await self.manager.send_to_one(websocket, {
                    "type": "download_error",
                    "filename": filename,
                    "success": False,
                    "error": "Invalid filename. Use alphanumeric characters, hyphens, underscores, spaces. Must end in .mp3, .wav, or .ogg"
                })
                return

            # Download file with timeout (60 seconds for large files)
            self.logger.info(f"Downloading song from URL: {url}")
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                audio_bytes = response.content

            # Check file size (max 20MB)
            file_size = len(audio_bytes)
            if file_size > 20 * 1024 * 1024:
                await self.manager.send_to_one(websocket, {
                    "type": "download_error",
                    "filename": filename,
                    "success": False,
                    "error": f"File too large: {file_size // (1024*1024)}MB. Maximum is 20MB"
                })
                return

            # Save to default songs folder
            songs_dir = "/home/morgan/dogbot/VOICEMP3/songs/default"
            os.makedirs(songs_dir, exist_ok=True)
            filepath = os.path.join(songs_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(audio_bytes)

            # Refresh playlist
            usb_audio = get_usb_audio_service()
            usb_audio.refresh_playlist()

            self.logger.info(f"Song downloaded: {filename} ({file_size} bytes) from {url}")

            await self.manager.send_to_one(websocket, {
                "type": "download_complete",
                "filename": filename,
                "success": True,
                "size_bytes": file_size,
                "path": f"default/{filename}"
            })

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Song download HTTP error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "download_error",
                "filename": filename,
                "success": False,
                "error": f"HTTP error: {e.response.status_code}"
            })
        except httpx.TimeoutException:
            self.logger.error(f"Song download timeout for {url}")
            await self.manager.send_to_one(websocket, {
                "type": "download_error",
                "filename": filename,
                "success": False,
                "error": "Download timeout (60 seconds exceeded)"
            })
        except IOError as e:
            error_msg = "Disk full" if "No space" in str(e) else str(e)
            self.logger.error(f"Song download IO error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "download_error",
                "filename": filename,
                "success": False,
                "error": error_msg
            })
        except Exception as e:
            self.logger.error(f"Song download error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "download_error",
                "filename": filename,
                "success": False,
                "error": str(e)
            })

    async def _handle_list_voices(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle list_voices command - return available custom voices"""
        try:
            voice_manager = get_voice_manager()

            dog_id = data.get("dog_id")

            if dog_id:
                # List voices for specific dog
                result = voice_manager.list_voices(dog_id)
            else:
                # List all dogs' voices
                result = voice_manager.get_all_dogs_voices()

            await self.manager.send_to_one(websocket, {
                "type": "voice_list",
                **result
            })

        except Exception as e:
            self.logger.error(f"List voices error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "voice_list",
                "error": str(e)
            })

    async def _handle_delete_voice(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle delete_voice command - remove custom voice recording"""
        try:
            voice_manager = get_voice_manager()

            command_name = data.get("name")
            dog_id = data.get("dog_id")

            if not command_name or not dog_id:
                await self.manager.send_to_one(websocket, {
                    "type": "voice_delete_result",
                    "success": False,
                    "error": "Missing required fields: name, dog_id"
                })
                return

            result = voice_manager.delete_voice(dog_id, command_name)

            await self.manager.send_to_one(websocket, {
                "type": "voice_delete_result",
                **result
            })

            if result.get("success"):
                self.logger.info(f"Voice deleted: dog={dog_id}, command={command_name}")

        except Exception as e:
            self.logger.error(f"Delete voice error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "voice_delete_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_delete_dog(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle delete_dog command - remove all custom resources for a dog

        Called when a dog is deleted from the system.
        Cleans up:
        - Custom voice recordings (VOICEMP3/talks/dog_{id}/)
        - Future: could also clean up photos, mission history, etc.
        """
        try:
            voice_manager = get_voice_manager()

            dog_id = data.get("dog_id")

            if not dog_id:
                await self.manager.send_to_one(websocket, {
                    "type": "dog_delete_result",
                    "success": False,
                    "error": "Missing required field: dog_id"
                })
                return

            # Delete all voices for this dog
            result = voice_manager.delete_dog_voices(dog_id)

            await self.manager.send_to_one(websocket, {
                "type": "dog_delete_result",
                **result
            })

            if result.get("success"):
                self.logger.info(f"Dog resources deleted: dog_id={dog_id}, files={result.get('files_deleted', 0)}")

        except Exception as e:
            self.logger.error(f"Delete dog error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "dog_delete_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_audio_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle audio_message - play audio received from app (push-to-talk)

        Expected format:
        {"type": "audio_message", "data": "<base64>", "format": "aac"}
        """
        try:
            ptt_service = get_push_to_talk_service()

            audio_data = data.get("data")
            audio_format = data.get("format", "aac")

            if not audio_data:
                self.logger.warning("[PTT] Local WS audio message missing data field")
                await self.manager.send_to_one(websocket, {
                    "type": "audio_played",
                    "success": False,
                    "error": "Missing audio data"
                })
                return

            self.logger.info(f"[PTT] Received from local WS: format={audio_format}, base64_len={len(audio_data)}")

            # Play the audio
            result = ptt_service.play_audio_base64(audio_data, audio_format)

            await self.manager.send_to_one(websocket, {
                "type": "audio_played",
                **result
            })

            if result.get("success"):
                self.logger.info(f"[PTT] Local WS ACK sent (size={result.get('size_bytes')} bytes)")
            else:
                self.logger.error(f"[PTT] Local WS playback failed: {result.get('error')}")

        except Exception as e:
            self.logger.error(f"[PTT] Local WS audio error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "audio_played",
                "success": False,
                "error": str(e)
            })

    async def _handle_audio_request(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle audio_request - record from mic and send back (listen feature)

        Expected format:
        {"type": "audio_request", "duration": 5, "format": "aac"}
        """
        try:
            ptt_service = get_push_to_talk_service()

            duration = data.get("duration", 5)
            audio_format = data.get("format", "aac")

            # Record audio
            self.logger.info(f"PTT recording requested: {duration}s, format={audio_format}")
            result = ptt_service.record_audio(duration=duration, format=audio_format)

            if result.get("success"):
                await self.manager.send_to_one(websocket, {
                    "type": "audio_message",
                    "data": result.get("data"),
                    "format": audio_format,
                    "duration_ms": result.get("duration_ms"),
                    "size_bytes": result.get("size_bytes")
                })
                self.logger.info(f"PTT audio sent: {result.get('size_bytes')} bytes")
            else:
                await self.manager.send_to_one(websocket, {
                    "type": "audio_error",
                    "error": result.get("error")
                })

        except Exception as e:
            self.logger.error(f"Audio request error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "audio_error",
                "error": str(e)
            })

    async def _handle_webrtc_request(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle WebRTC stream request (local mode signaling via /ws)

        Robot creates peer connection + offer and sends back to app.
        Local mode: no TURN servers needed (same LAN).
        """
        session_id = data.get("session_id", "local")
        self.logger.info(f"[LOCAL] WebRTC request via /ws: session={session_id}")

        try:
            from services.streaming.webrtc import get_webrtc_service
            webrtc = get_webrtc_service()

            # ICE candidate callback — trickle candidates back over this WS
            async def on_ice_candidate(candidate):
                if candidate:
                    try:
                        await self.manager.send_to_one(websocket, {
                            "type": "webrtc_ice",
                            "session_id": session_id,
                            "candidate": {
                                "candidate": candidate.candidate,
                                "sdpMid": candidate.sdpMid,
                                "sdpMLineIndex": candidate.sdpMLineIndex
                            }
                        })
                    except Exception:
                        pass

            # Local LAN: no STUN/TURN needed — both devices are on the same AP network.
            # Using an empty list forces host-only ICE candidates (direct LAN IPs).
            # External STUN (e.g. stun.l.google.com) hangs when there's no internet.
            local_ice_servers = []

            offer = await webrtc.create_offer(
                session_id=session_id,
                ice_servers=local_ice_servers,
                on_ice_candidate=on_ice_candidate
            )

            await self.manager.send_to_one(websocket, {
                "type": "webrtc_credentials",
                "session_id": session_id,
                "ice_servers": {"iceServers": local_ice_servers}
            })
            await self.manager.send_to_one(websocket, {
                "type": "webrtc_offer",
                "session_id": session_id,
                "sdp": offer
            })
            self.logger.info(f"[LOCAL] WebRTC offer sent for session {session_id}")

        except Exception as e:
            self.logger.error(f"[LOCAL] WebRTC request error: {e}")
            await self.manager.send_to_one(websocket, {
                "type": "webrtc_error",
                "session_id": session_id,
                "error": str(e)
            })

    async def _handle_webrtc_answer(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle SDP answer from app (local mode signaling via /ws)"""
        session_id = data.get("session_id", "local")
        sdp = data.get("sdp")
        self.logger.info(f"[LOCAL] WebRTC answer received: session={session_id}")

        try:
            from services.streaming.webrtc import get_webrtc_service
            webrtc = get_webrtc_service()
            await webrtc.handle_answer(session_id, sdp)
        except Exception as e:
            self.logger.error(f"[LOCAL] WebRTC answer error: {e}")

    async def _handle_webrtc_ice(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle ICE candidate from app (local mode signaling via /ws)"""
        session_id = data.get("session_id", "local")
        candidate = data.get("candidate")

        try:
            from services.streaming.webrtc import get_webrtc_service
            webrtc = get_webrtc_service()
            await webrtc.add_ice_candidate(session_id, candidate)
        except Exception as e:
            self.logger.error(f"[LOCAL] WebRTC ICE error: {e}")

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
            # {"count": N, "reason": "...", "dog_id": "..."}
            # Bug fix: count was being passed as the first positional arg,
            # which is dog_id — every multi-dispense request silently dispensed
            # one treat and mis-attributed it to a dog named "3"/"5"/etc.
            count = max(1, min(10, int(params.get("count", 1))))
            reason = params.get("reason", "manual")
            dog_id = params.get("dog_id")

            if not self.dispenser:
                return {"success": False, "error": "Dispenser service not available"}

            if count == 1:
                success = self.dispenser.dispense_treat(dog_id=dog_id, reason=reason)
                return {"success": success, "count": 1, "reason": reason}

            # Multi-dispense blocks ~1.5s per treat — run off the WS thread.
            # Per-treat progress is published via the treat_dispensed bus event,
            # so the app can update its UI as each one lands.
            threading.Thread(
                target=self.dispenser.dispense_multiple,
                kwargs={"count": count, "dog_id": dog_id, "reason": reason},
                daemon=True, name="TreatMultiDispense",
            ).start()
            return {"success": True, "count": count, "reason": reason, "async": True}

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

        elif command == "cancel_mission":
            success = self.mission_engine.stop_mission("app_cancelled")
            return {"success": success, "reason": "app_cancelled"}

        elif command == "mission_status":
            status = self.mission_engine.get_mission_status()
            return {"success": True, **status}

        elif command == "list_missions":
            missions = self.mission_engine.get_available_missions()
            return {"success": True, "missions": missions}

        elif command == "servo_center":
            try:
                from services.motion.pan_tilt import get_pan_tilt_service
                pan_tilt = get_pan_tilt_service()
                pan_tilt.center()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif command == "list_songs":
            try:
                from services.media.voice_lookup import get_songs_folder
                import os
                dog_id = data.get("dog_id")
                folder = get_songs_folder(dog_id)
                songs = [f for f in os.listdir(folder) if f.endswith('.mp3')] if os.path.isdir(folder) else []
                return {"success": True, "songs": songs, "folder": folder}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif command == "play_command":
            # {"command": "play_command", "voice_type": "sit", "dog_id": "dog_123"}
            from services.media.voice_lookup import resolve_voice_file
            voice_type = data.get("voice_type") or data.get("command_type")
            dog_id = data.get("dog_id")
            try:
                audio_path = resolve_voice_file(voice_type, dog_id_override=dog_id)
                if audio_path:
                    usb_audio = get_usb_audio_service()
                    if usb_audio and usb_audio.is_initialized:
                        usb_audio.play_file(audio_path)
                        return {"success": True, "played": audio_path}
                return {"success": False, "error": f"Voice not found: {voice_type}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif command == "treat_counter_set":
            # {"command": "treat_counter_set", "count": 44}
            count = data.get("count")
            if count is not None:
                try:
                    self.dispenser.set_treat_count(int(count))
                    remaining = self.dispenser.treats_remaining
                    # Send immediate status update so app display refreshes
                    await self._broadcast_contract_event("treat_counter", {
                        "remaining": remaining,
                        "set_to": int(count)
                    })
                    return {"success": True, "treats_remaining": remaining}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "count required"}

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
            status_counter = 0
            while True:
                telemetry = await self._collect_telemetry()

                # Legacy format (every second)
                await self.manager.send_to_all({
                    "type": "telemetry",
                    "data": telemetry,
                    "timestamp": time.time()
                })

                # Contract format status event (every 5 seconds)
                status_counter += 1
                if status_counter >= 5:
                    status_counter = 0
                    await self._broadcast_contract_status()

                await asyncio.sleep(1.0)  # Send every second

        except asyncio.CancelledError:
            self.logger.info("Telemetry loop cancelled")
        except Exception as e:
            self.logger.error(f"Telemetry loop error: {e}")

    def _get_treats_remaining(self) -> int:
        """Get actual treats remaining from dispenser service"""
        try:
            from services.reward.dispenser import get_dispenser_service
            return get_dispenser_service().treats_remaining
        except Exception:
            return 0

    async def _broadcast_contract_status(self):
        """Broadcast status in API contract format"""
        from datetime import datetime
        try:
            state = self.state.get_full_state()

            # Get battery percentage
            battery_voltage = state.get("hardware", {}).get("battery_voltage", 0)
            battery_pct = ((battery_voltage - 12.0) / 4.8 * 100) if battery_voltage else 0
            battery_pct = min(100, max(0, battery_pct))

            # Map internal mode to contract mode
            mode_map = {
                "idle": "idle",
                "silent_guardian": "silent_guardian",
                "coach": "coach",
                "mission": "mission",
                "manual": "manual",
                "photography": "manual",
                "emergency": "manual"
            }
            internal_mode = state.get("mode", "idle")
            contract_mode = mode_map.get(internal_mode, "idle")

            await self.manager.send_to_all({
                "event": "status",
                "data": {
                    "battery": round(battery_pct, 1),
                    "temperature": state.get("hardware", {}).get("cpu_temp", 0),
                    "mode": contract_mode,
                    "is_charging": state.get("hardware", {}).get("is_charging", False),
                    "treats_remaining": self._get_treats_remaining()
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        except Exception as e:
            self.logger.error(f"Contract status broadcast error: {e}")

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

    # Event handlers - broadcast in both legacy and contract formats
    def _register_controller_emitter(self):
        """Wire ControllerManager events to local-AP websocket clients (once).

        Runs inside the websocket coroutine, so we capture the live api event
        loop here and use run_coroutine_threadsafe to push from the manager's
        worker thread. Idempotent — register_emitter dedupes anyway."""
        if getattr(self, '_controller_emitter_registered', False):
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            from services.control.controller_manager import get_controller_manager
            mgr = get_controller_manager()

            def _emit(payload):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.manager.send_to_all(payload), loop)
                except Exception:
                    pass

            mgr.register_emitter(_emit)
            # Ensure the manager is running even if main() never started it
            # (e.g. relay disabled but local-AP active).
            mgr.start()
            self._controller_emitter_registered = True
            self.logger.info("Controller pairing events wired to local-AP")
        except Exception as e:
            self.logger.debug(f"controller emitter register failed: {e}")

    async def _broadcast_contract_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event in API contract format"""
        from datetime import datetime
        await self.manager.send_to_all({
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    async def _on_vision_event(self, event):
        """Handle vision events"""
        # Legacy format
        await self.manager.send_to_all({
            "type": "event",
            "category": "vision",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

        # Contract format for dog detection
        if event.subtype == "dog_detected":
            await self._broadcast_contract_event("detection", {
                "detected": True,
                "behavior": event.data.get("behavior", "unknown"),
                "confidence": event.data.get("confidence", 0.0),
                "bbox": event.data.get("bbox", [0, 0, 0, 0])
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
        # Legacy format
        await self.manager.send_to_all({
            "type": "event",
            "category": "reward",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

        # Contract format for treat dispensed
        if event.subtype == "treat_dispensed":
            await self._broadcast_contract_event("treat", {
                "dispensed": True,
                "remaining": event.data.get("remaining", 0)
            })

    async def _on_system_event(self, event):
        """Handle system events"""
        # Legacy format
        await self.manager.send_to_all({
            "type": "event",
            "category": "system",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

        # Contract format: mode_changed (same format as relay sends)
        if event.subtype == "mode_changed":
            mode_map = {
                "idle": "idle", "silent_guardian": "silent_guardian",
                "coach": "coach", "mission": "mission",
                "manual": "manual", "photography": "manual",
                "emergency": "manual"
            }
            new_mode = event.data.get("to_mode", event.data.get("mode", "idle"))
            prev_mode = event.data.get("from_mode", event.data.get("previous_mode", "idle"))
            await self.manager.send_to_all({
                "event": "mode_changed",
                "mode": mode_map.get(new_mode, new_mode),
                "previous_mode": mode_map.get(prev_mode, prev_mode),
                "locked": event.data.get("locked", False),
                "reason": event.data.get("reason", "unknown"),
                "timestamp": time.time()
            })

        # Contract format: battery_status
        elif event.subtype == "battery_status":
            voltage = event.data.get("voltage", 0)
            pct = ((voltage - 12.0) / 4.8 * 100) if voltage else 0
            pct = min(100, max(0, pct))
            await self._broadcast_contract_event("battery", {
                "level": round(pct, 1),
                "voltage": round(voltage, 2),
                "charging": event.data.get("charging", False)
            })

        # Contract format for errors
        elif event.subtype in ["low_battery", "overheat", "motor_fault", "camera_fault", "network_error"]:
            error_codes = {
                "low_battery": "LOW_BATTERY",
                "overheat": "OVERHEAT",
                "motor_fault": "MOTOR_FAULT",
                "camera_fault": "CAMERA_FAULT",
                "network_error": "NETWORK_ERROR"
            }
            await self._broadcast_contract_event("error", {
                "code": error_codes.get(event.subtype, "UNKNOWN"),
                "message": event.data.get("message", str(event.subtype)),
                "severity": event.data.get("severity", "warning")
            })

    async def _on_mission_event(self, event):
        """Handle mission events"""
        # Legacy format
        await self.manager.send_to_all({
            "type": "event",
            "category": "mission",
            "data": {
                "subtype": event.subtype,
                "data": event.data
            },
            "timestamp": time.time()
        })

        # Contract format for mission updates
        if event.subtype in ["mission_started", "mission_progress", "mission_completed"]:
            await self._broadcast_contract_event("mission", {
                "id": event.data.get("mission_id", "unknown"),
                "status": event.data.get("status", "running"),
                "progress": event.data.get("progress", 0.0),
                "rewards_given": event.data.get("rewards_given", 0),
                "success_count": event.data.get("success_count", 0),
                "fail_count": event.data.get("fail_count", 0)
            })


# Global WebSocket server instance
_websocket_server = None

def get_websocket_server():
    """Get the global WebSocket server instance"""
    global _websocket_server
    if _websocket_server is None:
        _websocket_server = TreatBotWebSocketServer()
    return _websocket_server