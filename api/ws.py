#!/usr/bin/env python3
"""
WebSocket server for real-time communication with web dashboard
Provides telemetry streaming, event notifications, and bidirectional control
"""

import asyncio
import json
import logging
import os
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

            # Handle ping/pong (API Contract)
            if data.get("type") == "ping":
                await self.manager.send_to_one(websocket, {"type": "pong"})
                return

            # Handle auth (API Contract - for cloud mode, local mode ignores)
            if data.get("type") == "auth":
                await self.manager.send_to_one(websocket, {"type": "auth_result", "success": True})
                return

            # Handle push-to-talk audio message (play from app)
            if data.get("type") == "audio_message":
                await self._handle_audio_message(websocket, data)
                return

            # Handle audio request (record and send back)
            if data.get("type") == "audio_request":
                await self._handle_audio_request(websocket, data)
                return

            # Handle contract-style commands (API Contract format)
            if "command" in data and "params" not in data:
                # This is an API contract format command
                response = await self._execute_contract_command(websocket, data)
                return

            # Handle legacy format with command + params
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

    async def _execute_contract_command(self, websocket: WebSocket, data: Dict[str, Any]):
        """Execute API contract format commands"""
        command = data.get("command")
        self._get_services()

        try:
            if command == "motor":
                # {"command": "motor", "left": 0.5, "right": 0.5}
                left = data.get("left", 0.0)
                right = data.get("right", 0.0)
                if self.motor:
                    left_pct = int(left * 100)
                    right_pct = int(right * 100)
                    self.motor.set_speed(left_pct, right_pct)

            elif command == "servo":
                # {"command": "servo", "pan": 15.0, "tilt": -10.0}
                pan = data.get("pan")
                tilt = data.get("tilt")
                if self.pantilt:
                    pan_internal = int(90 + pan) if pan is not None else None
                    tilt_internal = int(90 + tilt) if tilt is not None else None
                    self.pantilt.move_camera(pan=pan_internal, tilt=tilt_internal)

            elif command == "treat":
                # {"command": "treat"}
                if self.dispenser:
                    self.dispenser.dispense_treat()

            elif command == "led":
                # {"command": "led", "pattern": "celebration"}
                pattern = data.get("pattern", "idle")
                if self.led:
                    pattern_map = {
                        "breathing": "idle",
                        "rainbow": "gradient_flow",
                        "celebration": "treat_launching",
                        "searching": "searching",
                        "alert": "error",
                        "idle": "idle"
                    }
                    mode = pattern_map.get(pattern, pattern)
                    self.led.set_pattern(mode)

            elif command == "audio":
                # {"command": "audio", "file": "good.mp3"}
                audio_file = data.get("file")
                if audio_file and self.sfx:
                    self.sfx.play_sound(audio_file)

            elif command == "mode":
                # {"command": "mode", "mode": "training"}
                mode = data.get("mode")
                if mode:
                    from orchestrators.mode_fsm import get_mode_fsm
                    from core.state import SystemMode
                    mode_fsm = get_mode_fsm()
                    # Map contract modes to internal modes
                    mode_map = {
                        "idle": SystemMode.IDLE,
                        "guardian": SystemMode.SILENT_GUARDIAN,
                        "training": SystemMode.COACH,
                        "mission": SystemMode.MISSION,
                        "manual": SystemMode.MANUAL,
                        "docking": SystemMode.IDLE  # No docking mode yet
                    }
                    internal_mode = mode_map.get(mode)
                    if internal_mode:
                        mode_fsm.force_mode(internal_mode, "websocket_command")

            elif command == "take_photo":
                # {"command": "take_photo", "with_hud": true}
                with_hud = data.get("with_hud", True)
                await self._handle_take_photo(websocket, with_hud)

            elif command == "upload_voice":
                # {"command": "upload_voice", "name": "sit", "dog_id": "1", "data": "<base64>"}
                await self._handle_upload_voice(websocket, data)

            elif command == "upload_song":
                # {"command": "upload_song", "filename": "my_song.mp3", "data": "<base64>"}
                await self._handle_upload_song(websocket, data)

            elif command == "download_song":
                # BUILD 38: {"command": "download_song", "url": "https://...", "filename": "my_song.mp3"}
                # Downloads MP3 via HTTP instead of receiving base64 (avoids 5MB WebSocket crash)
                await self._handle_download_song(websocket, data)

            elif command == "list_voices":
                # {"command": "list_voices", "dog_id": "1"}
                await self._handle_list_voices(websocket, data)

            elif command == "delete_voice":
                # {"command": "delete_voice", "name": "sit", "dog_id": "1"}
                await self._handle_delete_voice(websocket, data)

            elif command == "delete_dog":
                # {"command": "delete_dog", "dog_id": "1"}
                # Deletes all custom voices for a dog when dog is removed
                await self._handle_delete_dog(websocket, data)

        except Exception as e:
            self.logger.error(f"Contract command error: {e}")

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
        """Handle upload_song command - save song to default songs folder

        BUILD 35: Added WebSocket handler for song uploads per app team request.
        Sends back upload_complete or upload_error response.
        """
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

        BUILD 38: Added to fix 5MB WebSocket crash. Downloads file directly on robot side.
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
                await self.manager.send_to_one(websocket, {
                    "type": "audio_played",
                    "success": False,
                    "error": "Missing audio data"
                })
                return

            # Play the audio
            result = ptt_service.play_audio_base64(audio_data, audio_format)

            await self.manager.send_to_one(websocket, {
                "type": "audio_played",
                **result
            })

            if result.get("success"):
                self.logger.info(f"PTT audio played: {result.get('size_bytes')} bytes")

        except Exception as e:
            self.logger.error(f"Audio message error: {e}")
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

    async def _broadcast_contract_status(self):
        """Broadcast status in API contract format"""
        from datetime import datetime
        try:
            state = self.state.get_full_state()

            # Get battery percentage
            battery_voltage = state.get("hardware", {}).get("battery_voltage", 0)
            battery_pct = (battery_voltage / 16.8 * 100) if battery_voltage else 0
            battery_pct = min(100, max(0, battery_pct))

            # Map internal mode to contract mode
            mode_map = {
                "idle": "idle",
                "silent_guardian": "guardian",
                "coach": "training",
                "mission": "mission",  # BUILD 35: Add missing mission mode mapping
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
                    "treats_remaining": 15  # TODO: Get from dispenser
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

        # Contract format for errors
        if event.subtype in ["low_battery", "overheat", "motor_fault", "camera_fault", "network_error"]:
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