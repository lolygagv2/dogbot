#!/usr/bin/env python3
"""
Cloud Relay WebSocket Client for WIM-Z

Maintains persistent connection to the cloud relay server for:
- WebRTC signaling (video streaming to mobile app)
- Command forwarding
- Event broadcasting
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import aiohttp

from core.bus import get_bus, CloudEvent
from core.state import get_state, SystemMode

# Robot version for connection announcement
ROBOT_VERSION = "1.0.0"


@dataclass
class RelayConfig:
    """Cloud relay configuration"""
    enabled: bool = False
    relay_url: str = "wss://api.wimzai.com/ws/device"
    device_id: str = ""
    device_secret: str = ""
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0
    connect_timeout: float = 10.0


class RelayClient:
    """
    WebSocket client for cloud relay server

    Handles:
    - Authentication via HMAC signature
    - WebRTC signaling messages
    - Command forwarding to robot
    - Event broadcasting to cloud
    - Auto-reconnection on disconnect
    """

    def __init__(self, config: RelayConfig):
        self.config = config
        self.logger = logging.getLogger('RelayClient')
        self.bus = get_bus()
        self.state = get_state()

        # Connection state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._running = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # WebRTC service reference (set during integration)
        self._webrtc_service = None

        # Track app connection state (informational only - no mode changes on disconnect)
        self._app_connected = False

        # Message handlers
        self._message_handlers: Dict[str, Callable] = {
            'webrtc_request': self._handle_webrtc_request,
            'webrtc_answer': self._handle_webrtc_answer,
            'webrtc_ice': self._handle_webrtc_ice,
            'webrtc_close': self._handle_webrtc_close,
            'command': self._handle_command,
            'ping': self._handle_ping,
            'profiles': self._handle_profiles,
            'audio_message': self._handle_audio_message,
            'audio_request': self._handle_audio_request,
            'user_connected': self._handle_user_connected,
            'user_disconnected': self._handle_user_disconnected,
        }

        # Track connected user
        self._connected_user_id: Optional[str] = None

        # Profile manager reference
        self._profile_manager = None

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # Send failure tracking - suppress repeated log spam
        self._send_fail_count = 0
        self._send_fail_logged = False

        # Message queue for offline buffering (limited)
        self._message_queue: list = []
        self._max_queue_size = 50

        self.logger.info(f"RelayClient initialized (enabled={config.enabled})")

    def set_webrtc_service(self, webrtc_service):
        """Set WebRTC service reference for signaling"""
        self._webrtc_service = webrtc_service
        self.logger.info("WebRTC service connected to relay client")

    def _generate_auth_signature(self, timestamp: int) -> str:
        """Generate HMAC signature for authentication"""
        message = f"{self.config.device_id}:{timestamp}"
        signature = hmac.new(
            self.config.device_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _connect(self) -> bool:
        """Establish WebSocket connection to relay server"""
        if not self.config.enabled:
            self.logger.debug("Relay client disabled")
            return False

        if not self.config.device_id or not self.config.device_secret:
            self.logger.warning("Missing device credentials - relay disabled")
            return False

        try:
            # Create session if needed
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            # Generate auth payload
            timestamp = int(time.time())
            signature = self._generate_auth_signature(timestamp)

            # Connect with auth headers
            headers = {
                'X-Device-ID': self.config.device_id,
                'X-Timestamp': str(timestamp),
                'X-Signature': signature,
            }

            self.logger.info(f"Connecting to relay: {self.config.relay_url}")

            self._ws = await self._session.ws_connect(
                self.config.relay_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.connect_timeout),
                heartbeat=self.config.heartbeat_interval
            )

            self._connected = True
            self.logger.info("Connected to cloud relay")
            print("[RelayClient] ✅ Connected to cloud relay", flush=True)

            # Send robot_connected announcement
            await self._send({
                'type': 'robot_connected',
                'device_id': self.config.device_id,
                'version': ROBOT_VERSION
            })

            # Request dog profiles from cloud
            await self.request_profiles()

            return True

        except aiohttp.ClientError as e:
            self.logger.error(f"Connection failed: {e}")
            self._connected = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {e}")
            self._connected = False
            return False

    async def _disconnect(self, send_goodbye: bool = False):
        """Close WebSocket connection

        Args:
            send_goodbye: If True, send robot_disconnecting before closing
        """
        # Send goodbye message if requested and still connected
        if send_goodbye and self._connected and self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({
                    'type': 'robot_disconnecting',
                    'device_id': self.config.device_id
                })
                self.logger.info("Sent robot_disconnecting to relay")
            except Exception as e:
                self.logger.debug(f"Could not send goodbye: {e}")

        self._connected = False

        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as e:
                self.logger.debug(f"Error closing WebSocket: {e}")

        self._ws = None
        self.logger.info("Disconnected from relay")

    async def _send(self, message: dict) -> bool:
        """Send message to relay server"""
        if not self._connected or not self._ws:
            # Queue message for later if not connected (limited buffer)
            if len(self._message_queue) < self._max_queue_size:
                self._message_queue.append(message)
            return False

        # Check if websocket is closing/closed
        if self._ws.closed:
            self._connected = False
            if len(self._message_queue) < self._max_queue_size:
                self._message_queue.append(message)
            return False

        try:
            await self._ws.send_json(message)
            # Reset failure tracking on success
            if self._send_fail_count > 0:
                self.logger.info(f"Send recovered after {self._send_fail_count} failures")
            self._send_fail_count = 0
            self._send_fail_logged = False
            return True
        except Exception as e:
            self._send_fail_count += 1
            # Only log first failure, then suppress until recovered
            if not self._send_fail_logged:
                self.logger.warning(f"Send failed (will suppress repeats): {e}")
                self._send_fail_logged = True
            self._connected = False
            return False

    async def _receive_loop(self):
        """Main message receive loop"""
        while self._running and self._connected and self._ws:
            try:
                msg = await self._ws.receive(timeout=60.0)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.info("WebSocket closed by server")
                    self._connected = False
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {self._ws.exception()}")
                    self._connected = False
                    break

            except asyncio.TimeoutError:
                # Timeout is normal, continue loop
                continue
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON received: {e}")
            except Exception as e:
                self.logger.error(f"Receive error: {e}")
                self._connected = False
                break

    async def _heartbeat_loop(self):
        """Send heartbeat every 30 seconds while connected"""
        while self._running and self._connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self._connected:
                    await self._send({
                        'type': 'heartbeat',
                        'device_id': self.config.device_id,
                        'timestamp': int(time.time())
                    })
                    self.logger.debug("💓 Heartbeat sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Heartbeat error: {e}")

    async def _handle_message(self, data: dict):
        """Route incoming message to appropriate handler"""
        msg_type = data.get('type')
        msg_device_id = data.get('device_id')

        # Log at INFO level to ensure visibility
        self.logger.info(f"📥 Relay message received: type={msg_type}, device_id={msg_device_id}")
        print(f"[RelayClient] 📥 Message: type={msg_type}, device_id={msg_device_id}, data={data}", flush=True)

        # Filter by device_id if present - ignore messages for other robots
        if msg_device_id and msg_device_id != self.config.device_id:
            self.logger.debug(f"Ignoring message for different device: {msg_device_id}")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"Handler error for {msg_type}: {e}")
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_webrtc_request(self, data: dict):
        """Handle WebRTC stream request from app via relay

        Note: App sends set_mode command separately - we just track the connection here.
        """
        session_id = data.get('session_id')
        ice_servers = data.get('ice_servers', {})

        self.logger.info(f"WebRTC request received: session={session_id}")
        print(f"[RelayClient] 🎥 WebRTC request: session={session_id}, ice_servers={ice_servers}", flush=True)

        # Track that app is connected (mode change comes via set_mode command)
        self._app_connected = True

        if not self._webrtc_service:
            self.logger.error("WebRTC service not available")
            await self._send({
                'type': 'webrtc_error',
                'device_id': self.config.device_id,
                'session_id': session_id,
                'error': 'WebRTC service not available'
            })
            return

        try:
            # Create ICE candidate callback
            async def on_ice_candidate(candidate):
                if candidate:
                    await self._send({
                        'type': 'webrtc_ice',
                        'device_id': self.config.device_id,
                        'session_id': session_id,
                        'candidate': {
                            'candidate': candidate.candidate,
                            'sdpMid': candidate.sdpMid,
                            'sdpMLineIndex': candidate.sdpMLineIndex
                        }
                    })

            # Create offer
            offer = await self._webrtc_service.create_offer(
                session_id=session_id,
                ice_servers=ice_servers,
                on_ice_candidate=on_ice_candidate
            )

            # Send offer to relay
            await self._send({
                'type': 'webrtc_offer',
                'device_id': self.config.device_id,
                'session_id': session_id,
                'sdp': offer
            })

            self.logger.info(f"Sent WebRTC offer for session {session_id}")
            print(f"[RelayClient] ✅ Sent WebRTC offer for {session_id}", flush=True)

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to create WebRTC offer: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"[RelayClient] ❌ WebRTC offer failed: {e}", flush=True)
            print(f"[RelayClient] Traceback: {traceback.format_exc()}", flush=True)
            await self._send({
                'type': 'webrtc_error',
                'device_id': self.config.device_id,
                'session_id': session_id,
                'error': str(e)
            })

    async def _handle_webrtc_answer(self, data: dict):
        """Handle SDP answer from app"""
        session_id = data.get('session_id')
        sdp = data.get('sdp')

        self.logger.info(f"WebRTC answer received: session={session_id}")

        if not self._webrtc_service:
            self.logger.error("WebRTC service not available")
            return

        try:
            await self._webrtc_service.handle_answer(session_id, sdp)
        except Exception as e:
            self.logger.error(f"Failed to handle answer: {e}")

    async def _handle_webrtc_ice(self, data: dict):
        """Handle ICE candidate from app"""
        session_id = data.get('session_id')
        candidate = data.get('candidate')

        self.logger.debug(f"ICE candidate received: session={session_id}")

        if not self._webrtc_service:
            return

        try:
            await self._webrtc_service.add_ice_candidate(session_id, candidate)
        except Exception as e:
            self.logger.error(f"Failed to add ICE candidate: {e}")

    async def _handle_webrtc_close(self, data: dict):
        """Handle WebRTC close request

        Mode is NOT changed here — only explicit set_mode commands change mode.
        WebRTC session lifecycle is independent of mode state.
        """
        session_id = data.get('session_id')
        self.logger.info(f"[WEBRTC] Close request: session={session_id} - mode unchanged")

        if self._webrtc_service:
            await self._webrtc_service.close_connection(session_id)

    async def _handle_command(self, data: dict):
        """Handle command from app

        Message formats supported:
        - {"type": "command", "command": "treat", "data": {...}}
        - {"type": "command", "command": "set_mode", "mode": "manual"}  (mode at top level)
        """
        command = data.get('command')
        params = data.get('data', {})  # Params may be in 'data' field

        # Also check for top-level params (app sends mode at top level)
        if not params:
            params = {k: v for k, v in data.items() if k not in ('type', 'command')}

        self.logger.info(f"☁️ Command: {command}, params: {params}")

        if command is None:
            self.logger.warning(f"☁️ Missing command in message: {data}")
            await self._send({
                'type': 'command_ack',
                'command': 'unknown',
                'success': False,
                'error': 'Missing command field'
            })
            return

        # Handle set_mode command directly here (app sends this on connect)
        if command == 'set_mode':
            mode_name = params.get('mode', '').lower()
            if mode_name:
                try:
                    new_mode = SystemMode(mode_name)
                    self.state.set_mode(new_mode, f"App command: {mode_name}")
                    self.logger.info(f"📱 Mode changed to {mode_name} via app command")
                    await self._send({
                        'type': 'command_ack',
                        'command': command,
                        'success': True
                    })
                    return
                except ValueError:
                    self.logger.warning(f"☁️ Invalid mode: {mode_name}")
                    await self._send({
                        'type': 'command_ack',
                        'command': command,
                        'success': False,
                        'error': f'Invalid mode: {mode_name}'
                    })
                    return

        # Handle local_mode command - switch to AP mode for direct control
        if command == 'local_mode':
            await self._handle_local_mode_switch()
            return

        # Handle cloud_mode command - switch back to client/cloud mode
        if command == 'cloud_mode':
            await self._handle_cloud_mode_switch()
            return

        # Publish command to event bus for other services to handle
        self.bus.publish(CloudEvent(
            subtype='command',
            data={'command': command, 'params': params},
            source='relay_client'
        ))

        # Acknowledge command received (processing happens async via bus)
        await self._send({
            'type': 'command_ack',
            'command': command,
            'success': True
        })

    async def _handle_ping(self, data: dict):
        """Handle ping from relay"""
        await self._send({'type': 'pong', 'device_id': self.config.device_id, 'timestamp': time.time()})

    async def _handle_local_mode_switch(self):
        """Switch to AP mode for direct local control.

        This sends the hotspot info BEFORE switching (since we'll lose relay connection).
        """
        try:
            from services.network.wifi_manager import WiFiManager

            wifi = WiFiManager()
            serial = wifi.get_device_serial()
            ssid = f"WIMZ-{serial}"
            password = "wimzsetup"

            self.logger.info(f"📡 Switching to Local Mode (AP: {ssid})")

            # Send response BEFORE switching (we'll lose relay connection)
            await self._send({
                'type': 'local_mode_starting',
                'ssid': ssid,
                'password': password,
                'ip': '192.168.4.1',
                'api': 'http://192.168.4.1:8000',
                'ws': 'ws://192.168.4.1:8000/ws/local'
            })

            # Give time for message to send
            await asyncio.sleep(1)

            # Now switch to AP mode (this will disconnect us from relay)
            success = wifi.start_hotspot(ssid, password)

            if success:
                self.logger.info(f"✅ Local Mode active - AP: {ssid}")
            else:
                self.logger.error("❌ Failed to start AP mode")

        except Exception as e:
            self.logger.error(f"Local mode switch error: {e}")
            await self._send({
                'type': 'command_ack',
                'command': 'local_mode',
                'success': False,
                'error': str(e)
            })

    async def _handle_cloud_mode_switch(self):
        """Switch back to client/cloud mode.

        This stops the hotspot and reconnects to saved WiFi.
        """
        try:
            from services.network.wifi_manager import WiFiManager

            wifi = WiFiManager()

            self.logger.info("☁️ Switching to Cloud Mode")

            # Stop hotspot
            wifi.stop_hotspot()

            # Try to reconnect to known networks
            connected = wifi.try_connect_known(timeout=30)

            if connected:
                status = wifi.get_connection_status()
                self.logger.info(f"✅ Cloud Mode active - connected to {status.get('ssid')}")
                # Note: The relay client will auto-reconnect to relay server
            else:
                self.logger.warning("⚠️ Hotspot stopped but could not reconnect to WiFi")

        except Exception as e:
            self.logger.error(f"Cloud mode switch error: {e}")

    async def _handle_profiles(self, data: dict):
        """Handle dog profiles from cloud

        Expected format:
        {
            "type": "profiles",
            "profiles": [
                {"name": "Bezik", "aruco_id": 832, "color": "black"},
                {"name": "Elsa", "aruco_id": 1, "color": "yellow"}
            ]
        }
        """
        profiles = data.get('profiles', [])
        self.logger.info(f"📥 Received {len(profiles)} dog profiles from cloud")

        # Get profile manager lazily to avoid circular import
        if self._profile_manager is None:
            try:
                from core.dog_profile_manager import get_dog_profile_manager
                self._profile_manager = get_dog_profile_manager()
            except ImportError as e:
                self.logger.warning(f"Could not import profile manager: {e}")
                return

        if self._profile_manager:
            self._profile_manager.update_profiles_from_cloud(profiles)
            self.logger.info(f"✅ Updated dog profiles from cloud")

    async def request_profiles(self):
        """Request dog profiles from cloud relay"""
        await self._send({
            'type': 'get_profiles',
            'device_id': self.config.device_id
        })
        self.logger.info("📤 Requested dog profiles from cloud")

    async def _handle_audio_message(self, data: dict):
        """Handle audio_message - play audio from app (push-to-talk)

        Expected format:
        {"type": "audio_message", "data": "<base64>", "format": "aac"}
        """
        try:
            from services.media.push_to_talk import get_push_to_talk_service
            ptt_service = get_push_to_talk_service()

            audio_data = data.get('data')
            audio_format = data.get('format', 'aac')

            if not audio_data:
                self.logger.warning("☁️ Audio message missing data")
                return

            self.logger.info(f"🔊 Playing PTT audio from cloud ({audio_format})")
            result = ptt_service.play_audio_base64(audio_data, audio_format)

            # Send response
            await self._send({
                'event': 'audio_played',
                'device_id': self.config.device_id,
                'success': result.get('success'),
                'error': result.get('error')
            })

        except Exception as e:
            self.logger.error(f"☁️ Audio message error: {e}")

    async def _handle_audio_request(self, data: dict):
        """Handle audio_request - record from mic and send back (listen feature)

        Expected format:
        {"type": "audio_request", "duration": 5, "format": "aac"}
        """
        try:
            import asyncio
            from services.media.push_to_talk import get_push_to_talk_service
            ptt_service = get_push_to_talk_service()

            duration = data.get('duration', 5)
            audio_format = data.get('format', 'aac')

            self.logger.info(f"🎤 AUDIO_REQUEST: Starting mic capture ({duration}s, format={audio_format})")

            # Run blocking record_audio in a thread to avoid freezing the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: ptt_service.record_audio(duration=duration, format=audio_format)
            )

            if result.get('success'):
                self.logger.info(f"🎤 Recording complete, sending {result.get('size_bytes')} bytes to app")
                await self._send({
                    'event': 'audio_message',
                    'device_id': self.config.device_id,
                    'data': result.get('data'),
                    'format': audio_format,
                    'duration_ms': result.get('duration_ms'),
                    'size_bytes': result.get('size_bytes')
                })
                self.logger.info(f"📤 Sent PTT audio to cloud ({result.get('size_bytes')} bytes)")
            else:
                self.logger.error(f"🎤 Recording failed: {result.get('error')}")
                await self._send({
                    'event': 'audio_error',
                    'device_id': self.config.device_id,
                    'error': result.get('error')
                })

        except Exception as e:
            self.logger.error(f"☁️ Audio request error: {e}", exc_info=True)
            await self._send({
                'event': 'audio_error',
                'device_id': self.config.device_id,
                'error': str(e)
            })

    async def _handle_user_connected(self, data: dict):
        """Handle user_connected event from relay (Build 32)

        Sent when an app user connects to the robot.
        Expected format: {"type": "user_connected", "user_id": "user_123"}
        """
        user_id = data.get('user_id')
        self.logger.info(f"📱 User connected: {user_id}")
        print(f"[RelayClient] 📱 User connected: {user_id}", flush=True)

        self._connected_user_id = user_id
        self._app_connected = True

        # Publish event for other services
        self.bus.publish(CloudEvent(
            subtype='user_connected',
            data={'user_id': user_id},
            source='relay_client'
        ))

        # Request fresh dog profiles when user connects
        await self.request_profiles()

    async def _handle_user_disconnected(self, data: dict):
        """Handle user_disconnected event from relay (Build 32)

        Sent after 5-min grace period when app user disconnects.
        Expected format: {"type": "user_disconnected", "user_id": "user_123"}

        Actions:
        - Clear user-specific state
        - Stop any pending operations that require user interaction
        - Keep Silent Guardian running (autonomous mode)
        """
        user_id = data.get('user_id')
        self.logger.info(f"📱 User disconnected: {user_id}")
        print(f"[RelayClient] 📱 User disconnected: {user_id}", flush=True)

        self._connected_user_id = None
        self._app_connected = False

        # Stop any active missions (require user interaction)
        try:
            from orchestrators.mission_engine import get_mission_engine
            mission_engine = get_mission_engine()
            if mission_engine.active_session:
                self.logger.info("Stopping active mission due to user disconnect")
                mission_engine.stop_mission()
        except Exception as e:
            self.logger.warning(f"Could not stop mission on disconnect: {e}")

        # Stop any active programs
        try:
            from orchestrators.program_engine import get_program_engine
            program_engine = get_program_engine()
            if program_engine.active_session:
                self.logger.info("Stopping active program due to user disconnect")
                program_engine.stop_program()
        except Exception as e:
            self.logger.warning(f"Could not stop program on disconnect: {e}")

        # Close WebRTC sessions if active
        if self._webrtc_service:
            try:
                await self._webrtc_service.cleanup()
                self.logger.info("Closed WebRTC connections on user disconnect")
            except Exception as e:
                self.logger.warning(f"Could not close WebRTC on disconnect: {e}")

        # Publish event for other services
        self.bus.publish(CloudEvent(
            subtype='user_disconnected',
            data={'user_id': user_id},
            source='relay_client'
        ))

        # Note: Silent Guardian continues running - it's autonomous

    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff (1s, 2s, 4s, ... max 30s)"""
        backoff = 1.0  # Start at 1 second

        while self._running:
            if not self._connected:
                self.logger.info(f"Attempting reconnection in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

                if await self._connect():
                    backoff = 1.0  # Reset to 1 second on success
                    # Flush queued messages
                    await self._flush_queue()
                    # Start heartbeat in background
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    # Start receive loop
                    await self._receive_loop()
                    # Cancel heartbeat when receive loop ends
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()
                else:
                    backoff = min(backoff * 2, 30.0)  # Double each time, cap at 30s
            else:
                await asyncio.sleep(1.0)

    async def _flush_queue(self):
        """Send queued messages after reconnection"""
        if not self._message_queue:
            return

        queue_size = len(self._message_queue)
        self.logger.info(f"Flushing {queue_size} queued messages")

        # Process queue
        messages_to_send = self._message_queue.copy()
        self._message_queue.clear()

        for msg in messages_to_send:
            if self._connected:
                await self._send(msg)
            else:
                break  # Stop if we lost connection again

    async def _run(self):
        """Main async run loop"""
        self._running = True

        if await self._connect():
            # Start heartbeat in background
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            await self._receive_loop()
            # Cancel heartbeat when receive loop ends
            if self._heartbeat_task:
                self._heartbeat_task.cancel()

        # Start reconnection loop if still running
        if self._running:
            await self._reconnect_loop()

    def _run_thread(self):
        """Thread entry point for running async loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()

    def start(self) -> bool:
        """Start relay client in background thread"""
        if not self.config.enabled:
            self.logger.info("Relay client disabled - not starting")
            return False

        if self._thread and self._thread.is_alive():
            self.logger.warning("Relay client already running")
            return True

        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name="RelayClient"
        )
        self._thread.start()
        self.logger.info("Relay client started")
        return True

    def stop(self):
        """Stop relay client"""
        self.logger.info("Stopping relay client...")
        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Schedule disconnect in the event loop and wait for it (with goodbye message)
        if self._loop and self._connected:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._disconnect(send_goodbye=True), self._loop
                )
                future.result(timeout=3.0)  # Wait for disconnect to complete
            except Exception as e:
                self.logger.debug(f"Disconnect wait error: {e}")

        # Close session and wait for it
        if self._session and not self._session.closed:
            if self._loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
                    future.result(timeout=2.0)  # Wait for session close
                except Exception as e:
                    self.logger.debug(f"Session close error: {e}")

        # Wait for thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self.logger.info("Relay client stopped")

    def send_event(self, event_type: str, data: dict):
        """Send event to relay (thread-safe)

        Format: {"event": "<type>", "device_id": "<id>", ...data fields...}
        The relay requires the "event" field to forward messages to the app.
        Device_id is included for routing/filtering on the relay/app side.
        """
        if not self._connected or not self._loop:
            return

        # Flat message format with "event" field as required by relay API
        # Always include device_id for proper routing
        message = {
            'event': event_type,
            'device_id': self.config.device_id,
            **data
        }

        asyncio.run_coroutine_threadsafe(self._send(message), self._loop)

    @property
    def connected(self) -> bool:
        """Check if connected to relay"""
        return self._connected

    def get_status(self) -> dict:
        """Get relay client status"""
        return {
            'enabled': self.config.enabled,
            'connected': self._connected,
            'relay_url': self.config.relay_url,
            'device_id': self.config.device_id[:8] + '...' if self.config.device_id else None
        }


# Singleton instance
_relay_client: Optional[RelayClient] = None
_relay_lock = threading.Lock()


def get_relay_client(config: Optional[RelayConfig] = None) -> RelayClient:
    """Get or create relay client singleton"""
    global _relay_client
    with _relay_lock:
        if _relay_client is None:
            if config is None:
                config = RelayConfig()
            _relay_client = RelayClient(config)
    return _relay_client


def configure_relay_from_yaml(config_dict: dict) -> RelayConfig:
    """Create RelayConfig from YAML config dictionary"""
    cloud_config = config_dict.get('cloud', {})

    # Get credentials from environment variables if not in config
    device_id = cloud_config.get('device_id') or os.environ.get('DEVICE_ID', '')
    device_secret = cloud_config.get('device_secret') or os.environ.get('DEVICE_SECRET', '')

    return RelayConfig(
        enabled=cloud_config.get('enabled', False),
        relay_url=cloud_config.get('relay_url', 'wss://api.wimzai.com/ws/device'),
        device_id=device_id,
        device_secret=device_secret,
        reconnect_delay=cloud_config.get('reconnect_delay', 5.0),
        heartbeat_interval=cloud_config.get('heartbeat_interval', 30.0),
    )
