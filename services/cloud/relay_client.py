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
        self._telemetry_task: Optional[asyncio.Task] = None

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
            self.logger.info(
                f"[RTC-TIMING] step2 relay signaling websocket OPEN "
                f"({self.config.relay_url}) | "
                f"t={time.time():.3f} mono={time.monotonic():.3f}"
            )

            # Send robot_connected announcement
            await self._send({
                'type': 'robot_connected',
                'device_id': self.config.device_id,
                'version': ROBOT_VERSION
            })
            self.logger.info(
                f"[RTC-TIMING] step3 sent robot_connected — registered as "
                f"available robot (device_id={self.config.device_id}) | "
                f"t={time.time():.3f} mono={time.monotonic():.3f}"
            )

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
                    self.logger.debug("Heartbeat sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Heartbeat error: {e}")

    async def _telemetry_loop(self):
        """Send telemetry (battery, temperature, mode) every 5 seconds while connected"""
        from datetime import datetime

        # Mode mapping (same as api/ws.py)
        mode_map = {
            "idle": "idle",
            "silent_guardian": "silent_guardian",
            "coach": "coach",
            "mission": "mission",
            "manual": "manual",
            "photography": "manual",
            "emergency": "manual"
        }

        while self._running and self._connected:
            try:
                await asyncio.sleep(5.0)  # Send every 5 seconds
                if self._connected and self._app_connected:
                    # Get state
                    state_dict = self.state.get_full_state()
                    hardware = state_dict.get("hardware", {})

                    # Calculate battery percentage
                    battery_voltage = hardware.get("battery_voltage", 0)
                    battery_pct = ((battery_voltage - 12.0) / 4.8 * 100) if battery_voltage else 0
                    battery_pct = min(100, max(0, battery_pct))

                    # Get mode
                    internal_mode = state_dict.get("mode", "idle")
                    contract_mode = mode_map.get(internal_mode, "idle")

                    # Get WebRTC connection type (LAN/WAN) for app badge
                    connection_type = None
                    if self._webrtc_service:
                        connection_type = self._webrtc_service.connection_type

                    # Current system volume (single source of truth, 0-100).
                    # Lets the app show the real volume without polling
                    # GET /audio/volume — survives reboots and reflects
                    # changes made from the Xbox controller.
                    try:
                        from services.media.volume_manager import get_volume_manager
                        current_volume = get_volume_manager().get_volume()
                    except Exception:
                        current_volume = None

                    # Send telemetry event
                    await self._send({
                        'event': 'status',
                        'device_id': self.config.device_id,
                        'data': {
                            'battery': round(battery_pct, 1),
                            'temperature': hardware.get("cpu_temp", 0),
                            'mode': contract_mode,
                            'is_charging': hardware.get("is_charging", False),
                            'treats_remaining': self._get_treats_remaining(),
                            'connection_type': connection_type,  # "LAN", "WAN", or null
                            'volume': current_volume,  # 0-100, or null if unavailable
                        },
                        'timestamp': datetime.utcnow().isoformat() + "Z"
                    })
                    self.logger.debug(f"Telemetry sent: {battery_pct:.0f}%, {hardware.get('cpu_temp', 0)}C")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"Telemetry error: {e}")

    async def _handle_message(self, data: dict):
        """Route incoming message to appropriate handler"""
        msg_type = data.get('type')
        msg_device_id = data.get('device_id')

        self.logger.debug(f"Relay message received: type={msg_type}, device_id={msg_device_id}")

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
        # [RTC-TIMING] step4 — robot is the offerer; relay's inbound msg is a
        # webrtc_request (not an SDP offer). Robot will build+send the offer next.
        _ice_in = data.get('ice_servers')
        self.logger.info(
            f"[RTC-TIMING] step4 webrtc_request received from relay "
            f"session={session_id} | ice_servers={'provided' if _ice_in else 'NONE'} | "
            f"t={time.time():.3f} mono={time.monotonic():.3f}"
        )
        if _ice_in:
            self.logger.info(f"[RTC-TIMING] step4 relay-supplied ICE servers: {_ice_in}")

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
            self.logger.info(
                f"[RTC-TIMING] step6 SDP offer sent to relay session={session_id} | "
                f"t={time.time():.3f} mono={time.monotonic():.3f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to create WebRTC offer: {e}", exc_info=True)
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
        self.logger.info(
            f"[RTC-TIMING] step6b SDP answer received from app session={session_id} | "
            f"t={time.time():.3f} mono={time.monotonic():.3f}"
        )

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

    # Command staleness thresholds (seconds)
    COMMAND_MAX_AGE_SEC = 30  # General commands
    SAFETY_COMMAND_MAX_AGE_SEC = 5  # Safety-critical commands (physical actions)
    SAFETY_COMMANDS = {'dispense_treat', 'motor_command', 'drive', 'emergency_stop'}

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

        # --- Staleness check: reject commands with old timestamps ---
        cmd_timestamp = data.get('timestamp')
        if cmd_timestamp is not None:
            now_ms = int(time.time() * 1000)
            age_sec = (now_ms - cmd_timestamp) / 1000.0

            # Pick threshold based on command type
            max_age = self.SAFETY_COMMAND_MAX_AGE_SEC if command in self.SAFETY_COMMANDS else self.COMMAND_MAX_AGE_SEC

            if age_sec > max_age:
                self.logger.warning(
                    f"STALE command rejected: {command}, age={age_sec:.1f}s (max={max_age}s), "
                    f"timestamp={cmd_timestamp}"
                )
                await self._send({
                    'type': 'command_ack',
                    'command': command,
                    'success': False,
                    'error': f'Command too old ({age_sec:.1f}s > {max_age}s limit)'
                })
                return

            if age_sec < -5:
                # Clock skew: command timestamp is >5s in the future
                self.logger.warning(
                    f"FUTURE command rejected: {command}, age={age_sec:.1f}s, "
                    f"timestamp={cmd_timestamp} (clock skew?)"
                )
                await self._send({
                    'type': 'command_ack',
                    'command': command,
                    'success': False,
                    'error': f'Command timestamp in future ({age_sec:.1f}s skew)'
                })
                return

        # --- Session check: warn if no user session is active ---
        if not self._app_connected and command not in ('set_mode', 'local_mode', 'cloud_mode'):
            self.logger.warning(
                f"Command received with no active user session: {command} "
                f"(user={self._connected_user_id})"
            )

        self.logger.debug(f"Command: {command}, params: {params}")

        if command is None:
            self.logger.warning(f"Missing command in message: {data}")
            await self._send({
                'type': 'command_ack',
                'command': 'unknown',
                'success': False,
                'error': 'Missing command field'
            })
            return

        # HOTFIX: Reset manual timeout on ANY app command while in MANUAL mode
        # This prevents 5-minute timeout when app is actively controlling
        if self.state.get_mode() == SystemMode.MANUAL:
            try:
                from orchestrators.mode_fsm import get_mode_fsm
                fsm = get_mode_fsm()
                if fsm:
                    fsm.last_manual_input_time = time.time()
            except Exception:
                pass

        # Handle set_mode command directly here (app sends this on connect)
        # API Contract v1.3: Accept source and timestamp fields for diagnostics
        if command == 'set_mode':
            mode_name = params.get('mode', '').lower()
            source = params.get('source', 'unknown')  # v1.3: dropdown, drive_enter, etc.
            app_timestamp = params.get('timestamp', '')  # v1.3: ISO8601 from app

            if mode_name:
                try:
                    from datetime import datetime
                    from orchestrators.mode_fsm import get_mode_fsm
                    new_mode = SystemMode(mode_name)
                    prev_mode = self.state.get_mode()  # HOTFIX: Capture BEFORE changing
                    server_ts = datetime.utcnow().isoformat() + 'Z'

                    # API Contract v1.3: Log with full context for debugging mode issues
                    self.logger.info(
                        f"MODE_CHANGE: {prev_mode.value} -> {mode_name} | "
                        f"source={source} | app_ts={app_timestamp} | server_ts={server_ts}"
                    )

                    # Use mode_fsm.set_mode_override() to protect from FSM auto-transitions.
                    # Without override, the FSM loop can revert SG/Coach back to IDLE
                    # within seconds (e.g. coach_timeout with no dog detection).
                    fsm = get_mode_fsm()
                    if fsm:
                        if mode_name == 'manual':
                            fsm.pre_manual_mode = prev_mode
                        success = fsm.set_mode_override(new_mode)
                        self.logger.debug(f"Mode override set via FSM: {mode_name} (source={source})")
                    else:
                        success = self.state.set_mode(new_mode, f"App command: {mode_name} (source={source})")

                    await self._send({
                        'type': 'command_ack',
                        'command': command,
                        'success': True
                    })
                    return
                except ValueError:
                    self.logger.warning(f"Invalid mode: {mode_name}")
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

        # Handle motor_command - relay-path fallback for analog drive.
        # App sends pre-mixed {left, right} floats in [-1.0, 1.0]; mixing
        # (Y +/- X -> L, R) happens app-side. Robot applies per-channel deadzone
        # (default 0.10, configurable via controller.app_input_deadzone in
        # robot_profiles/<unit>.yaml) and linearly scales to +/-100 PWM-%.
        # Motor bus then applies the existing +/-70 safety clamp.
        # Already covered by SAFETY_COMMANDS staleness gate (5s).
        if command == 'motor_command':
            try:
                from core.motor_command_bus import (
                    get_motor_bus, map_stick_to_motor_command, CommandSource,
                )
                try:
                    from config.config_loader import get_config
                    deadzone = float(
                        get_config().raw.get('controller', {}).get('app_input_deadzone', 0.10)
                    )
                except Exception:
                    deadzone = 0.10

                cmd = map_stick_to_motor_command(
                    params.get('left', 0),
                    params.get('right', 0),
                    CommandSource.RELAY,
                    deadzone=deadzone,
                )
                bus = get_motor_bus()
                ok = bool(bus and bus.running and bus.send_command(cmd))
                ack = {'type': 'command_ack', 'command': command, 'success': ok}
                if not ok:
                    ack['error'] = 'motor bus unavailable'
                await self._send(ack)
            except Exception as e:
                self.logger.error(f"motor_command relay handler error: {e}")
                await self._send({
                    'type': 'command_ack',
                    'command': command,
                    'success': False,
                    'error': str(e),
                })
            return

        # Handle audio_volume - set system volume over the cloud relay.
        # params: {"volume": 0-100}  ("level" accepted as an alias)
        if command == 'audio_volume':
            raw = params.get('volume', params.get('level'))
            ok = False
            try:
                vol = int(raw)
                from services.media.volume_manager import get_volume_manager
                ok = get_volume_manager().set_volume(vol)
            except (TypeError, ValueError):
                self.logger.warning(f"audio_volume: invalid value {raw!r}")
            except Exception as e:
                self.logger.error(f"audio_volume error: {e}")
            await self._send({
                'type': 'command_ack', 'command': command, 'success': ok,
            })
            return

        # Handle set_video_quality - adaptive bitrate manual override.
        # params: {"mode": "auto"|"low"|"medium"|"high"}
        if command == 'set_video_quality':
            mode = params.get('mode', 'auto')
            ok = False
            if self._webrtc_service:
                try:
                    ok = self._webrtc_service.set_video_quality(mode)
                except Exception as e:
                    self.logger.error(f"set_video_quality error: {e}")
            await self._send({
                'type': 'command_ack', 'command': command, 'success': ok,
            })
            return

        # Handle sg_config - live-tune Silent Guardian thresholds from the app.
        # Mirrors POST /sg/config but reachable via the relay socket so the
        # "punishment level" slider can push updates without HTTP.
        # params: {"bark_threshold", "confidence_minimum", "bark_count_threshold",
        #          "fast_escalation_bpm"} — all optional; only sent fields apply.
        if command == 'sg_config':
            try:
                from modes.silent_guardian import get_silent_guardian_mode
                sg = get_silent_guardian_mode()
                bark = sg.config.setdefault('bark_detection', {})
                updated = {}
                if 'bark_threshold' in params:
                    bark['loudness_threshold_db'] = float(params['bark_threshold'])
                    updated['loudness_threshold_db'] = bark['loudness_threshold_db']
                if 'confidence_minimum' in params:
                    bark['confidence_minimum'] = float(params['confidence_minimum'])
                    updated['confidence_minimum'] = bark['confidence_minimum']
                if 'bark_count_threshold' in params:
                    bark['threshold'] = int(params['bark_count_threshold'])
                    updated['bark_count_threshold'] = bark['threshold']
                if 'fast_escalation_bpm' in params:
                    bark['fast_escalation_bpm'] = int(params['fast_escalation_bpm'])
                    updated['fast_escalation_bpm'] = bark['fast_escalation_bpm']
                self.logger.info(f"sg_config applied: {updated}")
                await self._send({
                    'type': 'command_ack', 'command': command,
                    'success': True, 'updated': updated,
                })
            except Exception as e:
                self.logger.error(f"sg_config error: {e}")
                await self._send({
                    'type': 'command_ack', 'command': command,
                    'success': False, 'error': str(e),
                })
            return

        # Handle mood_led - blue LED toggle via relay.
        # params: {"action": "on"|"off"|"toggle"}
        # Routes through LedService's LedController — the single legitimate
        # owner of GPIO 25. The old api.server.blue_led_direct_control() path
        # made an independent lgpio.gpio_claim_output() on the same pin, which
        # loses a startup race with LedController and then fails 'GPIO busy'
        # for the rest of the process lifetime (silently returning False).
        if command == 'set_night_mode_override':
            override = (params.get('override') or '').lower()
            ok = False
            try:
                from modes.night_mode_controller import get_night_mode_controller
                nm = get_night_mode_controller()
                ok = nm.set_override(override)
            except Exception as e:
                self.logger.error(f"set_night_mode_override error: {e}")
            await self._send({
                'type': 'command_ack', 'command': command, 'success': ok,
                'error': None if ok else f"invalid override '{override}' (use auto/force_day/force_night)",
            })
            return

        if command == 'mood_led':
            action = params.get('action', 'toggle')
            ok = False
            try:
                from services.media.led import get_led_service
                led = get_led_service().led
                if led is None:
                    raise RuntimeError("LED service unavailable")
                # Don't pre-check blue_chip: blue_on/off lazily (re)claim GPIO25,
                # self-healing the boot-time 'GPIO busy' race. ok reflects reality.
                if action == 'on':
                    ok = bool(led.blue_on())
                elif action == 'off':
                    ok = bool(led.blue_off())
                else:
                    ok = bool(led.blue_off() if getattr(led, 'blue_is_on', False)
                              else led.blue_on())
            except Exception as e:
                self.logger.error(f"mood_led error: {e}")
            await self._send({
                'type': 'command_ack', 'command': command, 'success': ok,
            })
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
            from services.network.wifi_manager import get_wifi_manager

            wifi = get_wifi_manager()
            serial = wifi.get_device_serial()
            ssid = f"WIMZ-{serial}"
            password = "wimzsetup"

            self.logger.info(f"Switching to Local Mode (AP: {ssid})")

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
                self.logger.info(f"Local Mode active - AP: {ssid}")
            else:
                self.logger.error("Failed to start AP mode")

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
            from services.network.wifi_manager import get_wifi_manager

            wifi = get_wifi_manager()

            self.logger.info("Switching to Cloud Mode")

            # Stop hotspot
            wifi.stop_hotspot()

            # Try to reconnect to known networks
            connected = wifi.try_connect_known(timeout=30)

            if connected:
                status = wifi.get_connection_status()
                self.logger.info(f"Cloud Mode active - connected to {status.get('ssid')}")
                # Note: The relay client will auto-reconnect to relay server
            else:
                self.logger.warning("Hotspot stopped but could not reconnect to WiFi")

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
        self.logger.debug(f"Received {len(profiles)} dog profiles from cloud")

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
            self.logger.debug("Updated dog profiles from cloud")

    async def request_profiles(self):
        """Request dog profiles from cloud relay"""
        await self._send({
            'type': 'get_profiles',
            'device_id': self.config.device_id
        })
        self.logger.debug("Requested dog profiles from cloud")

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
                self.logger.warning("[PTT] Audio message missing data field")
                return

            import base64
            raw_size = len(base64.b64decode(audio_data))
            self.logger.info(f"[PTT] Received from relay: {raw_size} bytes, format={audio_format}")

            result = ptt_service.play_audio_base64(audio_data, audio_format)

            # Send ack immediately (playback is async/queued)
            await self._send({
                'event': 'audio_played',
                'device_id': self.config.device_id,
                'success': result.get('success'),
                'error': result.get('error'),
                'queue_depth': result.get('queue_depth', 1)
            })
            self.logger.info(f"[PTT] ACK sent to relay (success={result.get('success')})")

        except Exception as e:
            self.logger.error(f"[PTT] Relay audio message error: {e}")
            # Always send ack even on error
            try:
                await self._send({
                    'event': 'audio_played',
                    'device_id': self.config.device_id,
                    'success': False,
                    'error': str(e)
                })
            except Exception:
                pass

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

            self.logger.debug(f"Audio request: starting mic capture ({duration}s, format={audio_format})")

            # Run blocking record_audio in a thread to avoid freezing the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: ptt_service.record_audio(duration=duration, format=audio_format)
            )

            if result.get('success'):
                self.logger.debug(f"Recording complete, sending {result.get('size_bytes')} bytes to app")
                await self._send({
                    'event': 'audio_message',
                    'device_id': self.config.device_id,
                    'data': result.get('data'),
                    'format': audio_format,
                    'duration_ms': result.get('duration_ms'),
                    'size_bytes': result.get('size_bytes')
                })
                self.logger.debug(f"Sent PTT audio to cloud ({result.get('size_bytes')} bytes)")
            else:
                self.logger.error(f"Recording failed: {result.get('error')}")
                await self._send({
                    'event': 'audio_error',
                    'device_id': self.config.device_id,
                    'error': result.get('error')
                })

        except Exception as e:
            self.logger.error(f"Audio request error: {e}", exc_info=True)
            await self._send({
                'event': 'audio_error',
                'device_id': self.config.device_id,
                'error': str(e)
            })

    async def _handle_user_connected(self, data: dict):
        """Handle user_connected event from relay

        Sent when an app user connects to the robot.
        Expected format: {"type": "user_connected", "user_id": "user_123"}
        """
        user_id = data.get('user_id')
        self.logger.info(f"User connected: {user_id}")

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

        # Send current mode + status after a brief delay so the app's
        # listeners are ready (race condition: mode_changed can arrive
        # before the app finishes setting up its WebSocket handlers)
        await asyncio.sleep(1.0)

        _mode_map = {
            "idle": "idle", "silent_guardian": "silent_guardian",
            "coach": "coach", "mission": "mission",
            "manual": "manual", "photography": "manual",
            "emergency": "manual"
        }
        current_mode = self.state.get_mode().value
        contract_mode = _mode_map.get(current_mode, current_mode)

        # Send mode_changed event
        await self._send({
            'event': 'mode_changed',
            'device_id': self.config.device_id,
            'mode': contract_mode,
            'previous_mode': contract_mode,
            'locked': False,
            'reason': 'user_connected sync',
            'timestamp': time.time()
        })

        # Also send battery telemetry with mode (second chance for app to pick up mode)
        try:
            from services.power.battery_monitor import get_battery_monitor
            batt = get_battery_monitor()
            await self._send({
                'event': 'battery',
                'device_id': self.config.device_id,
                'level': batt.percentage,
                'voltage': batt.voltage,
                'charging': batt.charging_detected,
                'mode': contract_mode,
                'temperature': self.state.hardware.temperature,
            })
        except Exception as e:
            self.logger.debug(f"Could not send battery sync on connect: {e}")

        self.logger.info(f"Sent mode sync to app on connect: {contract_mode}")

    async def _handle_user_disconnected(self, data: dict):
        """Handle user_disconnected event from relay

        Sent after 5-min grace period when app user disconnects.
        Expected format: {"type": "user_disconnected", "user_id": "user_123"}

        Actions:
        - Clear user-specific state
        - Stop any pending operations that require user interaction
        - Keep Silent Guardian running (autonomous mode)
        """
        user_id = data.get('user_id')
        self.logger.info(f"User disconnected: {user_id}")

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

    async def _is_serving_local_ap(self) -> bool:
        """True if the robot is currently hosting its local AP (no cloud route).

        is_ap_mode() runs a blocking pgrep and may grab the wifi manager's lock
        during an AP transition, so run it in the default executor rather than
        on this asyncio loop.
        """
        try:
            from services.network.wifi_manager import get_wifi_manager
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, get_wifi_manager().is_ap_mode)
        except Exception:
            return False

    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff (1s, 2s, 4s, ... max 30s)"""
        backoff = 1.0  # Start at 1 second

        while self._running:
            if not self._connected:
                # Don't dial the cloud relay while the robot is serving its
                # local AP — there's no internet route, so every attempt fails
                # after the backoff and just adds churn/log noise during a demo.
                # Wait out AP mode instead. (Checked off-loop so the blocking
                # pgrep in is_ap_mode() doesn't stall this event loop.)
                if await self._is_serving_local_ap():
                    await asyncio.sleep(15)
                    continue

                self.logger.info(f"Attempting reconnection in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

                if await self._connect():
                    backoff = 1.0  # Reset to 1 second on success
                    # Flush queued messages
                    await self._flush_queue()
                    # Start heartbeat and telemetry in background
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    self._telemetry_task = asyncio.create_task(self._telemetry_loop())
                    # Start receive loop
                    await self._receive_loop()
                    # Cancel background tasks when receive loop ends
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()
                    if self._telemetry_task:
                        self._telemetry_task.cancel()
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
            # Start heartbeat and telemetry in background
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())
            await self._receive_loop()
            # Cancel background tasks when receive loop ends
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._telemetry_task:
                self._telemetry_task.cancel()

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

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._telemetry_task:
            self._telemetry_task.cancel()

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

    def _get_treats_remaining(self) -> int:
        """Get actual treats remaining from dispenser service"""
        try:
            from services.reward.dispenser import get_dispenser_service
            return get_dispenser_service().treats_remaining
        except Exception:
            return 0

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
    """Create RelayConfig from YAML config dictionary.

    Per-device .env overrides take precedence over the shared yaml so a single
    unit can be pointed at a different relay without forking robot_config.yaml.
    Order: RELAY_URL env > cloud.relay_url yaml > hard-coded fallback.
    """
    cloud_config = config_dict.get('cloud', {})

    device_id = cloud_config.get('device_id') or os.environ.get('DEVICE_ID', '')
    device_secret = cloud_config.get('device_secret') or os.environ.get('DEVICE_SECRET', '')
    relay_url = (
        os.environ.get('RELAY_URL')
        or cloud_config.get('relay_url')
        or 'wss://api.wimzai.com/ws/device'
    )

    return RelayConfig(
        enabled=cloud_config.get('enabled', False),
        relay_url=relay_url,
        device_id=device_id,
        device_secret=device_secret,
        reconnect_delay=cloud_config.get('reconnect_delay', 5.0),
        heartbeat_interval=cloud_config.get('heartbeat_interval', 30.0),
    )
