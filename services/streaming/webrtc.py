#!/usr/bin/env python3
"""
WebRTC Streaming Service for WIM-Z
Provides low-latency video streaming over WebRTC using aiortc
"""

import asyncio
import json
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay

from core.bus import get_bus, publish_system_event
from core.state import get_state
from services.streaming.video_track import WIMZVideoTrack

# Motor control imports - direct hardware access like Xbox controller
try:
    from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
    MOTOR_BUS_AVAILABLE = True
except ImportError:
    MOTOR_BUS_AVAILABLE = False


@dataclass
class WebRTCConfig:
    """WebRTC configuration"""
    stun_servers: List[str] = field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    turn_servers: List[Dict[str, str]] = field(default_factory=list)
    max_bitrate: int = 1_500_000  # 1.5 Mbps
    target_fps: int = 15
    video_codec: str = "VP8"
    enable_ai_overlay: bool = True
    # CRITICAL: Set to 1 to prevent SEGFAULT crashes from multiple simultaneous sessions
    # Multiple sessions cause race conditions in MediaRelay/VP8 encoding
    max_connections: int = 1


class WebRTCService:
    """
    WebRTC streaming service singleton

    Handles:
    - Peer connection lifecycle
    - Video track management
    - ICE candidate gathering and exchange
    - Connection state monitoring
    - Multiple client support (limited)
    """

    def __init__(self, config: Optional[WebRTCConfig] = None):
        self.config = config or WebRTCConfig()
        self.logger = logging.getLogger('WebRTCService')
        self.bus = get_bus()
        self.state = get_state()

        # Peer connections by session_id
        self.connections: Dict[str, RTCPeerConnection] = {}

        # Video track (shared via MediaRelay for multiple clients)
        self.video_track: Optional[WIMZVideoTrack] = None
        self.media_relay = MediaRelay()

        # ICE candidate callbacks (session_id -> callback)
        self.ice_callbacks: Dict[str, Callable] = {}

        # Connection state
        self._lock = threading.Lock()
        self._detector = None  # Lazy loaded

        self.logger.info(f"WebRTCService initialized (max {self.config.max_connections} connections)")

    async def _handle_data_channel_message(self, message: str, session_id: str):
        """
        Handle incoming data channel messages for motor control

        Message formats supported:
        - {"type": "motor", "left": -1.0 to 1.0, "right": -1.0 to 1.0}
        - {"command": "motor", "left": -1.0 to 1.0, "right": -1.0 to 1.0}

        Motor commands go DIRECTLY to hardware (like Xbox controller) for minimal latency.
        No HTTP API calls - direct motor bus access.
        """
        try:
            data = json.loads(message)
            # Support both "type" and "command" keys for flexibility
            msg_type = data.get('type') or data.get('command')

            if msg_type == 'motor':
                try:
                    left = float(data.get('left', 0))
                    right = float(data.get('right', 0))

                    # Clamp to -1.0 to 1.0 range
                    left = max(-1.0, min(1.0, left))
                    right = max(-1.0, min(1.0, right))

                    # Convert to percentage (-100 to 100) for motor bus
                    left_pct = int(left * 100)
                    right_pct = int(right * 100)

                    # Motor control - try direct bus first, fallback to API
                    motor_sent = False
                    if MOTOR_BUS_AVAILABLE:
                        try:
                            motor_bus = get_motor_bus()
                            if motor_bus and motor_bus.running:
                                cmd = create_motor_command(left_pct, right_pct, CommandSource.WEBRTC)
                                motor_bus.send_command(cmd)
                                motor_sent = True
                                # Only log periodically to avoid spam
                                if abs(left_pct) > 10 or abs(right_pct) > 10:
                                    self.logger.debug(f"Motor: L={left_pct}% R={right_pct}%")
                        except Exception as e:
                            self.logger.debug(f"Motor bus error, using API: {e}")

                    # Fallback to API if motor bus not available (Xbox controller owns GPIO)
                    if not motor_sent:
                        try:
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    'http://localhost:8000/motor/control',
                                    json={'left_speed': left_pct, 'right_speed': right_pct},
                                    timeout=aiohttp.ClientTimeout(total=0.5)
                                ) as resp:
                                    if resp.status == 200:
                                        if abs(left_pct) > 10 or abs(right_pct) > 10:
                                            self.logger.debug(f"Motor (API): L={left_pct}% R={right_pct}%")
                        except Exception as e:
                            self.logger.debug(f"Motor API fallback failed: {e}")

                except ValueError as e:
                    self.logger.warning(f"Invalid motor values: {e}")
                except Exception as e:
                    self.logger.error(f"Motor command error: {e}")

            elif msg_type == 'ping':
                # Heartbeat/latency check
                self.logger.debug(f"Data channel ping from {session_id}")

            elif msg_type == 'stop':
                # Emergency stop - try bus first, fallback to API
                stop_sent = False
                try:
                    if MOTOR_BUS_AVAILABLE:
                        motor_bus = get_motor_bus()
                        if motor_bus and motor_bus.running:
                            cmd = create_motor_command(0, 0, CommandSource.WEBRTC)
                            motor_bus.send_command(cmd)
                            stop_sent = True
                            self.logger.info("Emergency stop from data channel")
                except Exception as e:
                    self.logger.debug(f"Motor bus stop error: {e}")

                # API fallback for emergency stop
                if not stop_sent:
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                'http://localhost:8000/motor/stop',
                                json={'reason': 'webrtc_emergency'},
                                timeout=aiohttp.ClientTimeout(total=0.5)
                            ) as resp:
                                if resp.status == 200:
                                    self.logger.info("Emergency stop via API")
                    except Exception as e:
                        self.logger.error(f"Emergency stop API fallback failed: {e}")

            else:
                self.logger.debug(f"Unknown data channel message type: {msg_type}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON in data channel: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Data channel message error: {e}")

    def _setup_data_channel_handlers(self, channel, session_id: str):
        """Set up event handlers for a data channel"""
        @channel.on("open")
        def on_open():
            self.logger.info(f"Data channel opened for {session_id}")

        @channel.on("close")
        def on_close():
            self.logger.info(f"Data channel closed for {session_id}")

        @channel.on("message")
        async def on_message(message):
            self.logger.info(f"📥 Data channel message from {session_id}: {message[:200]}")
            print(f"[WebRTC] 📥 Data channel message: {message[:200]}", flush=True)
            await self._handle_data_channel_message(message, session_id)

    def _get_detector(self):
        """Lazy load detector service to avoid circular imports"""
        if self._detector is None:
            from services.perception.detector import get_detector_service
            self._detector = get_detector_service()
        return self._detector

    def _create_rtc_configuration(self) -> RTCConfiguration:
        """Create RTC configuration with ICE servers"""
        ice_servers = []

        # Add STUN servers
        for stun_url in self.config.stun_servers:
            ice_servers.append(RTCIceServer(urls=stun_url))

        # Add TURN servers
        for turn in self.config.turn_servers:
            ice_servers.append(RTCIceServer(
                urls=turn["urls"],
                username=turn.get("username"),
                credential=turn.get("credential")
            ))

        return RTCConfiguration(iceServers=ice_servers)

    async def create_peer_connection(
        self,
        session_id: str,
        on_ice_candidate: Optional[Callable] = None
    ) -> RTCPeerConnection:
        """Create a new peer connection for a client session"""
        # CRITICAL: Close existing sessions before creating new one to prevent SEGFAULT
        existing_sessions = list(self.connections.keys())
        if existing_sessions:
            self.logger.info(f"Closing {len(existing_sessions)} existing session(s) before creating new one")
            for old_session_id in existing_sessions:
                try:
                    await self._cleanup_connection(old_session_id)
                except Exception as e:
                    self.logger.warning(f"Error closing session {old_session_id}: {e}")
            await asyncio.sleep(0.2)

        with self._lock:
            if len(self.connections) >= self.config.max_connections:
                raise Exception(f"Max connections ({self.config.max_connections}) reached")

            if session_id in self.connections:
                raise Exception(f"Session {session_id} already exists")

        # Create peer connection with ICE configuration
        pc = RTCPeerConnection(configuration=self._create_rtc_configuration())

        # Store ICE callback
        if on_ice_candidate:
            self.ice_callbacks[session_id] = on_ice_candidate

        # Create or reuse video track
        if self.video_track is None:
            detector = self._get_detector()
            self.video_track = WIMZVideoTrack(
                detector=detector,
                fps=self.config.target_fps,
                enable_overlay=self.config.enable_ai_overlay
            )
            self.logger.info("Created new WIMZVideoTrack")

        # Add video track via media relay (allows multiple subscribers)
        relay_track = self.media_relay.subscribe(self.video_track)
        pc.addTrack(relay_track)

        # Create data channel for motor control (low-latency, unreliable)
        data_channel = pc.createDataChannel(
            "control",
            ordered=False,
            maxRetransmits=0
        )
        self._setup_data_channel_handlers(data_channel, session_id)
        self.logger.info(f"Created data channel for {session_id}")

        # Handle incoming data channels from app
        @pc.on("datachannel")
        def on_datachannel(channel):
            self.logger.info(f"Received data channel '{channel.label}' from {session_id}")
            self._setup_data_channel_handlers(channel, session_id)

        # Connection state change handler
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            self.logger.info(f"[WEBRTC] Connection {session_id}: {state}")

            publish_system_event('webrtc_connection_state', {
                'session_id': session_id,
                'state': state
            }, 'webrtc_service')

            if state in ["failed", "closed", "disconnected"]:
                self.logger.info(f"[WEBRTC] Session {session_id} {state} - cleaning up (mode unchanged)")
                await self._cleanup_connection(session_id)

        # ICE candidate handler
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            try:
                if candidate and session_id in self.ice_callbacks:
                    await self.ice_callbacks[session_id](candidate)
            except Exception as e:
                self.logger.warning(f"[WEBRTC] ICE candidate callback error for {session_id}: {e}")

        # ICE connection state handler
        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            try:
                state = pc.iceConnectionState
                self.logger.info(f"[WEBRTC] ICE state {session_id}: {state}")
                # Proactively clean up on ICE failure to prevent crashes
                if state in ["failed", "disconnected", "closed"]:
                    self.logger.warning(f"[WEBRTC] ICE {state} for {session_id}, scheduling cleanup...")
                    # Use asyncio.create_task to avoid blocking
                    asyncio.create_task(self._safe_cleanup_connection(session_id))
            except Exception as e:
                self.logger.error(f"[WEBRTC] ICE state change handler error: {e}")

        with self._lock:
            self.connections[session_id] = pc

        self.logger.info(f"[WEBRTC] Created peer connection: {session_id}")
        self.logger.info(f"[WEBRTC] Active connections: {list(self.connections.keys())}")
        return pc

    async def handle_offer(self, session_id: str, sdp: str) -> str:
        """
        Handle incoming SDP offer and return SDP answer

        Args:
            session_id: Unique session identifier
            sdp: SDP offer string from client

        Returns:
            SDP answer string
        """
        # Create peer connection if it doesn't exist
        with self._lock:
            if session_id not in self.connections:
                await self.create_peer_connection(session_id)

            pc = self.connections[session_id]

        # Set remote description (offer from client)
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self.logger.info(f"Created answer for session {session_id}")
        return pc.localDescription.sdp

    async def add_ice_candidate(self, session_id: str, candidate_dict: Dict[str, Any]):
        """Add ICE candidate to peer connection (with robust error handling)"""
        try:
            with self._lock:
                pc = self.connections.get(session_id)

            if pc is None:
                self.logger.warning(f"Session {session_id} not found for ICE candidate")
                return  # Don't raise - just log and continue

            # Check connection state before adding candidates
            if pc.connectionState in ["closed", "failed"]:
                self.logger.debug(f"Skipping ICE candidate for {session_id} - connection {pc.connectionState}")
                return

            # Parse the candidate string to create RTCIceCandidate
            # Format: "candidate:foundation component protocol priority ip port typ type ..."
            candidate_str = candidate_dict.get("candidate", "")
            sdp_mid = candidate_dict.get("sdpMid")
            sdp_mline_index = candidate_dict.get("sdpMLineIndex")

            if not candidate_str:
                # End-of-candidates signal
                try:
                    await pc.addIceCandidate(None)
                    self.logger.debug(f"End of ICE candidates for {session_id}")
                except Exception as e:
                    self.logger.warning(f"Error signaling end of ICE candidates for {session_id}: {e}")
                return

            # Parse candidate string using aioice
            from aioice import Candidate
            from aiortc.rtcicetransport import candidate_from_aioice

            # Remove "candidate:" prefix if present
            if candidate_str.startswith("candidate:"):
                candidate_str = candidate_str[10:]

            aioice_candidate = Candidate.from_sdp(candidate_str)
            rtc_candidate = candidate_from_aioice(aioice_candidate)
            rtc_candidate.sdpMid = sdp_mid
            rtc_candidate.sdpMLineIndex = sdp_mline_index

            await pc.addIceCandidate(rtc_candidate)
            self.logger.debug(f"Added ICE candidate for {session_id}")

        except Exception as e:
            # Catch all exceptions to prevent crashes - ICE failures are recoverable
            error_name = type(e).__name__
            if 'TransactionFailed' in error_name:
                self.logger.warning(f"ICE/TURN transaction failed for {session_id} (non-fatal): {e}")
            else:
                self.logger.error(f"Failed to add ICE candidate for {session_id}: {error_name}: {e}")

    async def create_offer(
        self,
        session_id: str,
        ice_servers: dict,
        on_ice_candidate: Optional[Callable] = None
    ) -> dict:
        """
        Create WebRTC offer for cloud relay signaling

        This is the robot-initiated flow:
        1. Relay sends webrtc_request with ICE servers
        2. Robot creates offer and sends back
        3. App responds with answer

        Args:
            session_id: Unique session identifier
            ice_servers: ICE server config from relay (includes TURN credentials)
            on_ice_candidate: Callback for ICE candidates

        Returns:
            SDP offer dict with type and sdp fields
        """
        # CRITICAL: Close existing sessions before creating new one to prevent SEGFAULT
        # Multiple simultaneous sessions cause race conditions in MediaRelay/VP8 encoding
        existing_sessions = list(self.connections.keys())
        if existing_sessions:
            self.logger.warning(f"[WEBRTC] Closing {len(existing_sessions)} existing session(s) before new request: {existing_sessions}")
            for old_session_id in existing_sessions:
                try:
                    await self._cleanup_connection(old_session_id)
                    self.logger.info(f"[WEBRTC] Closed existing session: {old_session_id}")
                except Exception as e:
                    self.logger.warning(f"[WEBRTC] Error closing session {old_session_id}: {e}")
            # Small delay to allow cleanup to complete
            await asyncio.sleep(0.2)
            self.logger.info(f"[WEBRTC] Active connections after cleanup: {list(self.connections.keys())}")

        with self._lock:
            if len(self.connections) >= self.config.max_connections:
                raise Exception(f"Max connections ({self.config.max_connections}) reached")

            if session_id in self.connections:
                raise Exception(f"Session {session_id} already exists")

        # Create configuration from provided ICE servers
        # ice_servers can be a list directly or a dict with 'iceServers' key
        rtc_ice_servers = []
        servers_list = ice_servers if isinstance(ice_servers, list) else ice_servers.get('iceServers', [])
        for server in servers_list:
            urls = server.get('urls', [])
            if isinstance(urls, str):
                urls = [urls]
            rtc_ice_servers.append(RTCIceServer(
                urls=urls,
                username=server.get('username'),
                credential=server.get('credential')
            ))

        # Fall back to default STUN if no servers provided
        if not rtc_ice_servers:
            for stun_url in self.config.stun_servers:
                rtc_ice_servers.append(RTCIceServer(urls=stun_url))

        config = RTCConfiguration(iceServers=rtc_ice_servers)
        pc = RTCPeerConnection(configuration=config)

        # Store ICE callback
        if on_ice_candidate:
            self.ice_callbacks[session_id] = on_ice_candidate

        # Create or reuse video track
        if self.video_track is None:
            detector = self._get_detector()
            self.video_track = WIMZVideoTrack(
                detector=detector,
                fps=self.config.target_fps,
                enable_overlay=self.config.enable_ai_overlay
            )
            self.logger.info("Created new WIMZVideoTrack for offer")

        # Add video track via media relay
        relay_track = self.media_relay.subscribe(self.video_track)
        pc.addTrack(relay_track)

        # Create data channel for motor control (low-latency, unreliable)
        data_channel = pc.createDataChannel(
            "control",
            ordered=False,
            maxRetransmits=0
        )
        self._setup_data_channel_handlers(data_channel, session_id)
        self.logger.info(f"Created data channel for offer {session_id}")

        # Handle incoming data channels from app
        @pc.on("datachannel")
        def on_datachannel(channel):
            self.logger.info(f"Received data channel '{channel.label}' from {session_id}")
            self._setup_data_channel_handlers(channel, session_id)

        # Connection state change handler
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            self.logger.info(f"[WEBRTC] Connection {session_id}: {state}")

            publish_system_event('webrtc_connection_state', {
                'session_id': session_id,
                'state': state
            }, 'webrtc_service')

            if state in ["failed", "closed", "disconnected"]:
                self.logger.info(f"[WEBRTC] Session {session_id} {state} - cleaning up (mode unchanged)")
                await self._cleanup_connection(session_id)

        # ICE candidate handler
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            try:
                if candidate and session_id in self.ice_callbacks:
                    await self.ice_callbacks[session_id](candidate)
            except Exception as e:
                self.logger.warning(f"[WEBRTC] ICE candidate callback error for {session_id}: {e}")

        # ICE connection state handler
        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            try:
                state = pc.iceConnectionState
                self.logger.info(f"[WEBRTC] ICE state {session_id}: {state}")
                if state in ["failed", "disconnected", "closed"]:
                    self.logger.warning(f"[WEBRTC] ICE {state} for {session_id}, scheduling cleanup...")
                    asyncio.create_task(self._safe_cleanup_connection(session_id))
            except Exception as e:
                self.logger.error(f"[WEBRTC] ICE state change handler error: {e}")

        with self._lock:
            self.connections[session_id] = pc

        self.logger.info(f"[WEBRTC] Active connections: {list(self.connections.keys())}")

        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        self.logger.info(f"[WEBRTC] Created offer for session {session_id}")

        return {
            "type": "offer",
            "sdp": pc.localDescription.sdp
        }

    async def handle_answer(self, session_id: str, sdp: dict):
        """
        Handle SDP answer from app via relay

        Args:
            session_id: Session identifier
            sdp: SDP answer dict with 'type' and 'sdp' fields
        """
        with self._lock:
            pc = self.connections.get(session_id)

        if pc is None:
            raise Exception(f"Session {session_id} not found")

        answer = RTCSessionDescription(
            sdp=sdp.get("sdp"),
            type=sdp.get("type", "answer")
        )
        await pc.setRemoteDescription(answer)
        self.logger.info(f"Set remote description (answer) for session {session_id}")

    async def _safe_cleanup_connection(self, session_id: str):
        """Safe cleanup wrapper that catches all exceptions to prevent crashes"""
        try:
            # Small delay to allow any in-flight operations to complete
            await asyncio.sleep(0.1)
            await self._cleanup_connection(session_id)
        except Exception as e:
            self.logger.error(f"Safe cleanup error for {session_id}: {e}")

    async def _cleanup_connection(self, session_id: str):
        """Clean up a peer connection"""
        with self._lock:
            pc = self.connections.pop(session_id, None)
            self.ice_callbacks.pop(session_id, None)

        if pc:
            try:
                # Check connection state before closing to avoid double-close
                if pc.connectionState not in ["closed"]:
                    await pc.close()
            except Exception as e:
                self.logger.warning(f"[WEBRTC] Error closing connection {session_id}: {e}")
            self.logger.info(f"[WEBRTC] Cleaned up connection: {session_id}")
            self.logger.info(f"[WEBRTC] Active connections: {list(self.connections.keys())}")

        # Stop video track if no connections remain
        with self._lock:
            if not self.connections and self.video_track:
                try:
                    self.video_track.stop()
                except Exception as e:
                    self.logger.warning(f"Error stopping video track: {e}")
                self.video_track = None
                self.logger.info("Stopped video track (no active connections)")

    async def close_connection(self, session_id: str):
        """Explicitly close a connection"""
        await self._cleanup_connection(session_id)

    def get_status(self) -> Dict[str, Any]:
        """Get WebRTC service status"""
        with self._lock:
            connections_info = {}
            for sid, pc in self.connections.items():
                connections_info[sid] = {
                    'connection_state': pc.connectionState,
                    'ice_state': pc.iceConnectionState,
                    'ice_gathering_state': pc.iceGatheringState
                }

            video_stats = None
            if self.video_track:
                video_stats = self.video_track.get_stats()

        return {
            'enabled': True,
            'active_connections': len(connections_info),
            'max_connections': self.config.max_connections,
            'connections': connections_info,
            'video_track_active': self.video_track is not None,
            'video_stats': video_stats,
            'config': {
                'target_fps': self.config.target_fps,
                'max_bitrate': self.config.max_bitrate,
                'ai_overlay': self.config.enable_ai_overlay,
                'stun_servers': self.config.stun_servers,
                'turn_configured': len(self.config.turn_servers) > 0
            }
        }

    async def cleanup(self):
        """Clean shutdown of all connections"""
        self.logger.info("Shutting down WebRTC service")

        with self._lock:
            session_ids = list(self.connections.keys())

        for session_id in session_ids:
            await self._cleanup_connection(session_id)

        self.logger.info("WebRTC service shutdown complete")


# Singleton instance
_webrtc_service: Optional[WebRTCService] = None
_webrtc_lock = threading.Lock()


def get_webrtc_service(config: Optional[WebRTCConfig] = None) -> WebRTCService:
    """Get the global WebRTC service singleton"""
    global _webrtc_service
    with _webrtc_lock:
        if _webrtc_service is None:
            _webrtc_service = WebRTCService(config)
    return _webrtc_service


def configure_webrtc_from_yaml(config_dict: Dict[str, Any]) -> WebRTCConfig:
    """Create WebRTCConfig from YAML config dictionary"""
    webrtc_config = config_dict.get('webrtc', {})

    return WebRTCConfig(
        stun_servers=webrtc_config.get('stun_servers', ["stun:stun.l.google.com:19302"]),
        turn_servers=webrtc_config.get('turn_servers', []),
        max_bitrate=webrtc_config.get('video', {}).get('bitrate', 1_500_000),
        target_fps=webrtc_config.get('video', {}).get('fps', 15),
        enable_ai_overlay=webrtc_config.get('video', {}).get('enable_ai_overlay', True),
        max_connections=webrtc_config.get('max_connections', 2)
    )
