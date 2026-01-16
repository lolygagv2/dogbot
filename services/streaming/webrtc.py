#!/usr/bin/env python3
"""
WebRTC Streaming Service for WIM-Z
Provides low-latency video streaming over WebRTC using aiortc
"""

import asyncio
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


@dataclass
class WebRTCConfig:
    """WebRTC configuration"""
    stun_servers: List[str] = field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    turn_servers: List[Dict[str, str]] = field(default_factory=list)
    max_bitrate: int = 1_500_000  # 1.5 Mbps
    target_fps: int = 15
    video_codec: str = "VP8"
    enable_ai_overlay: bool = True
    max_connections: int = 2


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

        # Connection state change handler
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            self.logger.info(f"Connection {session_id}: {state}")

            publish_system_event('webrtc_connection_state', {
                'session_id': session_id,
                'state': state
            }, 'webrtc_service')

            if state in ["failed", "closed", "disconnected"]:
                await self._cleanup_connection(session_id)

        # ICE candidate handler
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate and session_id in self.ice_callbacks:
                await self.ice_callbacks[session_id](candidate)

        # ICE connection state handler
        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            self.logger.debug(f"ICE state {session_id}: {pc.iceConnectionState}")

        with self._lock:
            self.connections[session_id] = pc

        self.logger.info(f"Created peer connection: {session_id}")
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
        """Add ICE candidate to peer connection"""
        with self._lock:
            pc = self.connections.get(session_id)

        if pc is None:
            raise Exception(f"Session {session_id} not found")

        candidate = RTCIceCandidate(
            sdpMid=candidate_dict.get("sdpMid"),
            sdpMLineIndex=candidate_dict.get("sdpMLineIndex"),
            candidate=candidate_dict.get("candidate")
        )
        await pc.addIceCandidate(candidate)
        self.logger.debug(f"Added ICE candidate for {session_id}")

    async def _cleanup_connection(self, session_id: str):
        """Clean up a peer connection"""
        with self._lock:
            pc = self.connections.pop(session_id, None)
            self.ice_callbacks.pop(session_id, None)

        if pc:
            try:
                await pc.close()
            except Exception as e:
                self.logger.warning(f"Error closing connection {session_id}: {e}")
            self.logger.info(f"Cleaned up connection: {session_id}")

        # Stop video track if no connections remain
        with self._lock:
            if not self.connections and self.video_track:
                self.video_track.stop()
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
