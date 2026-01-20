# Claude Code Instructions: WIM-Z Robot WebRTC Integration

## Context

You are working on the WIM-Z robot codebase (Raspberry Pi 5). The robot needs WebRTC video streaming capability to allow remote users to view live video through the cloud relay.

**Read API_CONTRACT.md first** - it contains the complete specification for WebRTC signaling.

## Your Task

Implement WebRTC video streaming using aiortc that:
1. Connects to the cloud relay server via WebSocket
2. Receives `webrtc_request` messages with ICE server credentials
3. Creates an SDP offer with the camera video track
4. Handles SDP answers and ICE candidates from the app
5. Streams video through Cloudflare TURN when needed

## Files to Create/Modify

### 1. Create: `services/streaming/__init__.py`
```python
from .webrtc_service import WebRTCService
from .video_track import WIMZVideoTrack

__all__ = ['WebRTCService', 'WIMZVideoTrack']
```

### 2. Create: `services/streaming/video_track.py`
- Extend `aiortc.VideoStreamTrack`
- Get frames from the existing detector/camera service
- Include AI detection overlay on frames
- Target 720p @ 15fps

### 3. Create: `services/streaming/webrtc_service.py`
- Manage RTCPeerConnection instances
- Handle multiple concurrent sessions (max 2)
- Create offers with video track
- Process answers and ICE candidates
- Clean up connections properly

### 4. Create: `services/cloud/relay_client.py`
- WebSocket client that connects to relay server
- Authenticate with device_id and HMAC signature
- Handle incoming messages (webrtc_request, webrtc_answer, webrtc_ice, commands)
- Forward robot events to relay
- Auto-reconnect on disconnect

### 5. Modify: `configs/robot_config.yaml`
Add:
```yaml
cloud:
  enabled: true
  relay_url: "wss://api.wimzai.com/ws/device"
  device_id: "${DEVICE_ID}"
  device_secret: "${DEVICE_SECRET}"
  reconnect_delay: 5

webrtc:
  enabled: true
  video:
    width: 1280
    height: 720
    fps: 15
    bitrate_kbps: 1500
  max_connections: 2
```

### 6. Modify: `main_treatbot.py`
- Initialize cloud relay client on startup
- Initialize WebRTC service
- Wire up message handling between relay client and WebRTC service

## Dependencies to Install

```bash
pip install aiortc aiohttp
sudo apt-get install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
```

## Key Implementation Details

### Video Track Integration
The robot already has a detector service that processes camera frames. Your video track should:
```python
# Get frame from existing detector
frame = self.detector.get_current_frame()  # Returns numpy array with AI overlay
```

### WebRTC Offer Creation
When receiving `webrtc_request`:
```python
async def handle_webrtc_request(self, session_id: str, ice_servers: dict):
    # Create peer connection with provided ICE servers
    config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=server['urls'], 
                     username=server.get('username'),
                     credential=server.get('credential'))
        for server in ice_servers.get('iceServers', [])
    ])
    
    pc = RTCPeerConnection(configuration=config)
    
    # Add video track
    video_track = WIMZVideoTrack(self.detector)
    pc.addTrack(video_track)
    
    # Create and return offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Send offer back through relay
    await self.relay_client.send({
        "type": "webrtc_offer",
        "session_id": session_id,
        "sdp": {
            "type": "offer",
            "sdp": pc.localDescription.sdp
        }
    })
```

### ICE Candidate Handling
```python
# Set up ICE candidate callback
@pc.on("icecandidate")
async def on_ice_candidate(candidate):
    if candidate:
        await self.relay_client.send({
            "type": "webrtc_ice",
            "session_id": session_id,
            "candidate": {
                "candidate": candidate.candidate,
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex
            }
        })
```

## Testing

1. Run relay server locally: `cd ~/wimz-relay && python run.py`
2. Run robot with cloud enabled
3. Check WebSocket connection to relay
4. Use webrtc test client or the mobile app to request video
5. Verify video stream establishes

## Success Criteria

- [ ] Robot connects to relay server on startup
- [ ] Robot handles `webrtc_request` and creates offer
- [ ] Robot sends ICE candidates to relay
- [ ] Robot handles answer and ICE candidates from app
- [ ] Video streams successfully through WebRTC
- [ ] Connection cleans up properly on close
- [ ] Robot reconnects to relay if connection drops
