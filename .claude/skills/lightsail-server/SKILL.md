---
name: lightsail-server
description: Manage the WIM-Z Amazon Lightsail backend server. Use when working on TURN/STUN server, WebRTC infrastructure, backend APIs, SSL certs, or server configuration.
---

# WIM-Z Lightsail Server Management

## Server Details
- **Provider:** Amazon Lightsail
- **Purpose:** TURN/STUN relay for WebRTC video streaming, backend API services
- **Access:** SSH via `ssh ubuntu@api.wimzai.com` or configured hostname

## Key Services
- **TURN/STUN server:** Required for WebRTC video streaming between robot and mobile app
- **Backend API:** User accounts, mission data sync, push notifications
- **SSL/TLS:** Certificates for secure WebSocket connections

## WebRTC Architecture
```
Mobile App (Flutter) ←→ TURN Server (Lightsail) ←→ Robot (Raspberry Pi)
                         ↕
                    STUN for NAT traversal
```

## Common Operations
```bash
# Check TURN server status
ssh ubuntu@api.wimzai.com 'sudo systemctl status coturn'

# View TURN server logs
ssh ubuntu@api.wimzai.com 'sudo journalctl -u coturn -f --no-pager -n 100'

# Restart TURN server
ssh ubuntu@api.wimzai.com 'sudo systemctl restart coturn'

# Check SSL certificate expiry
ssh ubuntu@api.wimzai.com 'sudo certbot certificates'

# Renew SSL certs
ssh ubuntu@api.wimzai.com 'sudo certbot renew'
```

## NEVER Do
- Expose database ports to public internet
- Store API keys in code (use environment variables)
- Restart TURN during active streaming sessions without warning
- Modify firewall rules without documenting changes

## Cost Awareness
- Lightsail is a fixed monthly cost — monitor bandwidth usage
- TURN relay is the biggest bandwidth consumer (video streams)
- LLM API calls (future Ask Wimz) are negligible vs TURN bandwidth
- Monitor with: `ssh ubuntu@api.wimzai.com 'vnstat'` for bandwidth tracking

## Ports (verify firewall allows these)
- 22: SSH
- 443: HTTPS / WSS
- 3478: TURN/STUN (TCP+UDP)
- 5349: TURN/STUN TLS
- 49152-65535: TURN relay range (UDP)
