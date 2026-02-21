---
name: flutter-app
description: Build and deploy the WIM-Z Flutter mobile app. Use when working on the iOS/Android app, Codemagic CI/CD, TestFlight builds, app UI, WebRTC client, or mobile features.
---

# WIM-Z Flutter Mobile App

## Project Overview
- **Framework:** Flutter (Dart)
- **Current build:** Build 45+ on TestFlight
- **CI/CD:** Codemagic for iOS builds
- **Distribution:** TestFlight (iOS), planned Android

## Key App Features (Implemented)
- WebRTC live video streaming from robot
- Mission creation and management
- Multi-dog recognition (ArUco marker profiles)
- Photography mode with quality scoring
- Per-robot calibration interface
- Voice cloning / emulated owner voice settings
- Silent Guardian and Coach mode controls
- Battery telemetry dashboard

## Build & Deploy
```bash
# Local development
flutter pub get
flutter run                    # Run on connected device/simulator

# Build for iOS
flutter build ios --release

# Codemagic handles TestFlight deployment automatically on push
# Check build status at: https://codemagic.io/
```

## Project Structure Notes
- WebRTC client code handles TURN/STUN connection via Lightsail server
- Robot communication: WebSocket for commands, WebRTC for video
- State management: (Flutter app is in separate repo - check there for implementation)
- Per-robot settings stored locally and synced to robot on connection

## Testing
```bash
flutter test                   # Run unit tests
flutter test integration_test/ # Run integration tests
```

## Common Issues
- **WebRTC won't connect:** Check TURN server is running on Lightsail, verify ICE candidates
- **Build fails on Codemagic:** Check signing certificates haven't expired
- **Video lag:** TURN relay bandwidth — check Lightsail metrics
- **App crashes on launch:** Check minimum iOS version compatibility

## Rules for App Code Changes
1. Test on physical device before pushing (simulator doesn't test camera/WebRTC properly)
2. Increment build number for every TestFlight push
3. Don't hardcode robot IP addresses — use discovery or settings
4. Keep app responsive even when robot is offline (graceful disconnection handling)
5. All user-facing strings should be ready for future localization
