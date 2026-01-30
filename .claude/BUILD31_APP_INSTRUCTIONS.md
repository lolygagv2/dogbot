# Build 31 - App Claude Instructions

**Date:** January 30, 2026  
**Focus:** Integrate new Missions/Programs API and display AI detection in real-time

---

## Context

Robot Claude has rewritten the mission engine with a coach-style flow:
```
WAITING_FOR_DOG → GREETING → COMMAND → WATCHING → SUCCESS/FAILURE
```

The robot now broadcasts `mission_progress` WebSocket events in real-time. The app needs to display these to create a guided training experience (not the "random chance" situation from Build 28).

**Reference Document:** I've attached `BUILD31_APP_GUIDE.md` - this is the complete API contract from Robot Claude. Use it as the source of truth for all endpoints and data structures.

---

## Priority 1: Mission Progress Display (MUST HAVE)

### 1.1 Listen for `mission_progress` Events

When user starts a mission, the app should subscribe to WebSocket events and display live progress.

**Event Structure:**
```json
{
  "type": "event",
  "event": "mission_progress",
  "data": {
    "status": "watching",
    "trick": "sit",
    "stage": 2,
    "total_stages": 5,
    "dog_name": "Elsa",
    "rewards": 1,
    "progress": 2.5,
    "target_sec": 5.0
  }
}
```

### 1.2 UI States to Display

| Status | UI Display |
|--------|------------|
| `waiting_for_dog` | Pulsing animation: "Waiting for [dog_name]..." or "Waiting for dog..." |
| `greeting` | "Greeting [dog_name]..." with speaker icon |
| `command` | "Commanding: SIT" (uppercase trick name) with command icon |
| `watching` | **Progress bar**: `progress / target_sec` as percentage + "Hold it..." |
| `success` | Green checkmark + confetti/celebration animation |
| `failed` | Brief red X indicator |
| `retry` | "Trying again..." |
| `completed` | Summary screen showing rewards given, stages completed |

### 1.3 Stage Indicator

Always show current progress: `Stage 2 of 5` or visual dots/stepper.

### 1.4 Progress Bar (Critical for "watching" state)

```dart
// Example: progress=2.5, target_sec=5.0 → 50%
double progressPercent = (data['progress'] ?? 0) / (data['target_sec'] ?? 1);
progressPercent = progressPercent.clamp(0.0, 1.0);

// Show as linear progress bar or circular indicator
LinearProgressIndicator(value: progressPercent)
```

---

## Priority 2: Mode Locking (MUST HAVE)

### 2.1 Listen for `mode_changed` Events

```json
{
  "type": "event",
  "event": "mode_changed",
  "data": {
    "mode": "mission",
    "previous_mode": "idle",
    "locked": true,
    "lock_reason": "Mission active: sit_training"
  }
}
```

### 2.2 Disable Mode Selector When Locked

When `locked: true`:
- Grey out or disable the mode selector dropdown/buttons
- Show lock icon with tooltip: "Mode locked: Mission active"
- User cannot change mode until mission ends

### 2.3 Check Mode Status on Connect

```dart
// On app connect/resume, check current mode
final response = await http.get(Uri.parse('$robotUrl/mode'));
final modeInfo = jsonDecode(response.body);
if (modeInfo['mode_info']?['locked'] == true) {
  // Disable mode selector
  setState(() => _modeLocked = true);
}
```

---

## Priority 3: Missions Screen (SHOULD HAVE)

### 3.1 List Available Missions

```http
GET /missions/available
```

Display as cards or list items:
- Mission name (display friendly: "Sit Training" not "sit_training")
- Description
- Duration estimate
- Max rewards

### 3.2 Start Mission Button

```dart
Future<void> startMission(String missionName, String? dogId) async {
  final response = await http.post(
    Uri.parse('$robotUrl/missions/start'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'mission_name': missionName,
      'parameters': {'dog_id': dogId}
    }),
  );
  
  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    if (data['success']) {
      // Navigate to mission progress view
      // Mode will auto-lock via WebSocket event
    }
  }
}
```

### 3.3 Stop Mission Button

Always visible during active mission:
```dart
Future<void> stopMission() async {
  await http.post(Uri.parse('$robotUrl/missions/stop'));
}
```

---

## Priority 4: Programs UI (NICE TO HAVE)

Programs are multi-mission sequences (e.g., "Puppy Basics" = sit + down + quiet).

### 4.1 List Programs

```http
GET /programs/available
```

### 4.2 Program Progress Display

```
Puppy Basics
━━━━━━━━━━━━━━━━━━━━━━
Mission 2 of 3: Down Sustained
✓ Sit Training (complete)
● Down Sustained (in progress)
○ Quiet Progressive (pending)

Treats: 4/15 | Time: 7:00
```

---

## Priority 5: Reports Dashboard (NICE TO HAVE)

### 5.1 Weekly Summary

```http
GET /reports/weekly
```

Display:
- Total treats dispensed
- Bark count with trend arrow
- Coaching sessions count
- Highlights list

### 5.2 Dog Progress

```http
GET /reports/dog/{dog_id}?weeks=8
```

Display:
- Trick success rates with trend indicators
- Bark trend (decreasing = good)
- Improvement areas vs strengths

---

## Known Issue: File Picker Still Wrong

From Build 28, the music upload is still opening Apple Music library instead of file picker.

**Fix Required:**
```dart
// WRONG - Opens Apple Music library
// ... whatever is currently implemented

// RIGHT - Use FilePicker package for actual file selection
import 'package:file_picker/file_picker.dart';

Future<void> pickMusicFile() async {
  try {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['mp3'],  // Only allow MP3 files
      allowMultiple: false,
    );
    
    if (result != null && result.files.isNotEmpty) {
      final file = result.files.first;
      if (file.path != null) {
        // Validate it's actually an MP3
        if (!file.name.toLowerCase().endsWith('.mp3')) {
          _showError('Please select an MP3 file');
          return;
        }
        // Read file and upload...
        final bytes = await File(file.path!).readAsBytes();
        await _uploadSong(file.name, bytes);
      }
    }
  } catch (e) {
    print('[UPLOAD] Error: $e');
    _showError('Could not open file picker');
  }
}
```

---

## WebSocket Event Handling Summary

Add handlers for these events in your WebSocket listener:

```dart
void _handleWebSocketMessage(dynamic message) {
  final data = jsonDecode(message);

  switch (data['event']) {
    case 'mission_progress':
      _updateMissionProgress(data['data']);
      break;
    case 'mode_changed':
      _handleModeChange(data['data']);
      break;
    case 'audio_state':
      _handleAudioState(data['data']);
      break;
    case 'bark_detected':
      _handleBarkDetected(data['data']);
      break;
    case 'dog_detected':
      _handleDogDetected(data['data']);
      break;
    case 'treat_dispensed':
      _handleTreatDispensed(data['data']);
      break;
  }
}

// NEW: Handle audio state changes to sync music player UI
void _handleAudioState(Map<String, dynamic> data) {
  setState(() {
    _isMusicPlaying = data['playing'] ?? false;
    _currentTrack = data['track'];
    _playlistIndex = data['playlist_index'] ?? 0;
  });
}
```

---

## Testing Checklist

### Must Pass:
- [ ] Start mission → see "Waiting for dog" status
- [ ] Dog appears → status changes to "greeting" then "command"
- [ ] During "watching" → progress bar shows and fills
- [ ] Trick success → green checkmark appears
- [ ] Mission complete → summary screen shows
- [ ] Mode selector disabled during active mission
- [ ] Stop button works mid-mission

### Should Pass:
- [ ] Programs list displays correctly
- [ ] Can start a program
- [ ] Weekly report loads and displays

### Audio State Sync (NEW):
- [ ] Play music → app button shows "stop/pause" state
- [ ] Stop music → app button shows "play" state
- [ ] Track name updates when song changes
- [ ] Toggle button actually stops (doesn't restart song)
- [ ] Dog progress shows trick trends

### File Upload Fix:
- [ ] Upload button opens file picker (not Apple Music)
- [ ] Only MP3 files selectable
- [ ] Upload completes and robot receives file

---

## Data Models Reference

See `BUILD31_APP_GUIDE.md` lines 1240-1572 for complete Swift/Dart model definitions including:
- `MissionStatus`
- `MissionProgress`
- `Program`
- `ProgramStatus`
- `WeeklyReport`
- `DogProgress`

---

## UI Decisions (Confirmed by Morgan)

1. **Mission progress UI** → **Overlay on video** (not separate screen)
2. **Programs** → **Fold into existing Missions UI** (not dedicated tab)
3. **Progress bar style** → **Circular pie** indicator

### Overlay Implementation Notes

The mission progress should appear as a semi-transparent overlay on the video stream:
- Top area: Stage indicator (e.g., "Stage 2/5")
- Center: Current status with icon (waiting, commanding, watching)
- **Circular pie progress** during "watching" state showing hold time
- Bottom: Dog name + rewards count

Example layout:
```
┌─────────────────────────────────────┐
│  VIDEO STREAM                       │
│                                     │
│     ┌─────────────────────┐         │
│     │   Stage 2 of 5      │         │
│     │                     │         │
│     │    ◐ 67%           │  ← Circular pie
│     │   "Hold it..."      │         │
│     │                     │         │
│     │   🐕 Elsa  🦴 2     │         │
│     └─────────────────────┘         │
│                                     │
│                          [STOP]     │
└─────────────────────────────────────┘
```

---

*Build 31 - Focus on making AI detection VISIBLE and training GUIDED*
