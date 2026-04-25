# Flutter App: Fix Treat Counter Showing Both Robots

## Problem
When user has multiple robots (tb1, tb2), the status bar flashes between both treat counts instead of showing only the currently connected robot's count.

## Root Cause
The app is receiving treat status events from all robots (via relay) and displaying all of them instead of filtering by the active `device_id`.

## Robot-Side Events (Already Correct)
Each robot sends events via relay with `device_id` added automatically:
```json
{
  "event": "treats_low",
  "device_id": "abc123-device-uuid",
  "treats_remaining": 4,
  "timestamp": 1714012345.678
}
```

The robot also publishes reward events:
```json
{
  "event": "treat_dispensed",
  "device_id": "abc123-device-uuid",
  "treats_remaining": 37,
  "dog_id": "aruco_315",
  "reason": "coach_success"
}
```

**Note:** The relay adds `device_id` automatically via `send_event()`. The robot's `device_id` comes from config (unique per robot).

## App-Side Fix

### 1. Track Active Robot
```dart
class RobotConnection {
  String? activeDeviceId;  // Set when connecting to a robot
  
  void onConnect(String robotId) {
    activeDeviceId = robotId;
  }
}
```

### 2. Filter Treat Events by Robot ID
```dart
void onTreatStatusEvent(Map<String, dynamic> event) {
  final eventDeviceId = event['device_id'];
  
  // Ignore events from other robots
  if (eventDeviceId != activeDeviceId) {
    return;
  }
  
  // Update UI with this robot's treat count
  setState(() {
    treatsRemaining = event['treats_remaining'];
    treatsLow = event['treats_low'] ?? false;
  });
}
```

### 3. Handle Reward Events Similarly
```dart
void onRewardEvent(Map<String, dynamic> event) {
  if (event['subtype'] != 'treat_dispensed') return;
  if (event['device_id'] != activeDeviceId) return;
  
  setState(() {
    treatsRemaining = event['treats_remaining'];
    treatsLow = event['treats_low'] ?? false;
  });
}
```

### 4. Request Status on Connect
When connecting to a robot, request current treat status:
```dart
void onRobotConnected(String robotId) {
  activeDeviceId = robotId;
  
  // Request current status
  sendCommand({'command': 'get_treat_status'});
}
```

### 5. Show "Treats Running Low" Alert
The robot sends `treats_low: true` when < 5 treats remain:
```dart
if (treatsLow && !_lowTreatAlertShown) {
  _lowTreatAlertShown = true;
  showSnackBar('Treats running low on $activeDeviceId!');
}
```

## Multi-Robot Dashboard (Optional)
For master users who want to see all robots:

```dart
// Store treats per robot
Map<String, int> treatsByDevice = {};

void onTreatStatusEvent(Map<String, dynamic> event) {
  final robotId = event['device_id'];
  treatsByDevice[robotId] = event['treats_remaining'];
  
  // Only update main display for active robot
  if (robotId == activeDeviceId) {
    treatsRemaining = event['treats_remaining'];
  }
}

// Dashboard widget showing all robots
Widget buildFleetStatus() {
  return Column(
    children: treatsByDevice.entries.map((e) => 
      ListTile(
        title: Text(e.key),
        trailing: Text('${e.value} treats'),
        selected: e.key == activeDeviceId,
      )
    ).toList(),
  );
}
```

## API Endpoints (Robot-Side)
Already available:
- `GET /treat/status` - Returns current count
- `POST /treat/counter/set` - Set count after refill
- `POST /treat/counter/reset` - Reset to 0

## Testing
1. Connect to tb1
2. Verify only tb1 treat count shows
3. Have tb2 dispense a treat (via another phone or Xbox)
4. Verify tb1 display doesn't change
5. Switch to tb2, verify tb2 count now shows

## Estimated Effort
~1-2 hours
