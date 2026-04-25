# Flutter App: Fix Treat Counter Showing Both Robots

## Problem
When user has multiple robots (tb1, tb2), the status bar flashes between both treat counts instead of showing only the currently connected robot's count.

## Root Cause
The app is receiving treat status events from all robots (via relay) and displaying all of them instead of filtering by the active `robot_id`.

## Robot-Side Events (Already Correct)
Each robot sends its own treat status with `robot_id`:
```json
{
  "type": "treat_status",
  "robot_id": "treatbot1",
  "treats_loaded": 44,
  "treats_remaining": 38,
  "treats_low": false
}
```

The robot also publishes events on dispense:
```json
{
  "type": "reward",
  "subtype": "treat_dispensed",
  "robot_id": "treatbot1",
  "treats_remaining": 37,
  "treats_low": false
}
```

## App-Side Fix

### 1. Track Active Robot
```dart
class RobotConnection {
  String? activeRobotId;  // Set when connecting to a robot
  
  void onConnect(String robotId) {
    activeRobotId = robotId;
  }
}
```

### 2. Filter Treat Events by Robot ID
```dart
void onTreatStatusEvent(Map<String, dynamic> event) {
  final eventRobotId = event['robot_id'];
  
  // Ignore events from other robots
  if (eventRobotId != activeRobotId) {
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
  if (event['robot_id'] != activeRobotId) return;
  
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
  activeRobotId = robotId;
  
  // Request current status
  sendCommand({'command': 'get_treat_status'});
}
```

### 5. Show "Treats Running Low" Alert
The robot sends `treats_low: true` when < 5 treats remain:
```dart
if (treatsLow && !_lowTreatAlertShown) {
  _lowTreatAlertShown = true;
  showSnackBar('Treats running low on $activeRobotId!');
}
```

## Multi-Robot Dashboard (Optional)
For master users who want to see all robots:

```dart
// Store treats per robot
Map<String, int> treatsByRobot = {};

void onTreatStatusEvent(Map<String, dynamic> event) {
  final robotId = event['robot_id'];
  treatsByRobot[robotId] = event['treats_remaining'];
  
  // Only update main display for active robot
  if (robotId == activeRobotId) {
    treatsRemaining = event['treats_remaining'];
  }
}

// Dashboard widget showing all robots
Widget buildFleetStatus() {
  return Column(
    children: treatsByRobot.entries.map((e) => 
      ListTile(
        title: Text(e.key),
        trailing: Text('${e.value} treats'),
        selected: e.key == activeRobotId,
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
