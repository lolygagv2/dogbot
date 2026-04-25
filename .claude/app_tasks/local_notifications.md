# Flutter App: Local Notifications for Activity Events

## Overview
Enable iOS/Android notifications for WIM-Z activity events (barks, treats, coaching results, etc.) so they appear on lock screen, notification center, and Apple Watch.

**This is NOT remote push** - the app already receives events via WebSocket. We just need to trigger local notifications when events arrive.

## Implementation

### 1. Add Package
```yaml
# pubspec.yaml
dependencies:
  flutter_local_notifications: ^17.0.0
```

### 2. iOS Configuration
```xml
<!-- ios/Runner/Info.plist -->
<key>UIBackgroundModes</key>
<array>
  <string>fetch</string>
  <string>remote-notification</string>
</array>
```

### 3. Initialize (main.dart or notification_service.dart)
```dart
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

final FlutterLocalNotificationsPlugin notifications = FlutterLocalNotificationsPlugin();

Future<void> initNotifications() async {
  const DarwinInitializationSettings iosSettings = DarwinInitializationSettings(
    requestAlertPermission: true,
    requestBadgePermission: true,
    requestSoundPermission: true,
  );
  
  const InitializationSettings initSettings = InitializationSettings(
    iOS: iosSettings,
    android: AndroidInitializationSettings('@mipmap/ic_launcher'),
  );
  
  await notifications.initialize(initSettings);
  
  // Request permission on iOS
  await notifications
      .resolvePlatformSpecificImplementation<IOSFlutterLocalNotificationsPlugin>()
      ?.requestPermissions(alert: true, badge: true, sound: true);
}
```

### 4. Show Notification When Event Arrives
```dart
Future<void> showActivityNotification({
  required String title,
  required String body,
  String? payload,
}) async {
  const NotificationDetails details = NotificationDetails(
    iOS: DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    ),
    android: AndroidNotificationDetails(
      'wimz_activity',
      'WIM-Z Activity',
      channelDescription: 'Dog training activity notifications',
      importance: Importance.high,
      priority: Priority.high,
    ),
  );
  
  await notifications.show(
    DateTime.now().millisecondsSinceEpoch ~/ 1000, // unique ID
    title,
    body,
    details,
    payload: payload,
  );
}
```

### 5. Hook Into Existing Event Handler
Where the app currently processes WebSocket/relay events for the activity feed, add:

```dart
void onActivityEvent(Map<String, dynamic> event) {
  // Existing activity feed logic...
  _activityFeed.add(event);
  
  // NEW: Trigger notification if app is backgrounded
  if (WidgetsBinding.instance.lifecycleState != AppLifecycleState.resumed) {
    final title = _getNotificationTitle(event);
    final body = _getNotificationBody(event);
    showActivityNotification(title: title, body: body);
  }
}

String _getNotificationTitle(Map<String, dynamic> event) {
  switch (event['type']) {
    case 'bark_detected': return '🐕 Bark Detected';
    case 'treat_dispensed': return '🦴 Treat Given';
    case 'coach_success': return '🎉 Training Success';
    case 'coach_failure': return '📋 Training Attempt';
    case 'silent_guardian_alert': return '⚠️ Guardian Alert';
    default: return 'WIM-Z Activity';
  }
}

String _getNotificationBody(Map<String, dynamic> event) {
  final dogName = event['dog_name'] ?? 'Dog';
  switch (event['type']) {
    case 'bark_detected': 
      return '$dogName barked (${event['emotion'] ?? 'detected'})';
    case 'treat_dispensed': 
      return '$dogName got a treat!';
    case 'coach_success': 
      return '$dogName completed ${event['trick']}!';
    case 'coach_failure': 
      return '$dogName attempted ${event['trick']}';
    default: 
      return event['message'] ?? 'New activity';
  }
}
```

### 6. Settings Toggle (Optional)
Add a user preference to enable/disable notifications:
```dart
// In settings screen
SwitchListTile(
  title: Text('Activity Notifications'),
  subtitle: Text('Show alerts on lock screen & Apple Watch'),
  value: notificationsEnabled,
  onChanged: (value) => setNotificationsEnabled(value),
)
```

## Event Types to Notify
| Event | Notify | Note |
|-------|--------|------|
| `bark_detected` | Yes | Include emotion if available |
| `treat_dispensed` | Yes | Dog name + count |
| `coach_success` | Yes | Dog name + trick |
| `coach_failure` | Optional | Could be noisy |
| `silent_guardian_alert` | Yes | Important |
| `treats_low` | Yes | Inventory warning |
| `mode_change` | No | Too frequent |
| `dog_detected` | No | Too frequent |

## Apple Watch
No extra work needed - iOS automatically mirrors notifications to paired Apple Watch.

## Testing
1. Connect to robot
2. Background the app (swipe up)
3. Trigger a bark or dispense a treat
4. Notification should appear on lock screen + Watch

## Estimated Effort
~2-3 hours including testing
