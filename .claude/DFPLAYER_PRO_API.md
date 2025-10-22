# DFPlayer Pro Audio API Documentation

## Overview
The DFPlayer Pro is integrated using AT commands over serial (/dev/ttyAMA0 at 115200 baud). The implementation is in `core/hardware/audio_controller.py`.

## Key AT Commands Used

### Initialization
```python
AT+VOL=22       # Set volume (0-30)
AT+LED=OFF      # Turn off LED
AT+PLAYMODE=3   # Play one song and pause
AT+PROMPT=OFF   # Turn off prompt tone
AT+AMP=ON       # Turn on amplifier
```

### Playback Commands
```python
AT+PLAYFILE=/talks/0003.mp3  # Play file by path
AT+PLAYNUM=15                 # Play file by number
AT+PLAY=PP                    # Play/pause toggle
AT+PLAY=NEXT                  # Next track
AT+PLAY=LAST                  # Previous track
```

### Query Commands
```python
AT+QUERY=5  # Query current file
AT+QUERY=3  # Query play time
```

## File Mappings (from config/settings.py)

The audio files are stored on the DFPlayer's SD card in two folders:

### /talks/ folder - Voice commands
- 0001.mp3 - Scooby Intro
- 0003.mp3 - Elsa (dog name)
- 0004.mp3 - Bezik (dog name)
- 0008.mp3 - Good Dog
- 0010.mp3 - Lie Down
- 0011.mp3 - Quiet
- 0012.mp3 - No
- 0013.mp3 - Treat
- 0015.mp3 - Sit
- 0016.mp3 - Spin
- 0017.mp3 - Stay

### /02/ folder - Music and effects
- Various background music and sound effects

## Usage in Mission System

### 1. Import the controller
```python
from core.hardware.audio_controller import AudioController
```

### 2. Initialize
```python
self.audio_controller = AudioController()
```

### 3. Play audio during training
```python
# Play by name (uses settings.py mapping)
self.audio_controller.play_sound("ELSA")
self.audio_controller.play_sound("SIT")
self.audio_controller.play_sound("GOOD_DOG")

# Play by path
self.audio_controller.play_file_by_path("/talks/0015.mp3")

# Play by number
self.audio_controller.play_file_by_number(15)
```

### 4. Cleanup
```python
self.audio_controller.cleanup()
```

## Integration with live_mission_training.py

The mission training system now uses the DFPlayer Pro for:
1. **Attention phase** - Calls dog's name (e.g., /talks/0003.mp3 for Elsa)
2. **Command phase** - Issues command (e.g., /talks/0015.mp3 for Sit)
3. **Reward phase** - Praises dog (e.g., /talks/0008.mp3 for Good Dog)

The `_play_audio()` method in `live_mission_training.py` handles this:
```python
def _play_audio(self, audio_file):
    if self.audio_controller:
        success = self.audio_controller.play_file_by_path(audio_file)
        if success:
            time.sleep(2.0)  # Allow audio to play
        return success
```

## Testing

Run the audio test script:
```bash
python test_dfplayer_audio.py
```

This tests:
1. Volume control
2. Playing by name mapping
3. Playing by file path
4. Playing by file number

## Troubleshooting

1. **No serial connection**: Check /dev/ttyAMA0 permissions and that serial is enabled
2. **No audio output**: Check MAX4544 relay is set to DFPlayer (GPIO 12 = LOW)
3. **Files not found**: Verify SD card has /talks/ and /02/ folders with MP3 files
4. **Command timeout**: Add delays between AT commands (0.1s minimum)

## Hardware Notes

- Serial: /dev/ttyAMA0 at 115200 baud
- Audio relay: GPIO 12 controls MAX4544 (LOW=DFPlayer, HIGH=Pi USB)
- Power: DFPlayer needs stable 5V supply
- SD Card: FAT32 formatted with folders /talks/ and /02/