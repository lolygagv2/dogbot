#!/usr/bin/env python3
"""
FastAPI server for TreatBot REST endpoints
Provides monitoring and control interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import logging
import threading

# TreatBot imports
from core.state import get_state, SystemMode
from core.store import get_store
from services.reward.dispenser import get_dispenser_service
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from orchestrators.mode_fsm import get_mode_fsm
from core.hardware.audio_controller import AudioController
from core.hardware.led_controller import LEDController, LEDMode
from config.settings import AudioFiles

# Singleton LED controller to prevent GPIO conflicts
_led_controller = None
_led_lock = threading.Lock()

def get_led_controller():
    global _led_controller
    with _led_lock:
        if _led_controller is None:
            _led_controller = LEDController()
        return _led_controller

# Models
class ModeRequest(BaseModel):
    mode: str
    duration: Optional[float] = None

class MissionRequest(BaseModel):
    mission_name: str
    parameters: Optional[Dict[str, Any]] = None

class TreatRequest(BaseModel):
    dog_id: Optional[str] = None
    reason: str = "manual"
    count: int = 1

class SequenceRequest(BaseModel):
    sequence_name: str
    context: Optional[Dict[str, Any]] = None
    interrupt: bool = False

# DFPlayer models
class DFPlayerPlayRequest(BaseModel):
    filepath: str

class DFPlayerVolumeRequest(BaseModel):
    volume: int  # 0-30

class DFPlayerNumberRequest(BaseModel):
    number: int

class DFPlayerSoundRequest(BaseModel):
    sound_name: str

# LED control models
class LEDColorRequest(BaseModel):
    color: str  # Color name or hex

class LEDBrightnessRequest(BaseModel):
    brightness: float  # 0.1 to 1.0

class LEDModeRequest(BaseModel):
    mode: str  # LEDMode values

class LEDAnimationRequest(BaseModel):
    animation: str  # spinning_dot, pulse_color, rainbow_cycle
    color: Optional[str] = "blue"
    delay: Optional[float] = 0.05
    steps: Optional[int] = 20

class LEDCustomColorRequest(BaseModel):
    red: int    # 0-255
    green: int  # 0-255
    blue: int   # 0-255

# FastAPI app
app = FastAPI(
    title="TreatBot API",
    description="REST API for TreatBot dog training robot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logger = logging.getLogger('TreatBotAPI')

# Cleanup handler
import atexit

def cleanup_hardware():
    """Clean up hardware resources on exit"""
    global _led_controller
    if _led_controller:
        logger.info("Cleaning up LED controller...")
        _led_controller.cleanup()
        _led_controller = None

atexit.register(cleanup_hardware)

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "TreatBot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    state = get_state()

    return {
        "status": "healthy",
        "mode": state.get_mode().value,
        "emergency": state.is_emergency(),
        "uptime": state.get_mode_duration(),
        "timestamp": state.get_full_state()["timestamp"]
    }

# Mode endpoints
@app.get("/mode")
async def get_mode():
    """Get current system mode"""
    state = get_state()
    mode_fsm = get_mode_fsm()

    return {
        "current_mode": state.get_mode().value,
        "previous_mode": state.previous_mode.value,
        "time_in_mode": state.get_mode_duration(),
        "fsm_status": mode_fsm.get_status()
    }

@app.post("/mode/set")
async def set_mode(request: ModeRequest):
    """Set system mode"""
    try:
        mode = SystemMode(request.mode)
        mode_fsm = get_mode_fsm()

        if request.duration:
            success = mode_fsm.set_mode_override(mode, request.duration)
        else:
            success = mode_fsm.force_mode(mode, "API request")

        if success:
            return {"success": True, "mode": mode.value}
        else:
            raise HTTPException(status_code=400, detail="Mode change failed")

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

@app.post("/mode/clear_override")
async def clear_mode_override():
    """Clear mode override"""
    mode_fsm = get_mode_fsm()
    mode_fsm.clear_override("API request")
    return {"success": True}

# Mission endpoints (placeholder - would need mission system)
@app.get("/missions/status")
async def get_mission_status():
    """Get current mission status"""
    state = get_state()
    return {
        "mission": state.mission.to_dict(),
        "active": state.mission.state.value != "inactive"
    }

@app.post("/missions/start")
async def start_mission(request: MissionRequest):
    """Start a mission"""
    # Placeholder - would integrate with mission system
    return {
        "success": False,
        "message": "Mission system not yet implemented",
        "mission_name": request.mission_name
    }

@app.post("/missions/stop")
async def stop_mission():
    """Stop current mission"""
    state = get_state()
    state.reset_mission()
    return {"success": True}

# Treat endpoints
@app.post("/treat/dispense")
async def dispense_treat(request: TreatRequest):
    """Dispense treats"""
    try:
        dispenser = get_dispenser_service()

        if request.count == 1:
            success = dispenser.dispense_treat(
                dog_id=request.dog_id,
                reason=request.reason
            )
        else:
            dispensed = dispenser.dispense_multiple(
                count=request.count,
                dog_id=request.dog_id,
                reason=request.reason
            )
            success = dispensed > 0

        return {
            "success": success,
            "treats_dispensed": request.count if success else 0,
            "dog_id": request.dog_id,
            "reason": request.reason
        }

    except Exception as e:
        logger.error(f"Treat dispense error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/treat/status")
async def get_treat_status():
    """Get treat dispenser status"""
    dispenser = get_dispenser_service()
    return dispenser.get_status()

@app.post("/treat/force_reward")
async def force_reward(dog_id: str = "api_test"):
    """Force a reward for testing"""
    try:
        reward_logic = get_reward_logic()
        success = reward_logic.force_reward(dog_id, "api_test")

        return {
            "success": success,
            "dog_id": dog_id,
            "type": "forced_reward"
        }

    except Exception as e:
        logger.error(f"Force reward error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Sequence endpoints
@app.post("/sequence/execute")
async def execute_sequence(request: SequenceRequest):
    """Execute a sequence"""
    try:
        sequence_engine = get_sequence_engine()

        sequence_id = sequence_engine.execute_sequence(
            request.sequence_name,
            request.context or {},
            request.interrupt
        )

        if sequence_id:
            return {
                "success": True,
                "sequence_id": sequence_id,
                "sequence_name": request.sequence_name
            }
        else:
            raise HTTPException(status_code=400, detail="Sequence execution failed")

    except Exception as e:
        logger.error(f"Sequence execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sequence/status")
async def get_sequence_status():
    """Get sequence engine status"""
    sequence_engine = get_sequence_engine()
    return sequence_engine.get_status()

@app.post("/sequence/stop/{sequence_id}")
async def stop_sequence(sequence_id: str):
    """Stop a specific sequence"""
    sequence_engine = get_sequence_engine()
    success = sequence_engine.stop_sequence(sequence_id)

    return {
        "success": success,
        "sequence_id": sequence_id
    }

@app.post("/sequence/stop_all")
async def stop_all_sequences():
    """Stop all sequences"""
    sequence_engine = get_sequence_engine()
    count = sequence_engine.stop_all_sequences()

    return {
        "success": True,
        "sequences_stopped": count
    }

# Telemetry endpoints
@app.get("/telemetry")
async def get_telemetry():
    """Get system telemetry"""
    state = get_state()
    store = get_store()

    return {
        "system": state.get_status_summary(),
        "hardware": state.hardware.to_dict(),
        "detection": state.detection.to_dict(),
        "recent_events": store.get_recent_events(10),
        "database_stats": store.get_database_stats()
    }

@app.get("/telemetry/hardware")
async def get_hardware_status():
    """Get hardware status"""
    state = get_state()
    return state.hardware.to_dict()

@app.get("/telemetry/detection")
async def get_detection_status():
    """Get detection status"""
    state = get_state()
    return state.detection.to_dict()

# Event endpoints
@app.get("/events/recent")
async def get_recent_events(limit: int = 50, event_type: Optional[str] = None):
    """Get recent events"""
    store = get_store()
    events = store.get_recent_events(limit, event_type)

    return {
        "events": events,
        "count": len(events),
        "limit": limit,
        "event_type": event_type
    }

@app.get("/events/stats")
async def get_event_stats():
    """Get event statistics"""
    store = get_store()
    return store.get_database_stats()

# Dog endpoints
@app.get("/dogs")
async def get_dogs():
    """Get all dogs"""
    store = get_store()
    dogs = store.get_dog_stats()

    return {
        "dogs": dogs,
        "count": len(dogs)
    }

@app.get("/dogs/{dog_id}")
async def get_dog(dog_id: str):
    """Get specific dog info"""
    store = get_store()
    reward_logic = get_reward_logic()

    dogs = store.get_dog_stats(dog_id)
    if not dogs:
        raise HTTPException(status_code=404, detail="Dog not found")

    dog = dogs[0]

    # Add reward stats
    reward_stats = reward_logic.get_dog_reward_stats(dog_id)

    return {
        "dog": dog,
        "reward_stats": reward_stats
    }

@app.get("/dogs/{dog_id}/rewards")
async def get_dog_rewards(dog_id: str, days: int = 7):
    """Get dog's reward history"""
    store = get_store()
    rewards = store.get_reward_history(dog_id, days)

    return {
        "dog_id": dog_id,
        "rewards": rewards,
        "count": len(rewards),
        "days": days
    }

# Manual Control endpoints
class ManualDriveRequest(BaseModel):
    direction: str
    speed: Optional[int] = 50
    duration: Optional[float] = 1.0

class KeyboardControlRequest(BaseModel):
    key: str

@app.post("/manual/drive")
async def manual_drive(request: ManualDriveRequest):
    """Manual vehicle control (RC car style)"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        success = motor_service.manual_drive(
            direction=request.direction,
            speed=request.speed,
            duration=request.duration
        )

        return {
            "success": success,
            "direction": request.direction,
            "speed": request.speed,
            "duration": request.duration
        }

    except Exception as e:
        logger.error(f"Manual drive error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/keyboard")
async def keyboard_control(request: KeyboardControlRequest):
    """Process keyboard input for manual control"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        success = motor_service.keyboard_control(request.key)

        return {
            "success": success,
            "key": request.key,
            "action": "processed" if success else "ignored"
        }

    except Exception as e:
        logger.error(f"Keyboard control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/emergency_stop")
async def manual_emergency_stop():
    """Emergency stop for manual control"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        motor_service.emergency_stop()

        return {
            "success": True,
            "message": "Manual control emergency stop"
        }

    except Exception as e:
        logger.error(f"Manual emergency stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/manual/status")
async def get_manual_control_status():
    """Get manual control status"""
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()

        return motor_service.get_status()

    except Exception as e:
        logger.error(f"Manual status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manual/mode/{mode}")
async def set_manual_mode(mode: str):
    """Set manual control mode (manual/auto/disabled)"""
    try:
        from services.motion.motor import get_motor_service, MovementMode
        motor_service = get_motor_service()

        mode_map = {
            'manual': MovementMode.MANUAL,
            'auto': MovementMode.AUTO,
            'disabled': MovementMode.DISABLED
        }

        if mode not in mode_map:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

        success = motor_service.set_movement_mode(mode_map[mode])

        return {
            "success": success,
            "mode": mode
        }

    except Exception as e:
        logger.error(f"Set manual mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# DFPlayer Control endpoints
@app.get("/audio/status")
async def get_audio_status():
    """Get DFPlayer and audio relay status"""
    try:
        audio = AudioController()
        status = audio.get_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Audio status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/files")
async def get_audio_files():
    """Get list of available audio files"""
    try:
        # Get all AudioFiles class attributes
        files = {}
        for attr_name in dir(AudioFiles):
            if not attr_name.startswith('_'):
                filepath = getattr(AudioFiles, attr_name)
                if isinstance(filepath, str) and filepath.startswith('/'):
                    files[attr_name.lower()] = {
                        "name": attr_name,
                        "path": filepath,
                        "category": filepath.split('/')[1] if '/' in filepath else "unknown"
                    }

        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Audio files listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/play/file")
async def play_audio_file(request: DFPlayerPlayRequest):
    """Play audio file by path"""
    try:
        audio = AudioController()
        success = audio.play_file_by_path(request.filepath)

        return {
            "success": success,
            "filepath": request.filepath,
            "message": f"Playing {request.filepath}" if success else "Failed to play file"
        }
    except Exception as e:
        logger.error(f"Audio play file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/play/number")
async def play_audio_number(request: DFPlayerNumberRequest):
    """Play audio file by number"""
    try:
        audio = AudioController()
        success = audio.play_file_by_number(request.number)

        return {
            "success": success,
            "number": request.number,
            "message": f"Playing file #{request.number}" if success else "Failed to play file"
        }
    except Exception as e:
        logger.error(f"Audio play number error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/play/sound")
async def play_audio_sound(request: DFPlayerSoundRequest):
    """Play audio by sound name (from AudioFiles)"""
    try:
        audio = AudioController()
        success = audio.play_sound(request.sound_name)

        return {
            "success": success,
            "sound_name": request.sound_name,
            "message": f"Playing {request.sound_name}" if success else f"Unknown sound: {request.sound_name}"
        }
    except Exception as e:
        logger.error(f"Audio play sound error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/volume")
async def set_audio_volume(request: DFPlayerVolumeRequest):
    """Set DFPlayer volume (0-30)"""
    try:
        if not 0 <= request.volume <= 30:
            raise HTTPException(status_code=400, detail="Volume must be between 0 and 30")

        audio = AudioController()
        success = audio.set_volume(request.volume)

        return {
            "success": success,
            "volume": request.volume,
            "message": f"Volume set to {request.volume}" if success else "Failed to set volume"
        }
    except Exception as e:
        logger.error(f"Audio volume error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/pause")
async def pause_audio():
    """Pause/resume audio playback"""
    try:
        audio = AudioController()
        success = audio.play_pause_toggle()

        return {
            "success": success,
            "message": "Audio pause/resume toggled" if success else "Failed to toggle pause"
        }
    except Exception as e:
        logger.error(f"Audio pause error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/next")
async def next_audio():
    """Play next track"""
    try:
        audio = AudioController()
        success = audio.play_next()

        return {
            "success": success,
            "message": "Playing next track" if success else "Failed to play next"
        }
    except Exception as e:
        logger.error(f"Audio next error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/previous")
async def previous_audio():
    """Play previous track"""
    try:
        audio = AudioController()
        success = audio.play_previous()

        return {
            "success": success,
            "message": "Playing previous track" if success else "Failed to play previous"
        }
    except Exception as e:
        logger.error(f"Audio previous error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/relay/pi")
async def switch_to_pi_audio():
    """Switch audio relay to Pi USB audio"""
    try:
        audio = AudioController()
        success = audio.switch_to_pi_audio()

        return {
            "success": success,
            "audio_path": "Pi USB Audio" if success else "DFPlayer (switch failed)",
            "message": "Switched to Pi audio" if success else "Failed to switch to Pi audio"
        }
    except Exception as e:
        logger.error(f"Audio relay Pi error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/relay/dfplayer")
async def switch_to_dfplayer_audio():
    """Switch audio relay to DFPlayer"""
    try:
        audio = AudioController()
        success = audio.switch_to_dfplayer()

        return {
            "success": success,
            "audio_path": "DFPlayer" if success else "Pi USB Audio (switch failed)",
            "message": "Switched to DFPlayer" if success else "Failed to switch to DFPlayer"
        }
    except Exception as e:
        logger.error(f"Audio relay DFPlayer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/relay/status")
async def get_relay_status():
    """Get audio relay status"""
    try:
        audio = AudioController()
        status = audio.get_relay_status()

        return {
            "success": True,
            "relay_status": status
        }
    except Exception as e:
        logger.error(f"Audio relay status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio/test")
async def test_audio_system():
    """Test audio system (relay switching)"""
    try:
        audio = AudioController()
        success = audio.test_relay_switching()

        return {
            "success": success,
            "message": "Audio system test completed" if success else "Audio system test failed"
        }
    except Exception as e:
        logger.error(f"Audio test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LED Control endpoints
@app.get("/leds/status")
async def get_led_status():
    """Get LED system status"""
    try:
        leds = get_led_controller()
        status = leds.get_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"LED status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leds/colors")
async def get_available_colors():
    """Get list of available LED colors"""
    try:
        # Standard colors available in the system
        colors = {
            'off': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'dim_white': (30, 30, 30),
            'warm_white': (255, 180, 120)
        }

        return {
            "success": True,
            "colors": colors,
            "count": len(colors)
        }
    except Exception as e:
        logger.error(f"LED colors error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leds/modes")
async def get_led_modes():
    """Get available LED modes"""
    try:
        modes = [mode.value for mode in LEDMode]
        return {
            "success": True,
            "modes": modes,
            "count": len(modes)
        }
    except Exception as e:
        logger.error(f"LED modes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/color")
async def set_led_color(request: LEDColorRequest):
    """Set LED to solid color"""
    try:
        leds = get_led_controller()
        success = leds.set_solid_color(request.color)

        return {
            "success": success,
            "color": request.color,
            "message": f"LEDs set to {request.color}" if success else f"Failed to set color {request.color}"
        }
    except Exception as e:
        logger.error(f"LED color error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/custom_color")
async def set_led_custom_color(request: LEDCustomColorRequest):
    """Set LED to custom RGB color"""
    try:
        if not all(0 <= val <= 255 for val in [request.red, request.green, request.blue]):
            raise HTTPException(status_code=400, detail="RGB values must be between 0 and 255")

        leds = get_led_controller()
        color_tuple = (request.red, request.green, request.blue)
        success = leds.set_solid_color(color_tuple)

        return {
            "success": success,
            "color": f"rgb({request.red}, {request.green}, {request.blue})",
            "rgb": color_tuple,
            "message": f"LEDs set to custom color" if success else "Failed to set custom color"
        }
    except Exception as e:
        logger.error(f"LED custom color error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/brightness")
async def set_led_brightness(request: LEDBrightnessRequest):
    """Set LED brightness (0.1 to 1.0)"""
    try:
        if not 0.1 <= request.brightness <= 1.0:
            raise HTTPException(status_code=400, detail="Brightness must be between 0.1 and 1.0")

        leds = get_led_controller()
        success = leds.set_neopixel_brightness(request.brightness)

        return {
            "success": success,
            "brightness": request.brightness,
            "message": f"Brightness set to {request.brightness}" if success else "Failed to set brightness"
        }
    except Exception as e:
        logger.error(f"LED brightness error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/mode")
async def set_led_mode(request: LEDModeRequest):
    """Set LED mode (with animations)"""
    try:
        # Validate mode
        valid_modes = [mode.value for mode in LEDMode]
        if request.mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}")

        leds = get_led_controller()
        led_mode = LEDMode(request.mode)
        leds.set_mode(led_mode)

        return {
            "success": True,
            "mode": request.mode,
            "message": f"LED mode set to {request.mode}"
        }
    except Exception as e:
        logger.error(f"LED mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/animation")
async def start_led_animation(request: LEDAnimationRequest):
    """Start custom LED animation"""
    try:
        leds = get_led_controller()

        # Map animation names to methods
        animations = {
            'spinning_dot': leds.spinning_dot,
            'pulse_color': leds.pulse_color,
            'rainbow_cycle': leds.rainbow_cycle
        }

        if request.animation not in animations:
            raise HTTPException(status_code=400, detail=f"Invalid animation. Available: {list(animations.keys())}")

        # Start the animation with parameters
        if request.animation == 'spinning_dot':
            leds.start_animation(animations[request.animation], request.color, request.delay)
        elif request.animation == 'pulse_color':
            leds.start_animation(animations[request.animation], request.color, request.steps, request.delay)
        elif request.animation == 'rainbow_cycle':
            leds.start_animation(animations[request.animation], request.delay)

        return {
            "success": True,
            "animation": request.animation,
            "color": request.color,
            "delay": request.delay,
            "steps": request.steps if request.animation == 'pulse_color' else None,
            "message": f"Started {request.animation} animation"
        }
    except Exception as e:
        logger.error(f"LED animation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/stop")
async def stop_led_animation():
    """Stop all LED animations"""
    try:
        leds = get_led_controller()
        leds.stop_animation()

        return {
            "success": True,
            "message": "LED animations stopped"
        }
    except Exception as e:
        logger.error(f"LED stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/blue/on")
async def blue_led_on():
    """Turn blue LED on"""
    try:
        leds = get_led_controller()
        success = leds.blue_on()

        return {
            "success": success,
            "message": "Blue LED turned on" if success else "Failed to turn blue LED on"
        }
    except Exception as e:
        logger.error(f"Blue LED on error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/blue/off")
async def blue_led_off():
    """Turn blue LED off"""
    try:
        leds = get_led_controller()
        success = leds.blue_off()

        return {
            "success": success,
            "message": "Blue LED turned off" if success else "Failed to turn blue LED off"
        }
    except Exception as e:
        logger.error(f"Blue LED off error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leds/off")
async def turn_leds_off():
    """Turn all LEDs off"""
    try:
        leds = get_led_controller()
        leds.set_mode(LEDMode.OFF)

        return {
            "success": True,
            "message": "All LEDs turned off"
        }
    except Exception as e:
        logger.error(f"LEDs off error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Emergency endpoints
@app.post("/emergency/stop")
async def emergency_stop():
    """Trigger emergency stop"""
    state = get_state()
    state.set_emergency("API emergency stop")

    # Also stop manual control
    try:
        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()
        motor_service.emergency_stop()
    except:
        pass

    return {
        "success": True,
        "message": "Emergency stop triggered"
    }

@app.post("/emergency/clear")
async def clear_emergency():
    """Clear emergency state"""
    state = get_state()
    state.clear_emergency()

    return {
        "success": True,
        "message": "Emergency cleared"
    }

# Bark Detection endpoints
@app.get("/bark/status")
async def get_bark_status():
    """Get bark detection status and statistics"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        return bark_detector.get_status()
    except Exception as e:
        logger.error(f"Failed to get bark detector status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/enable")
async def enable_bark_detection():
    """Enable bark detection"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_enabled(True)
        return {"success": True, "message": "Bark detection enabled"}
    except Exception as e:
        logger.error(f"Failed to enable bark detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/disable")
async def disable_bark_detection():
    """Disable bark detection"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_enabled(False)
        return {"success": True, "message": "Bark detection disabled"}
    except Exception as e:
        logger.error(f"Failed to disable bark detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BarkThresholdRequest(BaseModel):
    threshold: float  # 0.0 to 1.0

@app.post("/bark/threshold")
async def set_bark_threshold(request: BarkThresholdRequest):
    """Set bark detection confidence threshold"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_confidence_threshold(request.threshold)
        return {"success": True, "threshold": request.threshold}
    except Exception as e:
        logger.error(f"Failed to set bark threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class BarkEmotionsRequest(BaseModel):
    emotions: List[str]  # List of emotion names

@app.post("/bark/reward_emotions")
async def set_reward_emotions(request: BarkEmotionsRequest):
    """Set which bark emotions trigger rewards"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.set_reward_emotions(request.emotions)
        return {"success": True, "reward_emotions": request.emotions}
    except Exception as e:
        logger.error(f"Failed to set reward emotions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bark/reset_stats")
async def reset_bark_statistics():
    """Reset bark detection statistics"""
    try:
        from services.perception.bark_detector import get_bark_detector_service
        bark_detector = get_bark_detector_service()
        bark_detector.reset_statistics()
        return {"success": True, "message": "Bark statistics reset"}
    except Exception as e:
        logger.error(f"Failed to reset bark statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    state = get_state()

    # Get status from all major components
    status = {
        "state": state.get_full_state(),
        "mode_fsm": get_mode_fsm().get_status(),
        "sequence_engine": get_sequence_engine().get_status(),
        "reward_logic": get_reward_logic().get_status()
    }

    # Add service status if available
    try:
        from services.perception.detector import get_detector_service
        from services.perception.bark_detector import get_bark_detector_service
        from services.motion.pan_tilt import get_pantilt_service
        from services.reward.dispenser import get_dispenser_service
        from services.media.sfx import get_sfx_service
        from services.media.led import get_led_service

        status["services"] = {
            "detector": get_detector_service().get_status(),
            "bark_detector": get_bark_detector_service().get_status(),
            "pantilt": get_pantilt_service().get_status(),
            "dispenser": get_dispenser_service().get_status(),
            "sfx": get_sfx_service().get_status(),
            "led": get_led_service().get_status()
        }

    except Exception as e:
        logger.warning(f"Could not get service status: {e}")
        status["services"] = {"error": str(e)}

    return status

def create_app():
    """Create FastAPI app (for external use)"""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server"""
    logger.info(f"Starting TreatBot API server on {host}:{port}")

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    run_server(debug=True)