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

# TreatBot imports
from core.state import get_state, SystemMode
from core.store import get_store
from services.reward.dispenser import get_dispenser_service
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from orchestrators.mode_fsm import get_mode_fsm

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
        from services.motion.pan_tilt import get_pantilt_service
        from services.reward.dispenser import get_dispenser_service
        from services.media.sfx import get_sfx_service
        from services.media.led import get_led_service

        status["services"] = {
            "detector": get_detector_service().get_status(),
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