#!/usr/bin/env python3
"""
Motor tuning configuration - adjust for speed vs safety trade-off
"""

# Motor specifications
MOTOR_RATED_VOLTAGE = 6.0      # Rated voltage
MOTOR_MAX_VOLTAGE = 7.5         # Absolute maximum per spec
SUPPLY_VOLTAGE = 14.0           # Battery voltage
L298N_DROP = 1.4                # L298N voltage drop
EFFECTIVE_MAX = 12.6            # Max voltage available (14V - 1.4V)

# SPEED PROFILES - Choose your poison!

class SafeMode:
    """Conservative - motors will last forever but slower"""
    NAME = "SAFE"
    MAX_VOLTAGE = 6.0  # Stick to rated voltage
    MAX_PWM = int(MAX_VOLTAGE / EFFECTIVE_MAX * 100)  # ~48%
    DESCRIPTION = "Rated speed, maximum motor life"

class SportMode:
    """Balanced - bit more speed, still reasonable safety"""
    NAME = "SPORT"
    MAX_VOLTAGE = 7.0  # Push it a little (within spec)
    MAX_PWM = int(MAX_VOLTAGE / EFFECTIVE_MAX * 100)  # ~56%
    DESCRIPTION = "15% faster, slightly reduced motor life"

class TurboMode:
    """Performance - good speed boost with acceptable wear"""
    NAME = "TURBO"
    MAX_VOLTAGE = 8.0  # Push beyond rated for speed
    MAX_PWM = int(MAX_VOLTAGE / EFFECTIVE_MAX * 100)  # ~63%
    DESCRIPTION = "33% faster, moderate motor wear"

class YoloMode:
    """DANGER ZONE - You enjoyed this before! Motors will die eventually"""
    NAME = "YOLO"
    MAX_VOLTAGE = 9.0  # 50% over rated (YOU WERE HERE AT 12V!)
    MAX_PWM = int(MAX_VOLTAGE / EFFECTIVE_MAX * 100)  # ~71%
    DESCRIPTION = "50% faster, motors will fail prematurely"

class LudicrousMode:
    """ABSOLUTELY INSANE - What you were actually running before"""
    NAME = "LUDICROUS"
    MAX_VOLTAGE = 12.0  # Double rated voltage (kills motors)
    MAX_PWM = int(MAX_VOLTAGE / EFFECTIVE_MAX * 100)  # ~95%
    DESCRIPTION = "100% faster, motors die in hours/days"

# SELECT YOUR PROFILE HERE
# Change this to adjust speed vs motor life
CURRENT_PROFILE = YoloMode  # <-- CHANGE THIS!

# Export the current settings
MAX_MOTOR_PWM = CURRENT_PROFILE.MAX_PWM
MAX_MOTOR_VOLTAGE = CURRENT_PROFILE.MAX_VOLTAGE

def show_current_settings():
    """Display current motor tuning"""
    print("=" * 60)
    print(f"MOTOR TUNING: {CURRENT_PROFILE.NAME} MODE")
    print(f"Max Voltage: {CURRENT_PROFILE.MAX_VOLTAGE}V")
    print(f"Max PWM: {CURRENT_PROFILE.MAX_PWM}%")
    print(f"Description: {CURRENT_PROFILE.DESCRIPTION}")
    print("=" * 60)

    # Show all options
    print("\nAvailable profiles:")
    for profile in [SafeMode, SportMode, TurboMode, YoloMode, LudicrousMode]:
        status = "← CURRENT" if profile == CURRENT_PROFILE else ""
        print(f"  {profile.NAME:10s}: {profile.MAX_VOLTAGE:4.1f}V ({profile.MAX_PWM:2d}% PWM) - {profile.DESCRIPTION} {status}")

    print("\nTo change: Edit CURRENT_PROFILE in motor_tuning.py")

    # Calculate speed vs rated
    speed_factor = CURRENT_PROFILE.MAX_VOLTAGE / MOTOR_RATED_VOLTAGE
    print(f"\nSpeed vs rated: {speed_factor:.0%}")

    if CURRENT_PROFILE.MAX_VOLTAGE > MOTOR_MAX_VOLTAGE:
        print("⚠️  WARNING: Running above manufacturer max voltage!")
        print("⚠️  Motor life will be significantly reduced!")
    elif CURRENT_PROFILE.MAX_VOLTAGE > MOTOR_RATED_VOLTAGE:
        print("⚠️  Note: Running above rated voltage")

if __name__ == "__main__":
    show_current_settings()