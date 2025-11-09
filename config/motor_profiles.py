#!/usr/bin/env python3
"""
Dynamic motor profiles - switchable speed modes
Integrates with Xbox controller and app for on-the-fly speed changes
"""

from config.motor_tuning import SafeMode, SportMode, TurboMode
import logging

logger = logging.getLogger(__name__)

class MotorProfileManager:
    """Manages switchable motor speed profiles"""

    def __init__(self):
        # Available profiles
        self.profiles = {
            'safe': SafeMode,
            'sport': SportMode,
            'turbo': TurboMode
        }

        # Default profile
        self.current_profile_name = 'sport'
        self.current_profile = SportMode

        # Turbo boost state (for Xbox right trigger)
        self.turbo_boost_active = False
        self.base_profile_name = 'sport'  # Profile when not boosting

    def set_profile(self, profile_name: str):
        """Set the base motor profile"""
        profile_name = profile_name.lower()
        if profile_name in self.profiles:
            self.base_profile_name = profile_name
            if not self.turbo_boost_active:
                self.current_profile_name = profile_name
                self.current_profile = self.profiles[profile_name]
                logger.info(f"Motor profile changed to: {profile_name.upper()}")
                logger.info(f"Max voltage: {self.current_profile.MAX_VOLTAGE}V ({self.current_profile.MAX_PWM}% PWM)")
            return True
        else:
            logger.error(f"Unknown profile: {profile_name}")
            return False

    def enable_turbo_boost(self):
        """Enable turbo boost (Xbox right trigger)"""
        if not self.turbo_boost_active:
            self.turbo_boost_active = True
            self.current_profile_name = 'turbo'
            self.current_profile = TurboMode
            logger.info("TURBO BOOST ENGAGED! 8V power mode")
            return True
        return False

    def disable_turbo_boost(self):
        """Disable turbo boost, return to base profile"""
        if self.turbo_boost_active:
            self.turbo_boost_active = False
            self.current_profile_name = self.base_profile_name
            self.current_profile = self.profiles[self.base_profile_name]
            logger.info(f"Turbo boost released, returning to {self.base_profile_name.upper()}")
            return True
        return False

    def get_max_pwm(self):
        """Get current max PWM based on active profile"""
        return self.current_profile.MAX_PWM

    def get_max_voltage(self):
        """Get current max voltage based on active profile"""
        return self.current_profile.MAX_VOLTAGE

    def get_pwm_for_speed(self, speed_percent: int):
        """Convert user speed (0-100) to safe PWM for current profile"""
        safe_pwm = int(speed_percent * self.current_profile.MAX_PWM / 100)
        return min(safe_pwm, self.current_profile.MAX_PWM)

    def get_status(self):
        """Get current profile status"""
        return {
            'current_profile': self.current_profile_name,
            'base_profile': self.base_profile_name,
            'turbo_active': self.turbo_boost_active,
            'max_voltage': self.current_profile.MAX_VOLTAGE,
            'max_pwm': self.current_profile.MAX_PWM,
            'description': self.current_profile.DESCRIPTION
        }

# Global instance
_profile_manager = None

def get_profile_manager():
    """Get global profile manager instance"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = MotorProfileManager()
    return _profile_manager

# For backward compatibility
def get_current_max_pwm():
    """Get current max PWM setting"""
    return get_profile_manager().get_max_pwm()

def get_current_max_voltage():
    """Get current max voltage setting"""
    return get_profile_manager().get_max_voltage()