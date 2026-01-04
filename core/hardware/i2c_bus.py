#!/usr/bin/env python3
"""
Shared I2C bus singleton for all I2C devices
Prevents conflicts between PCA9685 servo controller and ADS1115 ADC
"""

import board
import busio
import threading

_i2c_instance = None
_i2c_lock = threading.RLock()  # Reentrant lock for nested calls


def get_i2c_bus():
    """
    Get the shared I2C bus instance.

    This singleton ensures all I2C devices (PCA9685, ADS1115, etc.)
    use the same bus instance to prevent conflicts.

    Returns:
        The shared I2C bus instance
    """
    global _i2c_instance
    with _i2c_lock:
        if _i2c_instance is None:
            _i2c_instance = busio.I2C(board.SCL, board.SDA)
        return _i2c_instance


def get_bus_lock():
    """Return the lock for external synchronization"""
    return _i2c_lock
