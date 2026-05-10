#!/usr/bin/env python3
"""
Cytron MDD10A motor controller (treatbot3/4/5).

Two-wire-per-motor interface: DIR (digital, sets rotation direction) + PWM
(0-100% duty, sets speed). 9V battery → 9V brushed motors, no encoders, no
PID. Hardware PWM via lgpio.tx_pwm on gpiochip0 (pinctrl-rp1 on Pi 5).

Mirrors the public surface of MotorControllerGpioset so MotorService can
swap it in without further changes (tank_steering, set_motor_speeds,
stop_all, emergency_stop, cleanup, is_initialized).
"""

import logging
import time
from enum import Enum
from typing import Optional

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


class MotorControllerCytron:
    """Cytron MDD10A dual-channel driver. DIR + PWM per motor."""

    GPIO_CHIP = 0  # pinctrl-rp1 on Pi 5 (also valid on Pi 4)
    DEFAULT_PWM_FREQ_HZ = 1000  # MDD10A supports up to 20kHz; 1kHz is quiet + responsive

    def __init__(
        self,
        left_dir_pin: int = 17,
        left_pwm_pin: int = 13,
        right_dir_pin: int = 27,
        right_pwm_pin: int = 19,
        left_invert: bool = False,
        right_invert: bool = False,
        left_multiplier: float = 1.0,
        right_multiplier: float = 1.0,
        max_pwm_pct: int = 100,
        pwm_freq_hz: int = DEFAULT_PWM_FREQ_HZ,
    ):
        self.left_dir_pin = left_dir_pin
        self.left_pwm_pin = left_pwm_pin
        self.right_dir_pin = right_dir_pin
        self.right_pwm_pin = right_pwm_pin
        self.left_invert = left_invert
        self.right_invert = right_invert
        self.left_multiplier = max(0.0, min(2.0, left_multiplier))
        self.right_multiplier = max(0.0, min(2.0, right_multiplier))
        self.max_pwm_pct = max(0, min(100, max_pwm_pct))
        self.pwm_freq_hz = pwm_freq_hz

        self.gpio_chip: Optional[int] = None
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized

    def start(self) -> bool:
        """Alias for initialize() — MotorCommandBus calls .start() on its controller."""
        return self.initialize()

    def set_motor_pwm_direct(self, left: float, right: float) -> bool:
        """
        Direct-PWM interface expected by MotorCommandBus when use_pid=False.
        Inputs are signed percentages (-100..100); routed to set_motor_speeds.
        """
        return self.set_motor_speeds(int(left), int(right))

    def initialize(self) -> bool:
        if not LGPIO_AVAILABLE:
            logger.error("lgpio not installed — Cytron controller cannot start")
            return False

        try:
            self.gpio_chip = lgpio.gpiochip_open(self.GPIO_CHIP)

            for pin in (self.left_dir_pin, self.left_pwm_pin,
                        self.right_dir_pin, self.right_pwm_pin):
                lgpio.gpio_claim_output(self.gpio_chip, pin, lgpio.SET_PULL_NONE)
                lgpio.gpio_write(self.gpio_chip, pin, 0)

            self.initialized = True
            logger.info(
                "Cytron MDD10A initialized — Left(DIR=%d,PWM=%d,inv=%s,mult=%.2f) "
                "Right(DIR=%d,PWM=%d,inv=%s,mult=%.2f) max=%d%% @ %dHz",
                self.left_dir_pin, self.left_pwm_pin, self.left_invert, self.left_multiplier,
                self.right_dir_pin, self.right_pwm_pin, self.right_invert, self.right_multiplier,
                self.max_pwm_pct, self.pwm_freq_hz,
            )
            return True

        except Exception as e:
            logger.error("Cytron init failed: %s", e)
            self.gpio_chip = None
            return False

    def _drive_motor(self, dir_pin: int, pwm_pin: int, signed_speed: int,
                     invert: bool, multiplier: float) -> None:
        """Apply a signed speed (-100..100) to one motor channel."""
        scaled = int(signed_speed * multiplier)
        scaled = max(-100, min(100, scaled))

        forward = scaled >= 0
        if invert:
            forward = not forward

        magnitude = min(abs(scaled), self.max_pwm_pct)

        lgpio.gpio_write(self.gpio_chip, dir_pin, 1 if forward else 0)
        lgpio.tx_pwm(self.gpio_chip, pwm_pin, self.pwm_freq_hz, magnitude)

    def set_motor_speeds(self, left_speed: int, right_speed: int) -> bool:
        """
        Differential drive: signed speeds (-100..100). Used by MotorService._differential_drive.
        Positive = forward for that wheel. Mixing (turning) is the caller's job.
        """
        if not self.initialized:
            return False
        try:
            self._drive_motor(self.left_dir_pin, self.left_pwm_pin,
                              left_speed, self.left_invert, self.left_multiplier)
            self._drive_motor(self.right_dir_pin, self.right_pwm_pin,
                              right_speed, self.right_invert, self.right_multiplier)
            return True
        except Exception as e:
            logger.error("set_motor_speeds failed: %s", e)
            return False

    def tank_steering(self, direction: MotorDirection,
                      speed_percent: Optional[int] = None) -> bool:
        """Tank-style high-level command. Translates direction to differential speeds."""
        if not self.initialized:
            return False

        if direction == MotorDirection.STOP:
            return self.stop_all()

        speed = 50 if speed_percent is None else max(0, min(100, speed_percent))

        if direction == MotorDirection.FORWARD:
            left, right = speed, speed
        elif direction == MotorDirection.BACKWARD:
            left, right = -speed, -speed
        elif direction == MotorDirection.LEFT:
            left, right = -speed, speed
        elif direction == MotorDirection.RIGHT:
            left, right = speed, -speed
        else:
            logger.warning("Unknown direction: %s", direction)
            return False

        return self.set_motor_speeds(left, right)

    def stop_all(self) -> bool:
        if not self.initialized:
            return False
        try:
            lgpio.tx_pwm(self.gpio_chip, self.left_pwm_pin, self.pwm_freq_hz, 0)
            lgpio.tx_pwm(self.gpio_chip, self.right_pwm_pin, self.pwm_freq_hz, 0)
            lgpio.gpio_write(self.gpio_chip, self.left_dir_pin, 0)
            lgpio.gpio_write(self.gpio_chip, self.right_dir_pin, 0)
            return True
        except Exception as e:
            logger.error("stop_all failed: %s", e)
            return False

    def emergency_stop(self) -> bool:
        logger.warning("EMERGENCY STOP")
        return self.stop_all()

    def cleanup(self) -> None:
        try:
            if self.initialized:
                self.stop_all()
            if self.gpio_chip is not None:
                lgpio.gpiochip_close(self.gpio_chip)
        except Exception as e:
            logger.debug("cleanup error: %s", e)
        finally:
            self.gpio_chip = None
            self.initialized = False

    def move_forward(self, speed_percent: Optional[int] = None) -> bool:
        return self.tank_steering(MotorDirection.FORWARD, speed_percent)

    def move_backward(self, speed_percent: Optional[int] = None) -> bool:
        return self.tank_steering(MotorDirection.BACKWARD, speed_percent)

    def turn_left(self, speed_percent: Optional[int] = None) -> bool:
        return self.tank_steering(MotorDirection.LEFT, speed_percent)

    def turn_right(self, speed_percent: Optional[int] = None) -> bool:
        return self.tank_steering(MotorDirection.RIGHT, speed_percent)

    def stop(self) -> bool:
        return self.stop_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Cytron MDD10A bring-up self-test")
    print("Pins: Left DIR=17 PWM=13 | Right DIR=27 PWM=19")
    print("Each step runs 2s. Watch which wheel spins which way.")
    print("If a wheel spins backward, set <side>_invert: true in treatbot3.yaml")
    print("-" * 60)

    c = MotorControllerCytron()
    if not c.initialize():
        raise SystemExit("init failed")

    try:
        for label, fn in (
            ("LEFT wheel forward only", lambda: c.set_motor_speeds(40, 0)),
            ("RIGHT wheel forward only", lambda: c.set_motor_speeds(0, 40)),
            ("BOTH forward 40%", lambda: c.tank_steering(MotorDirection.FORWARD, 40)),
            ("BOTH backward 40%", lambda: c.tank_steering(MotorDirection.BACKWARD, 40)),
            ("Spin LEFT 40%", lambda: c.tank_steering(MotorDirection.LEFT, 40)),
            ("Spin RIGHT 40%", lambda: c.tank_steering(MotorDirection.RIGHT, 40)),
        ):
            print(f"\n→ {label}")
            fn()
            time.sleep(2.0)
            c.stop()
            time.sleep(0.5)
    finally:
        c.cleanup()
        print("\nself-test done")
