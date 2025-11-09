"""DFRobot Devastator Motor Specifications
For reference when configuring motor control
"""

MOTOR_SPECS = {
    "model": "DFRobot Devastator",
    "rated_voltage": 6.0,  # V
    "operating_voltage_range": (2.0, 7.5),  # V
    "gear_reduction_ratio": "45:1",
    "shaft_diameter": 4,  # mm
    "no_load_speed": 133,  # RPM @ 6V
    "no_load_current": 0.13,  # A
    "locked_rotor_torque": 4.5,  # kg.cm
    "locked_rotor_current": 2.3,  # A
}

# PWM calculations for 14V system with L298N
L298N_INPUT_VOLTAGE = 14.0  # V
L298N_VOLTAGE_DROP = 1.4  # V (typical)
EFFECTIVE_MAX_VOLTAGE = L298N_INPUT_VOLTAGE - L298N_VOLTAGE_DROP  # 12.6V

# Safe PWM duty cycles for these motors
SAFE_PWM_RANGE = {
    "minimum": 0.16,  # ~2V effective
    "cruise": 0.35,   # ~4.4V effective
    "normal": 0.48,   # ~6V effective (rated voltage)
    "maximum": 0.60,  # ~7.5V effective (max operating)
}

def calculate_pwm_for_voltage(target_voltage):
    """Calculate PWM duty cycle for desired voltage"""
    return target_voltage / EFFECTIVE_MAX_VOLTAGE