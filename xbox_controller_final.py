#!/usr/bin/env python3
"""
Xbox Controller for DogBot - Final Working Version
Clean controls with proper button mappings
"""

import struct
import os
import sys
import time
import select
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.motion.motor import MotorService

# Import servo controller directly for gimbal
try:
    from core.hardware.servo_controller import ServoController
    SERVO_AVAILABLE = True
except Exception as e:
    SERVO_AVAILABLE = False
    print(f"[WARNING] Servo controller not available: {e}")

try:
    from services.reward.dispenser import get_dispenser_service
    DISPENSER_AVAILABLE = True
except Exception as e:
    DISPENSER_AVAILABLE = False
    print(f"[WARNING] Dispenser service not available: {e}")

try:
    from services.media.sfx import get_sfx_service
    SFX_AVAILABLE = True
except Exception as e:
    SFX_AVAILABLE = False
    print(f"[WARNING] Sound effects service not available: {e}")

try:
    import cv2
    CAMERA_AVAILABLE = True
except:
    CAMERA_AVAILABLE = False
    print("[WARNING] Camera (cv2) not available")

class XboxControllerFinal:
    # Event format for Linux input events
    # Changed to interpret axis values as signed
    EVENT_FORMAT = 'llHHi'  # long long unsigned_short unsigned_short signed_int (changed last from I to i)
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

    # Event types
    EV_KEY = 0x01  # Button
    EV_ABS = 0x03  # Axis

    # Button codes for Xbox controller
    BTN_A = 304      # A button
    BTN_B = 305      # B button
    BTN_X = 307      # X button
    BTN_Y = 308      # Y button
    BTN_LB = 310     # Left bumper
    BTN_RB = 311     # Right bumper
    BTN_BACK = 314   # Back/View button
    BTN_START = 315  # Start/Menu button
    BTN_XBOX = 316   # Xbox button
    BTN_LSTICK = 317 # Left stick click
    BTN_RSTICK = 318 # Right stick click

    # Axis codes
    AXIS_LX = 0      # Left stick X
    AXIS_LY = 1      # Left stick Y
    AXIS_LT = 2      # Left trigger
    AXIS_RX = 3      # Right stick X
    AXIS_RY = 4      # Right stick Y
    AXIS_RT = 5      # Right trigger
    AXIS_DPAD_X = 16 # D-pad X
    AXIS_DPAD_Y = 17 # D-pad Y

    def __init__(self):
        self.motor = None
        self.servo = None
        self.dispenser = None
        self.sfx = None
        self.camera = None
        self.running = False
        self.event_file = None

        # Controller state
        self.axes = {
            self.AXIS_LX: 0,
            self.AXIS_LY: 0,
            self.AXIS_RX: 0,
            self.AXIS_RY: 0,
            self.AXIS_LT: -32767,  # Triggers start at min
            self.AXIS_RT: -32767
        }

        # Camera gimbal state
        self.pan_angle = 90  # Center position
        self.tilt_angle = 90  # Center position
        self.gimbal_speed = 3  # Degrees per update

        # Sound state
        self.sound_list = ['good_dog', 'bark', 'whistle', 'beep', 'alert']
        self.current_sound_index = 0
        self.sounds_enabled = True

        # Control settings
        self.base_speed = 30  # Start with slower speed
        self.turbo_multiplier = 2.0  # More speed when needed
        self.slow_multiplier = 0.3  # Very slow for precision
        self.deadzone = 10000  # Larger deadzone (about 30% of stick range)
        self.trigger_threshold = 500  # Triggers are 0-1023 range

        # State flags
        self.turbo_mode = False
        self.slow_mode = False
        self.motors_enabled = True
        self._last_state = 'stopped'  # Initialize state to prevent initial stop error

    def initialize(self):
        """Initialize motor and controller"""
        print("Initializing Xbox Controller...")

        # Initialize motor service with fallback to gpioset
        try:
            self.motor = MotorService()
            self.motor.initialize()
            print("✓ Motors initialized")
        except Exception as e:
            print(f"Motor initialization warning: {e}")
            # Try fallback to gpioset controller
            from core.hardware.motor_controller_gpioset import MotorControllerGpioset, MotorDirection
            self.motor_controller = MotorControllerGpioset()
            print("✓ Using gpioset motor controller (fallback)")

        # Initialize servo controller for gimbal
        if SERVO_AVAILABLE:
            try:
                self.servo = ServoController()
                self.servo.initialize()
                # Center servos (channel 0 = pan, channel 1 = tilt)
                self.servo.set_angle(0, 90)  # Pan center
                self.servo.set_angle(1, 90)  # Tilt center
                print("✓ Camera gimbal initialized")
            except Exception as e:
                print(f"✗ Gimbal initialization failed: {e}")

        # Initialize dispenser service
        if DISPENSER_AVAILABLE:
            try:
                self.dispenser = get_dispenser_service()
                print("✓ Treat dispenser initialized")
            except Exception as e:
                print(f"✗ Dispenser initialization failed: {e}")

        # Initialize sound service
        if SFX_AVAILABLE:
            try:
                self.sfx = get_sfx_service()
                print("✓ Sound effects initialized")
            except Exception as e:
                print(f"✗ Sound initialization failed: {e}")

        # Initialize camera for photo capture
        if CAMERA_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 4K width
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4K height
                print("✓ Camera initialized for 4K capture")
            except Exception as e:
                print(f"✗ Camera initialization failed: {e}")

        # Find Xbox controller device
        event_path = self.find_xbox_device()
        if not event_path:
            print("✗ No Xbox controller found!")
            return False

        # Open the event device
        try:
            self.event_file = open(event_path, 'rb')
            os.set_blocking(self.event_file.fileno(), False)
            print(f"✓ Xbox controller connected at {event_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to open controller: {e}")
            return False

    def find_xbox_device(self):
        """Find the Xbox controller event device"""
        for i in range(20):
            try:
                name_path = f'/sys/class/input/event{i}/device/name'
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        if 'Xbox' in f.read():
                            return f'/dev/input/event{i}'
            except:
                pass
        return None

    def start(self):
        """Start the controller loop"""
        if not self.event_file:
            print("Controller not initialized!")
            return

        self.running = True
        print("\nXbox Controller Active!")
        print("=" * 50)
        print("MOVEMENT:")
        print("  Left Stick: Move robot (forward/back/turn)")
        print("  Right Stick: Camera gimbal (pan/tilt)")
        print("  Right Trigger: Turbo mode (hold)")
        print("  Left Trigger: Slow/precise mode (hold)")
        print("\nBUTTONS:")
        print("  A: Enable motors")
        print("  B: Emergency stop")
        print("  X: Normal speed")
        print("  Y: Play 'good dog' sound")
        print("  LB: Dispense treat 🦴")
        print("  RB: Take 4K photo 📸")
        print("  Start: Pause/Resume")
        print("  Back: Exit")
        print("\nD-PAD:")
        print("  Up: Disable sounds")
        print("  Down: Play current sound")
        print("  Left/Right: Select sound")
        print("=" * 50)
        print("\nReady for control!\n")

        # Start control loop
        self.control_loop()

    def control_loop(self):
        """Main control loop"""
        last_motor_update = 0
        update_interval = 0.1  # Update motors at 10Hz

        while self.running:
            try:
                # Check for input with timeout
                readable, _, _ = select.select([self.event_file], [], [], 0.01)

                if readable:
                    # Read event
                    data = self.event_file.read(self.EVENT_SIZE)

                    if data and len(data) == self.EVENT_SIZE:
                        _, _, event_type, code, value = struct.unpack(self.EVENT_FORMAT, data)

                        # Handle button events
                        if event_type == self.EV_KEY:
                            self.handle_button(code, value)

                        # Handle axis events
                        elif event_type == self.EV_ABS:
                            self.handle_axis(code, value)

                # Update motor control at fixed rate
                now = time.time()
                if now - last_motor_update > update_interval:
                    self.update_motors()
                    self.update_gimbal()  # Update camera gimbal
                    last_motor_update = now

            except BlockingIOError:
                # No data available, continue
                pass
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.01)

    def handle_button(self, code, value):
        """Handle button press/release"""
        if value:  # Button pressed
            if code == self.BTN_A:
                print("[A] Motors enabled")
                self.motors_enabled = True

            elif code == self.BTN_B:
                print("[B] EMERGENCY STOP!")
                self.motors_enabled = False
                self.motor.manual_drive('stop', 0)

            elif code == self.BTN_X:
                print("[X] Normal speed")
                self.base_speed = 50

            elif code == self.BTN_Y:
                print("[Y] Playing good dog sound!")
                self.play_sound('good_dog')

            elif code == self.BTN_LB:
                print("[LB] Dispensing treat!")
                self.dispense_treat()

            elif code == self.BTN_RB:
                print("[RB] Taking 4K photo!")
                self.take_photo()

            elif code == self.BTN_START:
                self.motors_enabled = not self.motors_enabled
                state = "enabled" if self.motors_enabled else "paused"
                print(f"[START] Motors {state}")

            elif code == self.BTN_BACK:
                print("[BACK] Exiting...")
                self.stop()

    def handle_axis(self, code, value):
        """Handle axis movement"""
        self.axes[code] = value

        # Handle triggers for speed modulation
        if code == self.AXIS_RT:
            # Right trigger for turbo
            self.turbo_mode = (value > self.trigger_threshold)
            if self.turbo_mode:
                print("TURBO MODE ON")

        elif code == self.AXIS_LT:
            # Left trigger for slow mode
            self.slow_mode = (value > self.trigger_threshold)
            if self.slow_mode:
                print("SLOW MODE ON")

        # Handle D-pad for sound control
        elif code == self.AXIS_DPAD_X:
            if value < 0:  # D-pad left
                self.previous_sound()
            elif value > 0:  # D-pad right
                self.next_sound()

        elif code == self.AXIS_DPAD_Y:
            if value < 0:  # D-pad up
                print("[D-Pad Up] Sounds OFF")
                self.sounds_enabled = False
            elif value > 0:  # D-pad down
                print("[D-Pad Down] Play last sound")
                self.play_current_sound()

    def update_motors(self):
        """Update motor speeds based on current controller state"""
        if not self.motors_enabled:
            return

        # Get left stick values
        lx = self.axes.get(self.AXIS_LX, 0)
        ly = self.axes.get(self.AXIS_LY, 0)

        # Apply deadzone
        if abs(lx) < self.deadzone:
            lx = 0
        if abs(ly) < self.deadzone:
            ly = 0

        # Calculate speed
        speed = self.base_speed

        if self.turbo_mode and not self.slow_mode:
            speed = int(speed * self.turbo_multiplier)
        elif self.slow_mode and not self.turbo_mode:
            speed = int(speed * self.slow_multiplier)

        # Normalize stick values (-1 to 1) with proper clamping
        lx_norm = max(-1.0, min(1.0, lx / 32767.0)) if lx != 0 else 0
        ly_norm = max(-1.0, min(1.0, -ly / 32767.0)) if ly != 0 else 0  # Invert Y for intuitive control

        # Apply exponential curve for better control (less sensitive near center)
        if abs(lx_norm) > 0:
            lx_norm = lx_norm * abs(lx_norm)  # Square it but keep sign
        if abs(ly_norm) > 0:
            ly_norm = ly_norm * abs(ly_norm)  # Square it but keep sign

        # Determine movement with better threshold
        if abs(ly_norm) > 0.2:  # Forward/backward has priority (lower threshold)
            # Scale speed based on stick position (variable speed)
            variable_speed = int(speed * min(1.0, abs(ly_norm) * 1.5))

            if ly_norm > 0:
                self.motor.manual_drive('forward', variable_speed)
                print(f"↑ Forward {variable_speed}%", end='\r')
                self._last_state = 'forward'
            else:
                self.motor.manual_drive('backward', variable_speed)
                print(f"↓ Backward {variable_speed}%", end='\r')
                self._last_state = 'backward'

        elif abs(lx_norm) > 0.2:  # Turning (lower threshold)
            # Scale turn speed based on stick position
            variable_speed = int(speed * min(1.0, abs(lx_norm) * 1.5))

            if lx_norm < 0:
                self.motor.manual_drive('left', variable_speed)
                print(f"← Left {variable_speed}%", end='\r')
                self._last_state = 'left'
            else:
                self.motor.manual_drive('right', variable_speed)
                print(f"→ Right {variable_speed}%", end='\r')
                self._last_state = 'right'

        else:  # No significant input
            # For stop, just don't send any motor commands rather than sending stop repeatedly
            # Only send stop once when transitioning to stopped state
            if not hasattr(self, '_last_state') or self._last_state != 'stopped':
                self.motor.manual_drive('stop', 0)
                print("■ Stopped      ", end='\r')
                self._last_state = 'stopped'

    def update_gimbal(self):
        """Update camera gimbal based on right stick"""
        if not self.servo:
            return

        # Get right stick values
        rx = self.axes.get(self.AXIS_RX, 0)
        ry = self.axes.get(self.AXIS_RY, 0)

        # Apply deadzone
        if abs(rx) < self.deadzone:
            rx = 0
        if abs(ry) < self.deadzone:
            ry = 0

        # Update angles based on stick input
        if rx != 0 or ry != 0:
            # Pan left/right
            self.pan_angle += (rx / 32767.0) * self.gimbal_speed
            self.pan_angle = max(0, min(180, self.pan_angle))

            # Tilt up/down (inverted for intuitive control)
            self.tilt_angle -= (ry / 32767.0) * self.gimbal_speed
            self.tilt_angle = max(0, min(180, self.tilt_angle))

            # Set servo angles directly
            try:
                self.servo.set_angle(0, int(self.pan_angle))  # Channel 0 = pan
                self.servo.set_angle(1, int(self.tilt_angle))  # Channel 1 = tilt
            except Exception as e:
                pass  # Silently ignore errors to avoid spam

    def dispense_treat(self):
        """Dispense a treat"""
        if self.dispenser:
            try:
                self.dispenser.dispense_treat(reason="xbox_controller")
                print("🦴 Treat dispensed!")
            except Exception as e:
                print(f"Dispenser error: {e}")
        else:
            print("Dispenser not available")

    def take_photo(self):
        """Take a 4K photo"""
        if self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"photo_4k_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Photo saved: {filename}")
                else:
                    print("Failed to capture photo")
            except Exception as e:
                print(f"Camera error: {e}")

    def play_sound(self, sound_name):
        """Play a specific sound"""
        if self.sfx and self.sounds_enabled:
            try:
                self.sfx.play_sound(sound_name)
                print(f"🔊 Playing: {sound_name}")
            except Exception as e:
                print(f"Sound error: {e}")
        elif not self.sfx:
            print("Sound system not available")
        else:
            print("Sounds disabled")

    def play_current_sound(self):
        """Play the current selected sound"""
        if self.sounds_enabled and self.sound_list:
            sound = self.sound_list[self.current_sound_index]
            print(f"🔊 Playing: {sound}")
            self.play_sound(sound)

    def next_sound(self):
        """Select next sound in list"""
        if self.sound_list:
            self.current_sound_index = (self.current_sound_index + 1) % len(self.sound_list)
            print(f"[D-Pad Right] Next sound: {self.sound_list[self.current_sound_index]}")

    def previous_sound(self):
        """Select previous sound in list"""
        if self.sound_list:
            self.current_sound_index = (self.current_sound_index - 1) % len(self.sound_list)
            print(f"[D-Pad Left] Previous sound: {self.sound_list[self.current_sound_index]}")

    def stop(self):
        """Stop the controller and clean up"""
        self.running = False

        # Stop motors
        if self.motor:
            self.motor.manual_drive('stop', 0)

        # Release camera
        if self.camera:
            self.camera.release()

        # Close event file
        if self.event_file:
            self.event_file.close()

        print("\nController stopped.")


def main():
    """Run the Xbox controller"""
    controller = XboxControllerFinal()

    if not controller.initialize():
        print("Failed to initialize controller!")
        return

    try:
        controller.start()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()