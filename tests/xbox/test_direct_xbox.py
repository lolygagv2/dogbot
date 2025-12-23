#!/usr/bin/env python3
"""
Direct Xbox test - minimal approach to bypass all complex systems
"""

import struct
import time

def test_direct_xbox():
    """Test direct Xbox input without motor controller"""

    print("🎮 DIRECT XBOX INPUT TEST")
    print("========================")
    print("Testing joystick input without motor system")
    print("Move LEFT STICK - should see immediate output")
    print("Press Ctrl+C to exit")
    print()

    try:
        print("📂 Opening /dev/input/js0...")

        with open('/dev/input/js0', 'rb') as js:
            print("✅ Joystick device opened successfully")
            print("👀 Waiting for input events...")
            print()

            event_count = 0
            while event_count < 50:  # Limit to 50 events for testing
                try:
                    # Read 8 bytes
                    event = js.read(8)
                    if len(event) == 8:
                        time_stamp, value, event_type, number = struct.unpack('IhBB', event)

                        event_count += 1

                        if event_type == 2:  # Axis
                            normalized = value / 32767.0
                            if number == 0:
                                print(f"🎮 LEFT STICK X: {normalized:.3f} (raw: {value})")
                            elif number == 1:
                                print(f"🎮 LEFT STICK Y: {normalized:.3f} (raw: {value})")
                            elif number == 2:
                                print(f"🎮 LEFT TRIGGER: {value}")
                            elif number == 5:
                                print(f"🎮 RIGHT TRIGGER: {value}")

                        elif event_type == 1:  # Button
                            state = "PRESSED" if value else "RELEASED"
                            button_names = {0: "A", 1: "B", 2: "X", 3: "Y"}
                            button_name = button_names.get(number, f"Button{number}")
                            print(f"🔘 {button_name}: {state}")

                            if number == 1 and value:  # B button pressed
                                print("🚨 B BUTTON - Exiting test!")
                                break

                except Exception as e:
                    print(f"⚠️ Read error: {e}")
                    break

        print(f"\n✅ Test completed - processed {event_count} events")

    except FileNotFoundError:
        print("❌ /dev/input/js0 not found - Xbox controller not connected")
    except PermissionError:
        print("❌ Permission denied - try running with sudo")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_xbox()