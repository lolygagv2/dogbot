#!/usr/bin/env python3
"""
Xbox Controller Diagnostic - Figure out what's really happening
"""

import struct
import os
import time
import select

class XboxDiagnostic:
    EVENT_FORMAT = 'llHHI'
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

    def __init__(self):
        self.event_file = None
        self.button_names = {
            304: 'A', 305: 'B', 307: 'X', 308: 'Y',
            310: 'LB', 311: 'RB', 314: 'Back', 315: 'Start',
            316: 'Xbox', 317: 'L-Stick', 318: 'R-Stick'
        }
        self.axis_names = {
            0: 'Left-X', 1: 'Left-Y', 2: 'L-Trigger',
            3: 'Right-X', 4: 'Right-Y', 5: 'R-Trigger',
            16: 'DPad-X', 17: 'DPad-Y'
        }

    def find_xbox(self):
        for i in range(20):
            try:
                with open(f'/sys/class/input/event{i}/device/name', 'r') as f:
                    if 'Xbox' in f.read():
                        return f'/dev/input/event{i}'
            except:
                pass
        return None

    def run(self):
        device = self.find_xbox()
        if not device:
            print("No Xbox controller found!")
            return

        self.event_file = open(device, 'rb')
        os.set_blocking(self.event_file.fileno(), False)

        print("Xbox Controller Diagnostic")
        print("="*60)
        print("Press buttons and move sticks to see raw values")
        print("Press Ctrl+C to exit\n")
        print("Format: [Button/Axis] Name = Raw Value (Normalized)")
        print("-"*60)

        try:
            while True:
                readable, _, _ = select.select([self.event_file], [], [], 0.01)
                if readable:
                    data = self.event_file.read(self.EVENT_SIZE)
                    if data and len(data) == self.EVENT_SIZE:
                        _, _, event_type, code, value = struct.unpack(self.EVENT_FORMAT, data)

                        if event_type == 0x01:  # Button
                            name = self.button_names.get(code, f'Unknown-{code}')
                            state = 'PRESSED' if value else 'released'
                            print(f"[BUTTON] {name:10} = {state}")

                        elif event_type == 0x03:  # Axis
                            name = self.axis_names.get(code, f'Unknown-{code}')
                            normalized = value / 32767.0 if code in [0,1,3,4] else value

                            # Show stick direction interpretation
                            direction = ""
                            if code == 0:  # Left X
                                direction = "LEFT" if value < -8000 else "RIGHT" if value > 8000 else "center"
                            elif code == 1:  # Left Y
                                direction = "UP" if value < -8000 else "DOWN" if value > 8000 else "center"

                            print(f"[AXIS]   {name:10} = {value:6} (norm: {normalized:+.2f}) {direction}")

        except KeyboardInterrupt:
            print("\nDiagnostic complete!")
        finally:
            self.event_file.close()

if __name__ == "__main__":
    diag = XboxDiagnostic()
    diag.run()