#!/usr/bin/env python3
"""
Test script for 165 LED NeoPixel strip (Adafruit 332 LED-per-meter)
Tests all patterns and verifies LED count

Usage:
    python3 tests/hardware/test_led_165.py
"""
import sys
sys.path.append('/home/morgan/dogbot')

import time
import board
import neopixel
from config.settings import SystemSettings


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB"""
    h = h / 60.0
    i = int(h) % 6
    f = h - int(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q

    return (int(r * 255), int(g * 255), int(b * 255))


def test_led_strip():
    """Test all LED patterns on 165 LED strip"""
    print("=" * 60)
    print(f"NeoPixel 165 LED Strip Test")
    print(f"LED Count: {SystemSettings.NEOPIXEL_COUNT}")
    print(f"Brightness: {SystemSettings.NEOPIXEL_BRIGHTNESS}")
    print("=" * 60)

    # Initialize pixels directly
    print("\nInitializing NeoPixels on GPIO 12...")
    try:
        pixels = neopixel.NeoPixel(
            board.D12,
            SystemSettings.NEOPIXEL_COUNT,
            brightness=SystemSettings.NEOPIXEL_BRIGHTNESS,
            auto_write=False,
            pixel_order=neopixel.GRB
        )
        print("NeoPixels initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize NeoPixels: {e}")
        return False

    try:
        # Test 1: LED count verification
        print("\n[Test 1] LED Count Verification")
        print("  Lighting every 10th LED white...")
        pixels.fill((0, 0, 0))
        for i in range(0, SystemSettings.NEOPIXEL_COUNT, 10):
            pixels[i] = (255, 255, 255)
        pixels.show()
        expected = SystemSettings.NEOPIXEL_COUNT // 10
        print(f"  You should see {expected} white LEDs spaced evenly")
        input("  Press Enter to continue...")

        # Test 2: Full strip color fill
        print("\n[Test 2] Full Strip Color Fill")
        colors = [
            ((255, 0, 0), "Red"),
            ((0, 255, 0), "Green"),
            ((0, 0, 255), "Blue"),
            ((255, 255, 0), "Yellow"),
            ((0, 255, 255), "Cyan"),
            ((255, 0, 255), "Magenta"),
        ]
        for color, name in colors:
            print(f"  {name}...", end=" ", flush=True)
            pixels.fill(color)
            pixels.show()
            time.sleep(0.5)
            print("OK")

        # Test 3: Sequential lighting (comet test)
        print("\n[Test 3] Chase/Comet Effect")
        print("  Running comet with 25-LED tail...")
        tail_length = 25
        for _ in range(2):  # Two full loops
            for pos in range(SystemSettings.NEOPIXEL_COUNT):
                pixels.fill((0, 0, 0))
                for i in range(tail_length):
                    pixel_pos = (pos - i) % SystemSettings.NEOPIXEL_COUNT
                    brightness = 1.0 - (i / tail_length)
                    pixels[pixel_pos] = (0, int(255 * brightness), int(255 * brightness))
                pixels.show()
                time.sleep(0.015)
        print("  Chase effect complete!")

        # Test 4: Rainbow gradient
        print("\n[Test 4] Rainbow Gradient")
        print("  Displaying static rainbow across strip...")
        for i in range(SystemSettings.NEOPIXEL_COUNT):
            hue = i * 360 / SystemSettings.NEOPIXEL_COUNT
            r, g, b = hsv_to_rgb(hue, 1.0, 0.5)
            pixels[i] = (r, g, b)
        pixels.show()
        input("  Press Enter to continue...")

        # Test 5: Flowing gradient
        print("\n[Test 5] Flowing Gradient (5 seconds)")
        print("  Watch the rainbow flow...")
        start = time.time()
        hue_offset = 0
        while time.time() - start < 5:
            for i in range(SystemSettings.NEOPIXEL_COUNT):
                hue = (hue_offset + (i * 360 / SystemSettings.NEOPIXEL_COUNT)) % 360
                r, g, b = hsv_to_rgb(hue, 1.0, 0.6)
                pixels[i] = (r, g, b)
            pixels.show()
            hue_offset = (hue_offset + 1) % 360
            time.sleep(0.03)
        print("  Gradient flow complete!")

        # Test 6: Fire effect
        print("\n[Test 6] Fire Effect (5 seconds)")
        print("  Simulating fire...")
        import random
        fire_colors = [
            (255, 0, 0),
            (255, 50, 0),
            (255, 100, 0),
            (255, 150, 0),
            (255, 200, 50),
        ]
        heat = [0] * SystemSettings.NEOPIXEL_COUNT
        start = time.time()
        while time.time() - start < 5:
            for i in range(SystemSettings.NEOPIXEL_COUNT):
                heat[i] = max(0, heat[i] - random.randint(0, 5))
            for i in range(SystemSettings.NEOPIXEL_COUNT - 1, 2, -1):
                heat[i] = (heat[i - 1] + heat[i - 2] + heat[i - 2]) // 3
            if random.randint(0, 100) < 60:
                spark_pos = random.randint(0, 15)
                heat[spark_pos] = min(255, heat[spark_pos] + random.randint(100, 200))
            for i in range(SystemSettings.NEOPIXEL_COUNT):
                color_index = min(len(fire_colors) - 1, heat[i] // 52)
                brightness = min(1.0, heat[i] / 255.0)
                base = fire_colors[color_index]
                dimmed = tuple(int(c * brightness) for c in base)
                pixels[i] = dimmed
            pixels.show()
            time.sleep(0.04)
        print("  Fire effect complete!")

        # Test 7: Power check
        print("\n[Test 7] Full White Brightness Test")
        current_draw = SystemSettings.NEOPIXEL_COUNT * 0.06 * SystemSettings.NEOPIXEL_BRIGHTNESS
        print(f"  Estimated current draw: ~{current_draw:.1f}A")
        print(f"  (Max possible: {SystemSettings.NEOPIXEL_COUNT * 0.06:.1f}A at 100%)")
        pixels.fill((255, 255, 255))
        pixels.show()
        time.sleep(2)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\nERROR during test: {e}")
        return False
    finally:
        print("\nTurning off LEDs...")
        pixels.fill((0, 0, 0))
        pixels.show()
        print("Done!")


def test_api_patterns():
    """Test patterns via API (requires server running)"""
    import requests

    print("\n" + "=" * 60)
    print("API Pattern Test (requires server on port 8000)")
    print("=" * 60)

    base_url = "http://localhost:8000"
    patterns = ["gradient_flow", "chase", "fire", "rainbow", "off"]

    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        try:
            response = requests.post(
                f"{base_url}/leds/mode",
                json={"mode": pattern},
                timeout=5
            )
            if response.status_code == 200:
                print(f"  {pattern}: OK")
                time.sleep(3)
            else:
                print(f"  {pattern}: Failed (status {response.status_code})")
        except Exception as e:
            print(f"  {pattern}: Error - {e}")

    # Turn off at end
    try:
        requests.post(f"{base_url}/leds/mode", json={"mode": "off"}, timeout=5)
    except:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test 165 LED NeoPixel strip")
    parser.add_argument("--api", action="store_true", help="Test via API (requires server)")
    args = parser.parse_args()

    if args.api:
        test_api_patterns()
    else:
        success = test_led_strip()
        sys.exit(0 if success else 1)
