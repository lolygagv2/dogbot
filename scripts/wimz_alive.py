#!/usr/bin/env python3
"""Early-boot sign of life (R-BOOT / work order §3A).

Runs from wimz-alive.service seconds after kernel boot, long before the full
stack: breathes the NeoPixel ring a few cycles, latches a dim glow, and exits
(releasing /dev/spidev0.0 for treatbot's LedService, which takes over later).
WS2812 pixels hold their last frame after we close the SPI device, so the glow
persists until the main stack repaints.

Deliberately minimal: no event bus, no config-driven services, no audio (a
pre-PipeWire ALSA grab can lock PipeWire out of the USB card — chime stays at
app start). Hardware init happens only inside main(), never at import time.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BREATHE_SECONDS = 4.0   # a few visible cycles, then get out of the way
BREATHE_PERIOD = 2.0    # one inhale/exhale
GLOW = (0, 60, 120)     # dim WIM-Z blue, latched on exit
BRIGHTNESS = 0.1        # low: 165 LEDs on boot power budget


def wait_for_spi(timeout: float = 5.0) -> bool:
    """SPI device node can appear slightly after early units start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists('/dev/spidev0.0'):
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    if not wait_for_spi():
        print("wimz-alive: /dev/spidev0.0 never appeared, skipping")
        return 0  # sign-of-life is best-effort; never block boot

    from core.hardware.led_controller import NeoPixelSPI
    from config.settings import SystemSettings

    pixels = NeoPixelSPI(SystemSettings.NEOPIXEL_COUNT, brightness=BRIGHTNESS)
    try:
        start = time.monotonic()
        while time.monotonic() - start < BREATHE_SECONDS:
            phase = ((time.monotonic() - start) % BREATHE_PERIOD) / BREATHE_PERIOD
            # triangle wave 0..1..0
            level = 2 * phase if phase < 0.5 else 2 * (1 - phase)
            pixels.fill(tuple(int(c * (0.15 + 0.85 * level)) for c in GLOW))
            pixels.show()
            time.sleep(0.05)
        # Latch a steady dim glow; WS2812 holds it after SPI closes
        pixels.fill(GLOW)
        pixels.show()
    finally:
        pixels.deinit()
    print("wimz-alive: sign of life shown, SPI released")
    return 0


if __name__ == '__main__':
    sys.exit(main())
