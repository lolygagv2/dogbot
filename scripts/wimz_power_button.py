#!/usr/bin/env python3
"""WIMZ power button watcher.

Watches GPIO20 for a momentary press and triggers a graceful shutdown.
Power-off latch (GPIO26 pulse) is handled by wimz-poweroff-pulse.service
during the shutdown sequence, not here.
"""

import logging
import subprocess
import sys

from gpiozero import Button

BUTTON_GPIO = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - wimz_power_button - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wimz_power_button")


def main() -> int:
    button = Button(BUTTON_GPIO, pull_up=True, bounce_time=0.05)
    log.info("Power button watcher armed on GPIO%d (pull-up, falling edge)", BUTTON_GPIO)

    while True:
        button.wait_for_press()
        log.warning("Power button pressed - initiating graceful shutdown")
        subprocess.call(["sudo", "shutdown", "-h", "now"])
        # After issuing shutdown, wait for the system to go down.
        # If shutdown fails for any reason, fall back to looping so
        # a subsequent press can retry.
        button.wait_for_release()


if __name__ == "__main__":
    sys.exit(main())
