#!/bin/bash
# Pulse GPIO26 HIGH for 500ms to trigger the Pololu #2809 OFF latch
# at the end of the Pi shutdown sequence. This cuts battery power.

set -u

GPIO=26
SYSFS=/sys/class/gpio

# Export (suppress error if already exported)
echo "$GPIO" > "$SYSFS/export" 2>/dev/null || true

# Wait briefly for the gpio sysfs node to appear
for _ in 1 2 3 4 5; do
    [ -d "$SYSFS/gpio$GPIO" ] && break
    sleep 0.05
done

echo out > "$SYSFS/gpio$GPIO/direction"
echo 1   > "$SYSFS/gpio$GPIO/value"
sleep 0.5
echo 0   > "$SYSFS/gpio$GPIO/value"

exit 0
