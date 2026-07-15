#!/bin/bash
# Live monitor for the treat-chute IR through-beam receiver (GPIO7).
# Prints a timestamped line on every transition. Ctrl+C to stop.
# Expected: hi = beam locked, lo = beam broken/no IR (beam_active_low: true).
echo "Watching GPIO7 (hi = locked, lo = broken). Ctrl+C to stop."
p=""
while true; do
  s=$(pinctrl get 7 | grep -oE '(hi|lo)')
  if [ "$s" != "$p" ]; then
    echo "$(date +%T.%3N)  GPIO7 = $s"
  fi
  p=$s
  sleep 0.02
done
