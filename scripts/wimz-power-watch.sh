#!/usr/bin/env bash
# WIM-Z power watch — freeze RCA evidence collector.
#
# The recurring silent hard-freeze (no-SSH, "needs physical power cycle")
# leaves NO kernel/software trace: the journal is corrupt/empty after the
# event, so we can't tell a power brownout apart from a true SoC/software
# hang. This daemon samples power-delivery state to a plain file on the SD
# card and fsyncs each line, so the LAST sample before a lock survives the
# power cycle. Read logs/power_watch.csv after the next freeze:
#   - EXT5V_V sagging (< ~4.7V) or throttled != 0x0 just before the gap
#     => brownout / power delivery  (hardware fix: supply, wiring, battery)
#   - rails healthy right up to the gap
#     => true hang                  (software / watchdog fix)
# A reboot boundary with no clean "STOP" line before it is itself a
# freeze signature (unclean shutdown).
#
# Unit-agnostic: identical on every robot. Driven by wimz-power-watch.service.
set -u

INTERVAL="${WIMZ_POWER_WATCH_INTERVAL:-30}"          # seconds between samples
LOG="${WIMZ_POWER_WATCH_LOG:-/home/morgan/dogbot/logs/power_watch.csv}"
MAX_LINES="${WIMZ_POWER_WATCH_MAX_LINES:-45000}"     # ring-buffer cap (~15 days @30s)
KEEP_LINES="${WIMZ_POWER_WATCH_KEEP_LINES:-40000}"   # trimmed-to size when capped
HEADER="epoch,iso_time,uptime_s,throttled,temp_c,core_v,ext5v_v,vdd_core_pmic_v"

mkdir -p "$(dirname "$LOG")"
# Write the header once, on a fresh/empty file.
if [ ! -s "$LOG" ]; then
    echo "$HEADER" > "$LOG"
fi

# vcgencmd field extractor: takes the value AFTER the '=' (so label/index
# digits like the 5 in "EXT5V_V" or the 15 in "volt(15)" aren't mistaken for
# the reading), then keeps the leading number.
#   "temp=59.3'C" -> 59.3   |   "EXT5V_V volt(24)=5.13622000V" -> 5.13622000
val() { echo "$1" | sed 's/.*=//' | grep -oE '[0-9]+\.?[0-9]*' | head -1; }

sample() {
    local now iso up thr temp core ext vcore
    now=$(date +%s)
    iso=$(date --iso-8601=seconds)
    up=$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null)
    # get_throttled is the single most important field — keep the raw hex.
    thr=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
    temp=$(val "$(vcgencmd measure_temp 2>/dev/null)")
    core=$(val "$(vcgencmd measure_volts core 2>/dev/null)")
    ext=$(val "$(vcgencmd pmic_read_adc EXT5V_V 2>/dev/null)")
    vcore=$(val "$(vcgencmd pmic_read_adc VDD_CORE_V 2>/dev/null)")
    echo "${now},${iso},${up:-},${thr:-NA},${temp:-},${core:-},${ext:-},${vcore:-}" >> "$LOG"
    # Force the line to physical media so it survives an abrupt power loss.
    sync "$LOG" 2>/dev/null || sync
}

trim() {
    # Ring-buffer: keep the header + most recent KEEP_LINES data rows.
    local n
    n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
    if [ "$n" -gt "$MAX_LINES" ]; then
        { head -1 "$LOG"; tail -n "$KEEP_LINES" "$LOG"; } > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
        sync "$LOG" 2>/dev/null || sync
    fi
}

# Mark daemon start: an unclean prior shutdown shows up as a START with no
# preceding STOP. Include uptime so a mid-run START (Restart=always) is visible.
echo "# START $(date --iso-8601=seconds) uptime_s=$(awk '{printf "%d", $1}' /proc/uptime)" >> "$LOG"
# Log STOP on graceful termination so clean reboots are distinguishable.
trap 'echo "# STOP $(date --iso-8601=seconds)" >> "$LOG"; sync "$LOG" 2>/dev/null; exit 0' TERM INT

sample                # capture boot/start state immediately
while true; do
    sleep "$INTERVAL"
    sample
    trim
done
