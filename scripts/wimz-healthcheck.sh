#!/usr/bin/env bash
# WIM-Z liveness healthcheck (Layer B of auto-recovery).
#
# Restarts treatbot.service when the app's HTTP /health endpoint stops
# responding — i.e. the Python process is hung/deadlocked but the OS is
# still alive. Full-board lockups (no-SSH, "needs physical power cycle")
# are NOT this script's job; the systemd hardware watchdog
# (RuntimeWatchdogSec, see scripts/systemd/10-wimz-watchdog.conf) covers
# those by resetting the SoC.
#
# Unit-agnostic: identical on every robot. Driven by wimz-healthcheck.timer.
set -u

URL="http://127.0.0.1:8000/health"
SERVICE="treatbot.service"
CURL_TIMEOUT=10          # seconds to wait for a health reply before calling it dead
FAIL_THRESHOLD=3         # consecutive failures before we restart (avoids flapping)
MIN_ACTIVE_SECS=120      # don't judge a service that just (re)started
STATE_FILE="/run/wimz-healthcheck.fails"

log() { logger -t wimz-healthcheck "$*"; echo "$*"; }

# Only police the service when systemd thinks it should be up. If it's
# already restarting or intentionally stopped, stay out of the way.
if [ "$(systemctl is-active "$SERVICE" 2>/dev/null || true)" != "active" ]; then
    rm -f "$STATE_FILE"
    exit 0
fi

# Give a freshly (re)started service time to bind port 8000 before judging it.
# Work in whole seconds: systemd's stamp is usec-since-boot and uptime is
# seconds-since-boot. (Don't multiply uptime to usec — awk's %d is 32-bit and
# overflows past ~35min of uptime; bash arithmetic below is 64-bit, so divide
# the usec stamp down instead.)
now_sec=$(awk '{printf "%d", $1}' /proc/uptime)
active_usec=$(systemctl show -p ActiveEnterTimestampMonotonic --value "$SERVICE" 2>/dev/null || echo 0)
if [ "${active_usec:-0}" -gt 0 ]; then
    if [ $(( now_sec - active_usec / 1000000 )) -lt "$MIN_ACTIVE_SECS" ]; then
        rm -f "$STATE_FILE"
        exit 0
    fi
fi

# Healthy: clear the counter and we're done.
if curl -fsS -m "$CURL_TIMEOUT" -o /dev/null "$URL"; then
    rm -f "$STATE_FILE"
    exit 0
fi

# Unhealthy: bump the consecutive-failure counter.
fails=0
[ -f "$STATE_FILE" ] && fails=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
fails=$((fails + 1))
echo "$fails" > "$STATE_FILE"
log "health check FAILED ($fails/$FAIL_THRESHOLD) — $URL unresponsive"

if [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
    log "restarting $SERVICE after $fails consecutive failures (app appears hung)"
    systemctl restart "$SERVICE"
    rm -f "$STATE_FILE"
fi
exit 0
