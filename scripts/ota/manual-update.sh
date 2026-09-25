#!/usr/bin/env bash
# Manual (git) update of a unit that is ALREADY on the OTA layout, without
# breaking OTA. Use when the fleet is on site and a release cut isn't worth
# it. Idempotent — safe to rerun.
#
#   cd /home/morgan/dogbot && git pull --ff-only \
#     && sudo bash scripts/ota/manual-update.sh
#
# What it does:
#   1. git pull --ff-only in the current release (works: .git -> shared/dotgit)
#   2. reinstall the power-button watcher (/usr/local/bin, NOT OTA-managed),
#      preserving this unit's RELAY_ACTIVE_HIGH line
#   3. refresh the updater copy (wimz/updater is outside the release tree)
#   4. restart wimz-power-button.service and treatbot.service
#   5. verify the OTA layout is still intact (symlinks, updater, path unit,
#      VERSION, health sw_version)
#
# The VERSION file is NOT bumped by a manual pull: the unit keeps reporting
# the release it was cut from until the next real OTA release, which will
# download into releases/<new>/ and flip normally.
#
#   sudo bash scripts/ota/manual-update.sh --verify-only   # checks only
set -uo pipefail

DOGBOT=/home/morgan/dogbot
WIMZ=/home/morgan/wimz
WATCHER_SRC=services/power/wimz_power_button.py
WATCHER_DST=/usr/local/bin/wimz_power_button.py
VERIFY_ONLY=0
[ "${1:-}" = "--verify-only" ] && VERIFY_ONLY=1

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo." >&2; exit 1
fi
if [ ! -L "$DOGBOT" ]; then
    echo "ERROR: $DOGBOT is not a symlink — this unit is NOT on the OTA layout." >&2
    echo "       Run scripts/ota/bootstrap-ota-layout.sh first." >&2; exit 1
fi
RELEASE=$(readlink -f "$WIMZ/current")
cd "$RELEASE"

fail=0
ok()   { echo "  OK   $*"; }
bad()  { echo "  FAIL $*"; fail=1; }

if [ "$VERIFY_ONLY" -eq 0 ]; then
    echo "== manual update of $(hostname) in $RELEASE =="

    echo "[1/4] git pull --ff-only"
    if ! sudo -u morgan git pull --ff-only; then
        echo "ERROR: git pull failed (local changes or diverged history?). Nothing restarted." >&2
        sudo -u morgan git status --short | head; exit 1
    fi

    echo "[2/4] power-button watcher -> $WATCHER_DST"
    if [ -f "$WATCHER_DST" ] && grep -q "^RELAY_ACTIVE_HIGH = True" "$WATCHER_DST"; then
        sed 's/^RELAY_ACTIVE_HIGH = False/RELAY_ACTIVE_HIGH = True/' "$WATCHER_SRC" > "$WATCHER_DST.new"
        install -m755 "$WATCHER_DST.new" "$WATCHER_DST"; rm -f "$WATCHER_DST.new"
        echo "  (kept RELAY_ACTIVE_HIGH = True for this unit)"
    else
        install -m755 "$WATCHER_SRC" "$WATCHER_DST"
    fi

    echo "[3/4] updater copy -> $WIMZ/updater/"
    install -m755 scripts/ota/wimz_updater.py "$WIMZ/updater/wimz_updater.py"

    echo "[4/4] restarting services"
    systemctl restart wimz-power-button.service
    systemctl restart treatbot.service
    for _ in $(seq 1 40); do
        sleep 3
        curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    done
fi

echo "== OTA layout verification on $(hostname) =="
[ "$(readlink "$DOGBOT")" = "$WIMZ/current" ] && ok "$DOGBOT -> wimz/current" || bad "$DOGBOT symlink wrong: $(readlink "$DOGBOT")"
[ -d "$RELEASE" ] && ok "current -> $RELEASE" || bad "wimz/current dangling"
[ "$(readlink "$RELEASE/.git" 2>/dev/null)" = "$WIMZ/shared/dotgit" ] && ok ".git -> shared/dotgit" || bad ".git is not the shared/dotgit symlink"
for item in data VOICEMP3/talks VOICEMP3/songs env_new logs state .env; do
    [ -L "$RELEASE/$item" ] && ok "shared link $item" || bad "$item is not a symlink into shared/"
done
[ -f "$WIMZ/updater/wimz_updater.py" ] && ok "updater present" || bad "wimz/updater/wimz_updater.py missing"
cmp -s "$WIMZ/updater/wimz_updater.py" "$RELEASE/scripts/ota/wimz_updater.py" && ok "updater matches release copy" || bad "updater differs from scripts/ota/wimz_updater.py"
[ "$(systemctl is-enabled wimz-updater.path 2>/dev/null)" = "enabled" ] && ok "wimz-updater.path enabled" || bad "wimz-updater.path not enabled"
[ "$(systemctl is-active wimz-updater.path 2>/dev/null)" = "active" ] && ok "wimz-updater.path active" || bad "wimz-updater.path not active"
ver=$(tr -d '[:space:]' < "$RELEASE/VERSION" 2>/dev/null)
[ -n "$ver" ] && ok "VERSION $ver" || bad "VERSION missing/empty"
[ "$(systemctl is-active wimz-power-button.service)" = "active" ] && ok "wimz-power-button.service active" || bad "wimz-power-button.service not active"
grep -q "network cancel" "$WATCHER_DST" 2>/dev/null && ok "watcher has long-press network cancel" || bad "installed watcher is the old version"
health=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
sw=$(printf '%s' "$health" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("sw_version",""))' 2>/dev/null)
[ "$sw" = "$ver" ] && ok "treatbot healthy, sw_version $sw" || bad "treatbot health/sw_version mismatch (got '$sw', VERSION '$ver')"
head=$(sudo -u morgan git rev-parse --short HEAD 2>/dev/null)
echo "  git HEAD $head — $(sudo -u morgan git log -1 --format=%s 2>/dev/null)"

if [ "$fail" -eq 0 ]; then
    echo "== ALL OK — OTA intact, unit updated =="
else
    echo "== PROBLEMS FOUND — see FAIL lines above ==" >&2; exit 1
fi
