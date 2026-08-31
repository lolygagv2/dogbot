#!/usr/bin/env bash
# One-time OTA bootstrap for a WIM-Z unit (OTA contract 2026-08-07).
#
# Converts the flat /home/morgan/dogbot checkout into the versioned layout:
#
#   /home/morgan/wimz/releases/<VERSION>/   the current checkout, moved
#   /home/morgan/wimz/shared/               per-unit data, symlinked into
#                                           every release
#   /home/morgan/wimz/current -> releases/<VERSION>
#   /home/morgan/wimz/updater/              stdlib-only wimz_updater.py
#   /home/morgan/dogbot -> /home/morgan/wimz/current
#
# Every hardcoded /home/morgan/dogbot/... path (code AND systemd unit) keeps
# working through the final symlink. Run with sudo. Idempotent guard: refuses
# to run twice. Stops treatbot.service for the duration (~seconds).
#
#   sudo /home/morgan/dogbot/scripts/ota/bootstrap-ota-layout.sh
set -euo pipefail

DOGBOT=/home/morgan/dogbot
WIMZ=/home/morgan/wimz
OWNER=morgan:morgan

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo." >&2; exit 1
fi
if [ -L "$DOGBOT" ]; then
    echo "ERROR: $DOGBOT is already a symlink — bootstrap already done." >&2; exit 1
fi
if [ ! -f "$DOGBOT/VERSION" ]; then
    echo "ERROR: $DOGBOT/VERSION missing — git pull the OTA release first." >&2; exit 1
fi
VERSION=$(tr -d '[:space:]' < "$DOGBOT/VERSION")
if [ -z "$VERSION" ]; then
    echo "ERROR: VERSION file is empty." >&2; exit 1
fi
RELEASE="$WIMZ/releases/$VERSION"
if [ -e "$RELEASE" ]; then
    echo "ERROR: $RELEASE already exists." >&2; exit 1
fi

# Per-unit data moved to shared/ and symlinked back into the release.
# MUST match SHARED_LINKS in wimz_updater.py.
SHARED_ITEMS=(data VOICEMP3 env_new logs state captures photos recordings .env)

echo "== WIM-Z OTA bootstrap: $VERSION on $(hostname) =="

echo "[1/8] stopping treatbot.service"
systemctl stop treatbot.service

echo "[2/8] creating $WIMZ layout"
mkdir -p "$WIMZ/releases" "$WIMZ/shared/claude-local" "$WIMZ/updater"

echo "[3/8] moving checkout -> $RELEASE"
mv "$DOGBOT" "$RELEASE"

echo "[4/8] relocating per-unit data -> $WIMZ/shared/"
for item in "${SHARED_ITEMS[@]}"; do
    src="$RELEASE/$item"
    dst="$WIMZ/shared/$item"
    if [ -e "$src" ] && [ ! -L "$src" ]; then
        mv "$src" "$dst"
    elif [ ! -e "$dst" ]; then
        # Missing on this unit (e.g. recordings/) — create an empty dir so the
        # symlink target exists. .env must genuinely exist though.
        if [ "$item" = ".env" ]; then
            echo "ERROR: $src missing — relay credentials required." >&2; exit 1
        fi
        mkdir -p "$dst"
    fi
    ln -s "$dst" "$RELEASE/$item"
done
# Per-unit Claude session files (untracked, would be lost on release swap)
for cfile in resume_chat.md settings.local.json; do
    src="$RELEASE/.claude/$cfile"
    dst="$WIMZ/shared/claude-local/$cfile"
    if [ -e "$src" ] && [ ! -L "$src" ]; then
        mv "$src" "$dst"
        ln -s "$dst" "$src"
    fi
done

echo "[5/8] current + dogbot symlinks"
ln -sfn "$RELEASE" "$WIMZ/current"
ln -sfn "$WIMZ/current" "$DOGBOT"

echo "[6/8] installing wimz-updater (outside the release tree)"
install -m755 "$RELEASE/scripts/ota/wimz_updater.py" "$WIMZ/updater/wimz_updater.py"
install -m644 "$RELEASE/scripts/systemd/wimz-updater.service" /etc/systemd/system/
install -m644 "$RELEASE/scripts/systemd/wimz-updater.path" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now wimz-updater.path

echo "[7/8] ownership"
chown -h "$OWNER" "$DOGBOT" "$WIMZ/current"
chown -R "$OWNER" "$WIMZ"

echo "[8/8] starting treatbot.service"
systemctl start treatbot.service

echo
echo "Done. Verify:"
echo "  readlink -f $DOGBOT          # -> $RELEASE"
echo "  systemctl status treatbot.service --no-pager | head -5"
echo "  curl -s localhost:8000/health | grep sw_version"
echo "  systemctl list-unit-files | grep wimz-updater"
echo
echo "Untracked leftovers in the release dir (survive until the next release"
echo "swap only — relocate anything precious to $WIMZ/shared/):"
cd "$RELEASE" && sudo -u morgan git status --porcelain 2>/dev/null | grep '^??' || true
