#!/usr/bin/env bash
# Cut an OTA release artifact from the current git HEAD.
#
# One-command release:
#   WIMZ_ADMIN_JWT=... ./scripts/ota/make-release.sh --bump --upload
#     -> next date-tag (2026.09.1, .2, ...), VERSION commit+push,
#        git archive, sha256, upload to relay. Done.
#
# Piecewise:
#   ./scripts/ota/make-release.sh --bump       # bump+commit VERSION only... then build
#   ./scripts/ota/make-release.sh              # build wimz-<VERSION>.tar.gz + sha256
#   WIMZ_ADMIN_JWT=... ./scripts/ota/make-release.sh --upload
#
# Rules (OTA contract 2026-08-07 / relay API_CONTRACT v1.3):
#   - Versions are immutable on the relay (409 on re-upload) and the app
#     compares by plain string inequality — never reuse a string, and never
#     upload an artifact tagged with a version a robot already reports.
#     --bump enforces this by always generating a fresh YYYY.MM.N tag.
#   - `git archive HEAD` ships only TRACKED files, so per-unit data (data/,
#     VOICEMP3 talks/songs, .env, venv, logs...) is excluded by
#     construction, while models, configs and VOICEMP3/wimz/ ship.
#   - Upload needs an ADMIN JWT (not a device credential — robots can't
#     publish releases): POST /api/releases, multipart.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd -P)"
RELAY_BASE="${WIMZ_RELAY_BASE:-https://api.wimzai.com}"
OUT_DIR="${WIMZ_RELEASE_OUT:-$HOME/wimz-artifacts}"

cd "$REPO"

DO_BUMP=false; DO_UPLOAD=false
for arg in "$@"; do
    case "$arg" in
        --bump)   DO_BUMP=true ;;
        --upload) DO_UPLOAD=true ;;
        *) echo "ERROR: unknown arg '$arg' (use --bump / --upload)" >&2; exit 1 ;;
    esac
done

if $DO_BUMP; then
    # Next date-tag: YYYY.MM.N — N restarts at 1 each month, else increments.
    YM="$(date +%Y.%m)"   # zero-padded month: 2026.09, matching 2026.08.1 style
    CUR="$(tr -d '[:space:]' < VERSION)"
    if [[ "$CUR" == "$YM."* ]]; then
        NEW="$YM.$(( ${CUR##*.} + 1 ))"
    else
        NEW="$YM.1"
    fi
    echo "$NEW" > VERSION
    git add VERSION
    git commit -m "release: $NEW"
    git push origin main
    echo "VERSION bumped: $CUR -> $NEW (committed + pushed)"
fi

VERSION="$(tr -d '[:space:]' < VERSION)"
[ -n "$VERSION" ] || { echo "ERROR: VERSION file empty" >&2; exit 1; }

# Refuse a dirty tree: the artifact is git archive HEAD — uncommitted edits
# would silently NOT ship, which is worse than failing here.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: working tree has uncommitted tracked changes — commit first." >&2
    git status --short --untracked-files=no >&2
    exit 1
fi
# Refuse to package a VERSION that isn't committed at HEAD
if ! git show HEAD:VERSION 2>/dev/null | tr -d '[:space:]' | grep -qx "$VERSION"; then
    echo "ERROR: VERSION ($VERSION) differs from HEAD — commit the bump first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
ARTIFACT="$OUT_DIR/wimz-$VERSION.tar.gz"

echo "Building $ARTIFACT from $(git rev-parse --short HEAD)..."
git archive --format=tar.gz -o "$ARTIFACT" HEAD

SHA256="$(sha256sum "$ARTIFACT" | awk '{print $1}')"
SIZE="$(stat -c%s "$ARTIFACT")"
echo "version: $VERSION"
echo "sha256:  $SHA256"
echo "size:    $SIZE bytes"

if $DO_UPLOAD; then
    : "${WIMZ_ADMIN_JWT:?--upload needs WIMZ_ADMIN_JWT (admin token, not a device secret)}"
    echo "Uploading to $RELAY_BASE/api/releases ..."
    curl -sS -f -X POST "$RELAY_BASE/api/releases" \
        -H "Authorization: Bearer $WIMZ_ADMIN_JWT" \
        -F "version=$VERSION" \
        -F "sha256=$SHA256" \
        -F "file=@$ARTIFACT;type=application/gzip"
    echo
    echo "Uploaded. Verify a robot can see it:"
else
    echo
    echo "To upload (from any machine with the admin JWT):"
    echo "  WIMZ_ADMIN_JWT=<token> $0 --upload"
    echo "or hand-roll the multipart POST per .claude/API_CONTRACT_v1_3.md"
    echo "(relay repo) if field names differ. Then verify:"
fi
echo "  curl -s $RELAY_BASE/api/releases/$VERSION  # (device HMAC) -> manifest"
