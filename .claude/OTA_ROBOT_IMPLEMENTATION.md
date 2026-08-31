# OTA — Robot Slice Implementation (2026-08-31)

Robot side of `OTA_UPDATE_CONTRACT_2026-08-07.md` (app repo). App slice
shipped in app Build 154; its event shapes are frozen and implemented here
verbatim.

## Layout (after bootstrap)

```
/home/morgan/wimz/
├── releases/<version>/      code (git clone at bootstrap; tarballs later)
├── shared/                  per-unit data: data/ VOICEMP3/ env_new/ logs/
│                            state/ captures/ photos/ recordings/ .env
│                            claude-local/{resume_chat.md,settings.local.json}
├── current -> releases/<v>  flipped atomically by the updater
├── updater/wimz_updater.py  stdlib-only; NEVER updated by OTA itself
├── update-request.json      main app writes; systemd path unit triggers
├── update-status.json       updater writes; main app forwards to relay
└── update-result.json       terminal state; consumed+emitted after restart
/home/morgan/dogbot -> /home/morgan/wimz/current
```

The `/home/morgan/dogbot` symlink is the whole trick: every hardcoded path
in code and in `treatbot.service` keeps working, per-unit data survives
release swaps via the `shared/` symlinks, and `git pull` dev still works
inside the current release. `/data` (wimz.db) was already outside the tree.

## Pieces

| piece | where |
|---|---|
| `VERSION` file (e.g. `2026.08.1`) | release root; read by `core/version.py` |
| `sw_version` reporting | relay `status` telemetry frame, `/health`, `/telemetry` |
| `start_update` handler + gates | `services/system/ota_manager.py` (wired in `main_treatbot.py`) |
| updater | `scripts/ota/wimz_updater.py` → installed at `wimz/updater/` |
| systemd trigger | `scripts/systemd/wimz-updater.{path,service}` (root, oneshot) |
| bootstrap (once per unit) | `sudo scripts/ota/bootstrap-ota-layout.sh` |

## Flow

1. App sends `start_update {data:{version}}`. Gates: layout bootstrapped,
   no update in flight, mode == idle, battery ≥ 30% and not critical, no
   active WebRTC connection. Refusal ⇒ `update_status {state:"failed",
   error:<reason>}` (the app has no separate refusal channel).
2. Pass ⇒ emit `checking`, write `update-request.json`; `wimz-updater.path`
   fires the root oneshot service.
3. Updater: consume request → manifest `GET /api/releases/<v>` → download
   `GET /api/releases/<v>/download` (`.part`, never promoted unverified;
   interrupted downloads fail clean) → sha256 vs manifest → unpack →
   symlink shared data in → `pip install -r requirements.txt` into the
   SHARED venv (skipped when requirements.txt sha unchanged) → flip
   `current` → `systemctl restart treatbot` → poll `/health` ≤150 s → on
   failure flip back + restart + `rolled_back`. Keeps 2 prior releases.
4. Progress: updater rewrites `update-status.json` per state; the main
   app's OTA watcher forwards each change as an `update_status` relay
   event (`checking|downloading(+progress_pct)|verifying|installing|
   restarting`). Terminal `success|failed|rolled_back` goes via
   `update-result.json`, consumed by whichever process boots after the
   restart — that is how the NEW code emits the terminal event.

## Auth (relay must implement to match)

`GET /api/releases/<version>` (JSON manifest: `{version, sha256, size}`)
and `GET /api/releases/<version>/download` (tar.gz of the repo tree, flat
or single top dir) authenticated with the same HMAC as the WS leg:
`X-Device-Id`, `X-Timestamp` (unix seconds), `X-Signature` =
HMAC-SHA256(device_secret, "{device_id}:{timestamp}") hex.

## Trade-offs on record

- Shared venv: rollback restores code, not pip state (reqs rarely change;
  pip runs BEFORE the symlink flip so a pip failure never breaks current).
- Release strings: date-tags (`2026.08.1`), plain string compare, never
  reused. A running process reports the version it BOOTED with.
- Updater self-update is deliberate/manual only (bootstrap or by hand).
