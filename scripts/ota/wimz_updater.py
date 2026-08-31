#!/usr/bin/env python3
"""wimz-updater — OTA release installer for WIM-Z robots.

DELIBERATELY TINY AND STDLIB-ONLY. This file must never import anything
from the main dogbot application: a broken main app must never break
updatability. It is installed OUTSIDE the release tree (at
/home/morgan/wimz/updater/wimz_updater.py by the bootstrap script) and is
only ever replaced deliberately, never by an OTA update.

Layout it owns (OTA contract 2026-08-07, robot slice):

    /home/morgan/wimz/
    ├── releases/<version>/      code (tarball unpack; per-unit data is
    │                            symlinked in, see SHARED_LINKS)
    ├── shared/                  per-unit data + the shared venv (env_new)
    ├── current -> releases/<v>  flipped atomically by this script
    ├── updater/                 this script + nothing else
    ├── update-request.json      written by the main app (trigger)
    ├── update-status.json       written by this script (progress)
    └── update-result.json       terminal state; consumed by the main app
    /home/morgan/dogbot -> /home/morgan/wimz/current   (path stability)

Trigger: systemd path unit (wimz-updater.path) watches update-request.json
and starts wimz-updater.service (oneshot, root) which runs this script.

Flow: consume request -> fetch manifest -> download (resumable-safe: fail
clean, .part never promoted unverified) -> sha256 verify -> unpack ->
symlink shared data in -> pip install pinned reqs into the SHARED venv
(skipped when requirements.txt is unchanged) -> flip `current` -> restart
treatbot.service -> health check (<= HEALTH_TIMEOUT) -> on failure flip
back + restart + report rolled_back. Keeps KEEP_RELEASES previous releases.

Progress protocol: update-status.json is atomically rewritten on every
state change; the main app tails it and forwards `update_status` events to
the relay. Terminal states (success/failed/rolled_back) additionally land
in update-result.json, which the (new or rolled-back) main app consumes on
startup and emits as the terminal relay event — this survives the process
death at `restarting`.

Auth to relay (https://api.wimzai.com): same HMAC scheme as the WS leg —
X-Device-Id, X-Timestamp, X-Signature = HMAC-SHA256(device_secret,
"{device_id}:{timestamp}") hex. Credentials come from shared/.env.
"""

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

WIMZ = '/home/morgan/wimz'
RELEASES = os.path.join(WIMZ, 'releases')
SHARED = os.path.join(WIMZ, 'shared')
CURRENT = os.path.join(WIMZ, 'current')
REQUEST = os.path.join(WIMZ, 'update-request.json')
STATUS = os.path.join(WIMZ, 'update-status.json')
RESULT = os.path.join(WIMZ, 'update-result.json')
LOCK = os.path.join(WIMZ, 'updater.lock')
ENV_FILE = os.path.join(SHARED, '.env')

RELAY_BASE = os.environ.get('WIMZ_RELAY_BASE', 'https://api.wimzai.com')
SERVICE = 'treatbot.service'
HEALTH_URL = 'http://localhost:8000/health'
HEALTH_TIMEOUT = 150   # contract says 120s; +30s covers the unit's cold
                       # ExecStartPre pulse-socket wait worst case
KEEP_RELEASES = 2      # previous releases kept for rollback (beyond current)
DOWNLOAD_TIMEOUT = 600
PIP_TIMEOUT = 1800
OWNER_UID_GID = 'morgan:morgan'

log = lambda msg: print(f"[wimz-updater] {msg}", flush=True)


def atomic_write(path: str, data: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _chown_morgan(path)


def _chown_morgan(path: str):
    # Files this root process writes must stay readable/consumable by the
    # main app (runs as morgan).
    try:
        subprocess.run(['chown', OWNER_UID_GID, path], check=False)
    except Exception:
        pass


def set_status(state: str, version: str, progress_pct=None, error=None):
    payload = {'state': state, 'version': version, 'ts': time.time()}
    if progress_pct is not None:
        payload['progress_pct'] = int(progress_pct)
    if error:
        payload['error'] = str(error)
    atomic_write(STATUS, payload)
    log(f"status: {state}" + (f" {progress_pct}%" if progress_pct is not None else '')
        + (f" error={error}" if error else ''))


def set_result(state: str, version: str, error=None):
    payload = {'state': state, 'version': version, 'ts': time.time()}
    if error:
        payload['error'] = str(error)
    atomic_write(RESULT, payload)
    set_status(state, version, error=error)


def read_env_credentials():
    device_id = device_secret = None
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('DEVICE_ID='):
                device_id = line.split('=', 1)[1].strip().strip('"\'')
            elif line.startswith('DEVICE_SECRET='):
                device_secret = line.split('=', 1)[1].strip().strip('"\'')
    if not device_id or not device_secret:
        raise RuntimeError(f"DEVICE_ID/DEVICE_SECRET missing in {ENV_FILE}")
    return device_id, device_secret


def auth_headers(device_id: str, device_secret: str) -> dict:
    ts = str(int(time.time()))
    sig = hmac.new(device_secret.encode(), f"{device_id}:{ts}".encode(),
                   hashlib.sha256).hexdigest()
    return {'X-Device-Id': device_id, 'X-Timestamp': ts, 'X-Signature': sig}


def http_get(url: str, headers: dict, timeout: int = 30):
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_manifest(version: str, headers: dict) -> dict:
    url = f"{RELAY_BASE}/api/releases/{version}"
    with http_get(url, headers) as resp:
        manifest = json.loads(resp.read().decode())
    if not manifest.get('sha256'):
        raise RuntimeError(f"manifest for {version} has no sha256")
    return manifest


def download_release(version: str, headers: dict, expected_size, cb) -> str:
    url = f"{RELAY_BASE}/api/releases/{version}/download"
    part = os.path.join(RELEASES, f".{version}.tar.gz.part")
    final = os.path.join(RELEASES, f".{version}.tar.gz")
    if os.path.exists(part):
        os.remove(part)  # never resume a possibly-corrupt partial
    done = 0
    with http_get(url, headers, timeout=DOWNLOAD_TIMEOUT) as resp, \
            open(part, 'wb') as out:
        total = int(resp.headers.get('Content-Length') or expected_size or 0)
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                cb(min(99, int(done * 100 / total)))
        out.flush()
        os.fsync(out.fileno())
    if expected_size and done != int(expected_size):
        os.remove(part)
        raise RuntimeError(f"download size mismatch: got {done}, expected {expected_size}")
    os.replace(part, final)
    return final


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack_release(tarball: str, version: str) -> str:
    dest = os.path.join(RELEASES, version)
    tmp = dest + '.unpack'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    if os.path.exists(dest):
        shutil.rmtree(dest)  # re-install of same version: replace cleanly
    os.makedirs(tmp)
    with tarfile.open(tarball, 'r:gz') as tar:
        # Refuse path traversal; tolerate both flat and single-top-dir layouts
        members = tar.getmembers()
        for m in members:
            if m.name.startswith('/') or '..' in m.name.split('/'):
                raise RuntimeError(f"unsafe path in tarball: {m.name}")
        tar.extractall(tmp)
    entries = os.listdir(tmp)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])) \
            and not os.path.exists(os.path.join(tmp, 'main_treatbot.py')):
        os.replace(os.path.join(tmp, entries[0]), dest)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        os.replace(tmp, dest)
    if not os.path.exists(os.path.join(dest, 'main_treatbot.py')):
        raise RuntimeError("unpacked release has no main_treatbot.py — bad artifact")
    os.remove(tarball)
    subprocess.run(['chown', '-R', OWNER_UID_GID, dest], check=False)
    return dest


# Per-unit data symlinked into every release. Must match the bootstrap script.
# VOICEMP3 itself stays a REAL directory in the release (VOICEMP3/wimz/ system
# prompts are git-tracked and ship with every release); only the per-unit
# talks/ and songs/ subdirs are shared.
SHARED_LINKS = ['data', 'VOICEMP3/talks', 'VOICEMP3/songs', 'env_new',
                'logs', 'state', 'captures', 'photos', 'recordings', '.env']


def link_shared(release_dir: str):
    for name in SHARED_LINKS:
        target = os.path.join(SHARED, name)  # shared keeps the literal names
        link = os.path.join(release_dir, name)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            os.remove(link)
        elif os.path.isdir(link):
            shutil.rmtree(link)   # tracked skeleton dirs from the tarball lose
        elif os.path.exists(link):
            os.remove(link)       # to the shared per-unit data
        os.symlink(target, link)
    # Per-unit Claude session files live in shared/claude-local
    for cname in ('resume_chat.md', 'settings.local.json'):
        src = os.path.join(SHARED, 'claude-local', cname)
        link = os.path.join(release_dir, '.claude', cname)
        if os.path.exists(src) and os.path.isdir(os.path.dirname(link)):
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(src, link)


def pip_install(release_dir: str, version: str):
    reqs = os.path.join(release_dir, 'requirements.txt')
    if not os.path.exists(reqs):
        log("no requirements.txt in release — skipping pip")
        return
    marker = os.path.join(SHARED, '.requirements.sha256')
    reqs_sha = sha256_file(reqs)
    if os.path.exists(marker) and open(marker).read().strip() == reqs_sha:
        log("requirements unchanged — skipping pip")
        return
    pip = os.path.join(SHARED, 'env_new', 'bin', 'pip')
    log("pip install -r requirements.txt (shared venv)...")
    proc = subprocess.run([pip, 'install', '--no-input', '-r', reqs],
                          capture_output=True, text=True, timeout=PIP_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"pip install failed: {proc.stderr[-400:]}")
    with open(marker, 'w') as f:
        f.write(reqs_sha)
    _chown_morgan(marker)


def flip_current(release_dir: str):
    tmp = CURRENT + '.tmp'
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(release_dir, tmp)
    os.replace(tmp, CURRENT)
    log(f"current -> {release_dir}")


def restart_service():
    subprocess.run(['systemctl', 'restart', SERVICE], check=True, timeout=90)


def wait_healthy(timeout: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        active = subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE])
        if active.returncode != 0:
            continue
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                body = json.loads(resp.read().decode())
                if body.get('status') == 'healthy':
                    return True
        except Exception:
            continue
    return False


def prune_releases(current_target: str):
    try:
        dirs = [os.path.join(RELEASES, d) for d in os.listdir(RELEASES)
                if os.path.isdir(os.path.join(RELEASES, d))]
        keep = {os.path.realpath(current_target)}
        others = sorted((d for d in dirs if os.path.realpath(d) not in keep),
                        key=os.path.getmtime, reverse=True)
        for old in others[KEEP_RELEASES:]:
            log(f"pruning old release {old}")
            shutil.rmtree(old, ignore_errors=True)
    except Exception as e:
        log(f"prune failed (non-fatal): {e}")


def main() -> int:
    # Single-instance lock
    import fcntl
    lock_fh = open(LOCK, 'w')
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another updater instance is running — exiting")
        return 0

    if not os.path.exists(REQUEST):
        log("no update request — exiting")
        return 0
    try:
        with open(REQUEST) as f:
            request = json.load(f)
    except Exception as e:
        log(f"bad request file ({e}) — discarding")
        os.remove(REQUEST)
        return 0
    os.remove(REQUEST)  # consume: a crash mid-run must not loop the path unit

    version = str(request.get('version') or '').strip()
    if not version or '/' in version or version.startswith('.'):
        set_result('failed', version or 'unknown', error='invalid version string')
        return 1

    prev_target = os.path.realpath(CURRENT) if os.path.lexists(CURRENT) else None
    log(f"updating to {version} (current: {prev_target})")

    try:
        set_status('checking', version)
        device_id, device_secret = read_env_credentials()
        headers = auth_headers(device_id, device_secret)
        manifest = fetch_manifest(version, headers)

        set_status('downloading', version, progress_pct=0)
        tarball = download_release(
            version, auth_headers(device_id, device_secret),
            manifest.get('size'),
            lambda pct: set_status('downloading', version, progress_pct=pct))

        set_status('verifying', version)
        actual = sha256_file(tarball)
        if actual != manifest['sha256']:
            os.remove(tarball)
            raise RuntimeError(f"sha256 mismatch: {actual[:12]} != {manifest['sha256'][:12]}")

        set_status('installing', version)
        release_dir = unpack_release(tarball, version)
        link_shared(release_dir)
        pip_install(release_dir, version)
    except Exception as e:
        # Nothing was flipped yet — fail clean, current release untouched
        log(f"FAILED before switch: {e}")
        set_result('failed', version, error=str(e))
        return 1

    # Point of no return: flip + restart + health check
    try:
        set_status('restarting', version)
        flip_current(release_dir)
        restart_service()
        if wait_healthy():
            set_result('success', version)
            log("update healthy")
            prune_releases(release_dir)
            return 0
        raise RuntimeError(f"health check failed within {HEALTH_TIMEOUT}s")
    except Exception as e:
        log(f"post-switch failure: {e} — rolling back")
        try:
            if prev_target and os.path.isdir(prev_target):
                flip_current(prev_target)
                restart_service()
                healthy = wait_healthy()
                set_result('rolled_back', version,
                           error=f"{e} (rollback {'healthy' if healthy else 'UNHEALTHY'})")
            else:
                set_result('failed', version,
                           error=f"{e} (no previous release to roll back to)")
        except Exception as rb_err:
            set_result('failed', version, error=f"{e}; rollback also failed: {rb_err}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
