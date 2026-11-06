#!/usr/bin/env python3
"""Factory sounds read-only, per-id custom overrides — 2026-11-06.

SoundRegistry discovers ids from factory dirs (wimz/, talks/default/), stores a
user replacement at <override_root>/<group>/<stem>.mp3 and plays it instead of
the factory file while it exists. Reset/reset_all remove overrides only; the
factory files must be byte-identical afterwards. Everything runs against temp
dirs so no real VOICEMP3 content is touched. ffmpeg conversion is not run here;
only the "ffmpeg missing" failure path is exercised via a monkeypatched
subprocess.run.

Run: /home/morgan/dogbot/env_new/bin/python tests/media/test_sound_registry_override.py
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import services.media.sound_registry as sr  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sndreg_"))
    wimz = tmp / "wimz"
    default = tmp / "talks" / "default"
    override = tmp / "talks" / "_custom"
    wimz.mkdir(parents=True)
    default.mkdir(parents=True)

    FAKE_MP3 = b"ID3\x03\x00factory-bytes"
    (wimz / "WimZOnline.mp3").write_bytes(FAKE_MP3 + b"-online")
    (wimz / "BatteryLow.wav").write_bytes(b"RIFFfactory-wav")
    (default / "sit.mp3").write_bytes(FAKE_MP3 + b"-sit")
    (default / "good_dog.mp3").write_bytes(FAKE_MP3 + b"-good")
    (default / "notes.txt").write_bytes(b"ignored")

    factory_hashes = {str(p): sha(p) for p in list(wimz.iterdir()) + list(default.iterdir())}

    reg = sr.SoundRegistry(factory_roots={"wimz": wimz, "default": default},
                           override_root=override)

    # --- list ---
    items = reg.list()
    ids = [i["id"] for i in items]
    check(ids == ["default/good_dog", "default/sit", "wimz/BatteryLow", "wimz/WimZOnline"],
          f"list: ids discovered and sorted ({ids})")
    check(all(not i["has_override"] and i["override_path"] is None for i in items),
          "list: no overrides initially")
    check("default/notes" not in ids, "list: non-audio extension ignored")
    check(reg.resolve("default/sit") == str(default / "sit.mp3"), "resolve: factory path before override")
    check(reg.resolve("nope/x") is None, "resolve: unknown id -> None")
    check(reg.has_override("default/sit") is False, "has_override: false initially")

    # --- replace mp3 ---
    custom = b"ID3\x03\x00custom-sit"
    r = reg.replace("default/sit", custom, "mp3")
    check(r.get("success") is True and r.get("id") == "default/sit", f"replace mp3: success ({r})")
    ov_path = override / "default" / "sit.mp3"
    check(r.get("path") == str(ov_path) and ov_path.is_file(), "replace: file at <override_root>/default/sit.mp3")
    check(ov_path.read_bytes() == custom and r.get("size_bytes") == len(custom), "replace: bytes and size match")
    check(reg.has_override("default/sit") is True, "has_override: true after replace")
    check(reg.resolve("default/sit") == str(ov_path), "resolve: returns override path")
    item = [i for i in reg.list() if i["id"] == "default/sit"][0]
    check(item["has_override"] and item["override_path"] == str(ov_path)
          and item["factory_path"] == str(default / "sit.mp3"), "list: shows override + factory path")
    leftovers = [p for p in (override / "default").iterdir() if p.name != "sit.mp3"]
    check(leftovers == [], f"replace: no temp files left behind ({leftovers})")

    # --- resolve_for_path ---
    check(reg.resolve_for_path(str(default / "sit.mp3")) == str(ov_path),
          "resolve_for_path: factory path -> override")
    check(reg.resolve_for_path(str(default / "good_dog.mp3")) == str(default / "good_dog.mp3"),
          "resolve_for_path: factory path without override unchanged")
    check(reg.resolve_for_path("/some/random/file.mp3") == "/some/random/file.mp3",
          "resolve_for_path: unknown path unchanged")
    check(reg.resolve_for_path("") == "" and reg.resolve_for_path(None) is None,
          "resolve_for_path: never raises on junk input")

    # --- reset ---
    r = reg.reset("default/sit")
    check(r.get("success") is True and r.get("removed") is True, f"reset: removed override ({r})")
    check(not ov_path.exists(), "reset: override file gone")
    check(reg.resolve("default/sit") == str(default / "sit.mp3"), "reset: resolve back to factory")
    r = reg.reset("default/sit")
    check(r.get("success") is True and r.get("removed") is False, "reset: idempotent when no override")
    check(reg.reset("nope/x").get("success") is False, "reset: unknown id fails")

    # --- reset_all ---
    reg.replace("default/sit", custom, "mp3")
    reg.replace("wimz/WimZOnline", b"ID3custom-online", "mp3")
    reg.replace("wimz/BatteryLow", b"ID3custom-batt", "mp3")
    check(sum(i["has_override"] for i in reg.list()) == 3, "reset_all: three overrides present")
    r = reg.reset_all()
    check(r.get("success") is True and r.get("removed") == 3, f"reset_all: removed 3 ({r})")
    check(sum(i["has_override"] for i in reg.list()) == 0, "reset_all: no overrides remain")
    check(all(sha(p) == h for p, h in factory_hashes.items()),
          "reset_all: factory files byte-identical")
    check(sorted(os.listdir(wimz)) == ["BatteryLow.wav", "WimZOnline.mp3"]
          and sorted(os.listdir(default)) == ["good_dog.mp3", "notes.txt", "sit.mp3"],
          "factory dirs: no files added or removed")

    # --- bad inputs ---
    r = reg.replace("nope/x", custom, "mp3")
    check(r == {"success": False, "error": "unknown sound id"}, f"replace: unknown id fails ({r})")
    r = reg.replace("default/../../x", custom, "mp3")
    check(r == {"success": False, "error": "unknown sound id"}, f"replace: traversal id rejected ({r})")
    check(reg.resolve("default/../../x") is None and reg.has_override("default/../../x") is False,
          "resolve/has_override: traversal id unknown")
    check(not (tmp / "x.mp3").exists() and not (tmp / "talks" / "x.mp3").exists(),
          "traversal: nothing written outside override root")
    r = reg.replace("default/sit", custom, "flac")
    check(r.get("success") is False, f"replace: unsupported fmt rejected ({r})")
    r = reg.replace("default/sit", b"", "mp3")
    check(r.get("success") is False, "replace: empty data rejected")

    # --- ffmpeg missing ---
    orig_run = sr.subprocess.run

    def fake_run(*a, **k):
        raise FileNotFoundError("ffmpeg")

    sr.subprocess.run = fake_run
    try:
        r = reg.replace("default/good_dog", b"RIFFfake-wav", "wav")
    finally:
        sr.subprocess.run = orig_run
    check(r.get("success") is False and "ffmpeg" in r.get("error", ""), f"replace wav w/o ffmpeg: clean failure ({r})")
    check(not reg.has_override("default/good_dog"), "replace wav w/o ffmpeg: no override created")
    gd_dir = override / "default"
    leftovers = [p.name for p in gd_dir.iterdir()] if gd_dir.exists() else []
    check(leftovers == [], f"replace wav w/o ffmpeg: no temp files left ({leftovers})")
    check(sha(default / "good_dog.mp3") == factory_hashes[str(default / "good_dog.mp3")],
          "replace wav w/o ffmpeg: factory untouched")

    # --- singleton ---
    check(sr.get_sound_registry() is sr.get_sound_registry(), "get_sound_registry: singleton")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
