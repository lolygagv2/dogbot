#!/usr/bin/env python3
"""Xbox read_event must never busy-spin on a closed device — 2026-10-24.

Journal 02:44:04 (tb4): 28,408 "ERROR:XboxFixed:Error reading event: I/O
operation on closed file" lines in ONE second. _reopen_device() had closed the
stale fd, open() of the new node failed, and select() on the closed file raised
ValueError, which the generic handler logged and returned from with no sleep.
Now ValueError/EBADF/ENODEV take the disconnect path (5 s back-off + reopen),
other errors are rate-limited to one log per 5 s, and _reopen_device only
swaps fds once the new open succeeded.

Run: /usr/bin/python3 tests/hardware/test_xbox_read_event_closed_file.py
"""
import io
import os
import sys
import logging
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import xbox_hybrid_controller as xb  # noqa: E402

FAILS = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: FAILS.append(m)


class LogCatcher(logging.Handler):
    def __init__(self): super().__init__(); self.records = []
    def emit(self, r): self.records.append(r.getMessage())


def bare(device):
    c = xb.XboxHybridController.__new__(xb.XboxHybridController) if hasattr(xb, 'XboxHybridController') else None
    if c is None:
        cls = [v for v in vars(xb).values() if isinstance(v, type) and hasattr(v, 'read_event')][0]
        c = cls.__new__(cls)
    c.device = device
    c.device_path = '/dev/input/js_does_not_exist'
    return c


def main():
    sleeps = []
    xb.time.sleep = lambda s: sleeps.append(s)
    catcher = LogCatcher(); xb.logger.addHandler(catcher); xb.logger.setLevel(logging.DEBUG)

    # 1. closed file -> disconnect path: one warning, 5 s back-off, reopen attempted, NO error flood
    f = tempfile.NamedTemporaryFile(); closed = open(f.name, 'rb'); closed.close()
    c = bare(closed)
    for _ in range(50):
        c.read_event()
    errs = [m for m in catcher.records if m.startswith("Error reading event")]
    check(errs == [], f"closed file produces no 'Error reading event' flood (got {len(errs)})")
    check(sum(1 for m in catcher.records if "disconnected" in m) == 1, "exactly one 'disconnected' warning")
    check(len(sleeps) == 50 and all(s == 5.0 for s in sleeps), "every closed-file read backs off 5 s")
    check(c.device is closed, "device left untouched while the new node cannot be opened (no None, no re-close)")

    # 2. generic unexpected error -> rate-limited log + 0.1 s back-off
    catcher.records.clear(); sleeps.clear()
    class Weird(io.RawIOBase):
        def fileno(self): raise RuntimeError("boom")
    c2 = bare(Weird())
    for _ in range(100):
        c2.read_event()
    errs = [m for m in catcher.records if m.startswith("Error reading event")]
    check(len(errs) == 1, f"100 unexpected errors -> 1 log line (got {len(errs)})")
    check(len(sleeps) == 100 and all(s == 0.1 for s in sleeps), "each unexpected error backs off 0.1 s")

    # 3. successful reopen swaps fds and closes the old one
    good = tempfile.NamedTemporaryFile(); good.write(b"\0" * 64); good.flush()
    c3 = bare(open(good.name, 'rb')); old = c3.device; c3.device_path = good.name
    c3._device_disconnected = True
    c3._reopen_device()
    check(c3.device is not old and not c3.device.closed and old.closed, "reopen swaps to the new fd and closes the old")
    check(c3._device_disconnected is False, "reopen clears the disconnected flag")

    print(); print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
