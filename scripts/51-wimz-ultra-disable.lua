-- WIM-Z: keep PipeWire/WirePlumber off the ultrasonic sound card (USB 12d1:0010).
-- Without this PipeWire opens it at 48 kHz and may route the robot's voice to it
-- (seen 2026-11-11: echo-cancel playback landed on it, speakers went silent).
-- ultrasonic.py plays to it directly with aplay at 96 kHz (hw:ultra).
-- Install on a LIVE unit (verified tb1 2026-11-12 — the short form below is NOT enough):
--   install -m644 scripts/51-wimz-ultra-disable.lua ~/.config/wireplumber/main.lua.d/
--   sudo systemctl stop treatbot.service                      # REQUIRED, see (b)
--   systemctl --user restart pipewire pipewire-pulse wireplumber   # all three, see (a)
--   sudo systemctl start treatbot.service
--   journalctl -u treatbot.service | grep "Bark capture"      # want: 'default' (PipeWire healthy)
--
-- Two traps if you shortcut this:
--  (a) echo-cancel is loaded by *pipewire*, not wireplumber
--      (~/.config/pipewire/pipewire.conf.d/echo-cancel.conf). Restarting only
--      wireplumber leaves the echo-cancel playback stream orphaned on sink
--      4294967295, so `arecord -D default` times out, bark detection falls back to
--      plughw:2,0 logging "AEC will be bypassed", and "Bark detection suppressed
--      (speaker echo)" then fires every ~0.4 s forever.
--  (b) While treatbot runs it holds the C-Media card's hw PCM through that same
--      plughw:2,0 fallback, so pipewire cannot reacquire the card on restart and
--      C-Media vanishes from `pactl list short cards` altogether.
--
-- Verify: `pactl list short sink-inputs` — the echo-cancel stream must target the
-- C-Media sink id, never 4294967295 — and 0 "speaker echo" suppressions when quiet.
-- The companion udev rule (85-wimz-ultra-audio.rules, ALSA id "ultra") is independent
-- and takes effect on hot-plug with no restart of anything.
table.insert(alsa_monitor.rules, {
  matches = {
    {
      { "device.vendor.id", "equals", "0x12d1" },
      { "device.product.id", "equals", "0x0010" },
    },
  },
  apply_properties = {
    ["device.disabled"] = true,
  },
})
