-- WIM-Z: keep PipeWire/WirePlumber off the ultrasonic sound card (USB 12d1:0010).
-- Without this PipeWire opens it at 48 kHz and may route the robot's voice to it
-- (seen 2026-11-11: echo-cancel playback landed on it, speakers went silent).
-- ultrasonic.py plays to it directly with aplay at 96 kHz (hw:ultra).
-- Install: cp scripts/51-wimz-ultra-disable.lua ~/.config/wireplumber/main.lua.d/ &&
--          systemctl --user restart wireplumber
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
