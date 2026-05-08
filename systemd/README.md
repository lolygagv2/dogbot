# WIMZ Systemd Services

## Power Button Setup

The power button system uses a Pololu Mini Pushbutton Power Switch with relay isolation.

### Hardware Connections
- GPIO21 (Pin 40) → relay module IN
- GPIO20 (Pin 38) → button press sense (10kΩ external pull-down to GND)
- GPIO26 (Pin 37) → Pololu OFF kill pulse output
- 5V (Pin 2) → relay VCC
- GND (Pin 39) → relay GND

### Relay Polarity (CRITICAL)
Edit `/usr/local/bin/wimz_power_button.py` before starting:
- **Robot 1** (Teyleten optocoupler): `RELAY_ACTIVE_HIGH = True`
- **Robots 2-5** (5V SRD module): `RELAY_ACTIVE_HIGH = False`

### Installation
```bash
# Copy files
sudo cp services/power/wimz_power_button.py /usr/local/bin/
sudo chmod +x /usr/local/bin/wimz_power_button.py
sudo cp systemd/wimz-power-button.service /etc/systemd/system/
sudo cp systemd/wimz-killpulse /lib/systemd/system-shutdown/
sudo chmod +x /lib/systemd/system-shutdown/wimz-killpulse

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable wimz-power-button.service
sudo systemctl start wimz-power-button.service

# Verify
systemctl status wimz-power-button.service
journalctl -u wimz-power-button.service -n 20
```

### Expected Behavior
1. On boot: Relay clicks, button isolated from Pololu latch
2. Press button: Pi runs graceful shutdown (~15-20s), then power cuts
3. Press button again: Pololu cold-starts Pi, ~20s later relay clicks, ready

### Logs
```bash
journalctl -u wimz-power-button.service -f
```

Expected output:
```
WIMZ: Relay engaged on GPIO21 — button isolated from Pololu
WIMZ: Waiting for GPIO20 to settle LOW...
WIMZ: GPIO20 settled LOW. Arming press detection (100ms debounce).
```
