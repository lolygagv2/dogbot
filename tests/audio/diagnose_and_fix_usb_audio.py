#!/usr/bin/env python3
"""
Comprehensive USB audio diagnostic and fix tool
"""

import subprocess
import sys
import os
import time
import numpy as np

def run_command(cmd, shell=False):
    """Run a shell command and return output"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout, result.returncode
    except Exception as e:
        return str(e), -1

def diagnose_usb_audio():
    """Comprehensive USB audio diagnosis"""
    print("="*60)
    print("USB AUDIO DIAGNOSTIC")
    print("="*60)

    issues = []
    fixes = []

    # 1. Check USB autosuspend
    print("\n1. Checking USB power management...")
    autosuspend, _ = run_command("cat /sys/module/usbcore/parameters/autosuspend")
    autosuspend = autosuspend.strip()
    print(f"   USB autosuspend: {autosuspend}")

    if autosuspend != "-1":
        issues.append("USB autosuspend is enabled - device may sleep during use")
        fixes.append("Disable USB autosuspend")

    # 2. Check specific device power control
    print("\n2. Checking USB Audio device power control...")
    # Find the USB device path
    usb_devices, _ = run_command("find /sys/bus/usb/devices -name '*0d8c*' 2>/dev/null", shell=True)

    for device_path in usb_devices.strip().split('\n'):
        if device_path:
            power_control_path = os.path.join(os.path.dirname(device_path), "power/control")
            if os.path.exists(power_control_path):
                with open(power_control_path, 'r') as f:
                    power_state = f.read().strip()
                print(f"   Device power control: {power_state}")
                if power_state == "auto":
                    issues.append(f"USB device has auto power management")
                    fixes.append(f"Set device to 'on': echo on > {power_control_path}")

    # 3. Check sample rate capabilities
    print("\n3. Testing sample rates...")
    test_rates = [8000, 16000, 22050, 44100, 48000]
    working_rates = []

    for rate in test_rates:
        cmd = f"timeout 1 arecord -D hw:2,0 -f S16_LE -r {rate} -d 0.1 -t raw 2>&1"
        output, ret = run_command(cmd, shell=True)
        if "Rate" not in output and ret == 0:
            working_rates.append(rate)
            print(f"   ✅ {rate}Hz works")
        else:
            print(f"   ❌ {rate}Hz failed")

    if 44100 not in working_rates and 48000 not in working_rates:
        issues.append("Standard sample rates not working properly")

    # 4. Check buffer sizes
    print("\n4. Testing buffer configurations...")
    buffer_tests = [
        ("Small buffer", "-B 1000"),
        ("Medium buffer", "-B 10000"),
        ("Large buffer", "-B 100000"),
    ]

    for name, buffer_arg in buffer_tests:
        cmd = f"timeout 1 arecord -D hw:2,0 -f S16_LE -r 44100 {buffer_arg} -d 0.5 -t raw 2>&1"
        output, ret = run_command(cmd, shell=True)
        if ret == 0:
            print(f"   ✅ {name} works")
        else:
            print(f"   ❌ {name} failed")
            if "xrun" in output.lower():
                issues.append(f"{name} causes buffer underruns")

    # 5. Check ALSA configuration
    print("\n5. Checking ALSA configuration...")
    asound_conf = os.path.expanduser("~/.asoundrc")
    if os.path.exists(asound_conf):
        print(f"   ~/.asoundrc exists")
        with open(asound_conf, 'r') as f:
            content = f.read()
            if "pcm.!default" in content:
                print("   Custom default device configured")
    else:
        print("   No ~/.asoundrc (using system defaults)")

    # 6. Check for IRQ issues
    print("\n6. Checking interrupts...")
    interrupts, _ = run_command("cat /proc/interrupts | grep -i 'usb\\|xhci'", shell=True)
    print("   USB interrupts:")
    for line in interrupts.strip().split('\n')[:3]:
        if line:
            print(f"   {line[:80]}")

    # 7. Test continuous recording
    print("\n7. Testing continuous recording stability...")
    print("   Recording for 5 seconds...")

    start_time = time.time()
    failures = 0
    chunks = 0

    while time.time() - start_time < 5:
        cmd = "timeout 0.5 arecord -D hw:2,0 -f S16_LE -r 44100 -d 0.3 -t raw 2>/dev/null"
        output, ret = run_command(cmd, shell=True)
        chunks += 1
        if ret != 0:
            failures += 1
        time.sleep(0.1)

    success_rate = (chunks - failures) / chunks * 100
    print(f"   Success rate: {success_rate:.1f}% ({chunks - failures}/{chunks})")

    if success_rate < 90:
        issues.append(f"Recording unstable - only {success_rate:.1f}% success rate")

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    if issues:
        print("\n❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n🔧 RECOMMENDED FIXES:")
        for i, fix in enumerate(fixes, 1):
            print(f"   {i}. {fix}")
    else:
        print("\n✅ No major issues found")

    return issues, fixes

def apply_fixes():
    """Apply recommended fixes"""
    print("\n" + "="*60)
    print("APPLYING FIXES")
    print("="*60)

    print("\n1. Disabling USB autosuspend for this session...")
    cmd = "echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend"
    output, ret = run_command(cmd, shell=True)
    if ret == 0:
        print("   ✅ USB autosuspend disabled")
    else:
        print("   ❌ Failed (need sudo)")

    print("\n2. Setting USB Audio device to always-on...")
    # Find USB audio device
    cmd = "find /sys/bus/usb/devices -name '*0d8c*' 2>/dev/null | head -1"
    device_path, _ = run_command(cmd, shell=True)
    device_path = device_path.strip()

    if device_path:
        power_path = os.path.join(os.path.dirname(device_path), "power/control")
        cmd = f"echo on | sudo tee {power_path}"
        output, ret = run_command(cmd, shell=True)
        if ret == 0:
            print(f"   ✅ Device power set to 'on'")
        else:
            print(f"   ❌ Failed to set power (need sudo)")

    print("\n3. Creating optimized ALSA config...")
    asound_config = """# Optimized USB Audio configuration
pcm.usb_buffered {
    type plug
    slave {
        pcm "hw:2,0"
        format S16_LE
        rate 44100
        channels 1
        period_size 1024
        buffer_size 8192
    }
}

pcm.usb_softvol {
    type softvol
    slave {
        pcm "usb_buffered"
    }
    control {
        name "USB Mic Boost"
        card 2
    }
    min_dB -5.0
    max_dB 20.0
    resolution 100
}
"""

    asound_path = os.path.expanduser("~/.asoundrc")

    # Backup existing config
    if os.path.exists(asound_path):
        backup_path = asound_path + ".backup"
        os.rename(asound_path, backup_path)
        print(f"   Backed up existing config to {backup_path}")

    with open(asound_path, 'w') as f:
        f.write(asound_config)
    print(f"   ✅ Created optimized ALSA config at {asound_path}")

    print("\n4. Testing with new configuration...")
    cmd = "timeout 2 arecord -D usb_buffered -f S16_LE -r 44100 -d 1 /tmp/test_fixed.wav 2>&1"
    output, ret = run_command(cmd, shell=True)

    if ret == 0:
        print("   ✅ Recording works with new config!")
        # Check file
        if os.path.exists("/tmp/test_fixed.wav"):
            size = os.path.getsize("/tmp/test_fixed.wav")
            print(f"   File size: {size} bytes")
    else:
        print("   ❌ Still having issues")
        print(f"   Error: {output[:200]}")

def test_pyaudio_with_fixes():
    """Test PyAudio after fixes"""
    print("\n" + "="*60)
    print("TESTING PYAUDIO WITH FIXES")
    print("="*60)

    import pyaudio

    # Suppress ALSA warnings
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)

    p = pyaudio.PyAudio()

    os.dup2(old_stderr, 2)
    os.close(old_stderr)
    os.close(devnull)

    print("\nTrying different configurations...")

    configs = [
        ("Direct hardware", 1, 44100, 1024),
        ("Direct with larger buffer", 1, 44100, 4096),
        ("Lower rate", 1, 16000, 1024),
        ("Tiny chunks", 1, 44100, 256),
    ]

    for name, device, rate, chunk in configs:
        print(f"\nTesting: {name}")
        print(f"  Device: {device}, Rate: {rate}, Chunk: {chunk}")

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device,
                frames_per_buffer=chunk
            )

            print("  Reading 5 chunks...")
            success = 0
            for i in range(5):
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(audio**2))
                    print(f"    Chunk {i+1}: RMS={rms:.4f}")
                    success += 1
                    time.sleep(0.1)
                except Exception as e:
                    print(f"    Chunk {i+1}: ERROR - {e}")
                    break

            stream.close()

            if success == 5:
                print(f"  ✅ Configuration works!")
                return True, device, rate, chunk
            else:
                print(f"  ❌ Failed after {success} chunks")

        except Exception as e:
            print(f"  ❌ Could not open: {e}")

    p.terminate()
    return False, None, None, None

def main():
    print("USB AUDIO FIX UTILITY")
    print("="*60)

    # Run diagnostics
    issues, fixes = diagnose_usb_audio()

    if issues:
        print("\nWould you like to apply the recommended fixes?")
        print("Note: Some fixes require sudo access")
        print("\nPress Enter to apply fixes, or Ctrl+C to skip...")

        try:
            input()
            apply_fixes()

            # Test after fixes
            print("\nTesting PyAudio after fixes...")
            works, device, rate, chunk = test_pyaudio_with_fixes()

            if works:
                print("\n" + "="*60)
                print("✅ SUCCESS!")
                print(f"Working configuration found:")
                print(f"  Device: {device}")
                print(f"  Sample Rate: {rate}")
                print(f"  Chunk Size: {chunk}")
                print("\nYou can now use these settings in your bark detection scripts.")
            else:
                print("\n⚠️  PyAudio still has issues.")
                print("Try using the 'usb_buffered' ALSA device with arecord instead.")

        except KeyboardInterrupt:
            print("\nSkipping fixes.")
    else:
        print("\n✅ No issues found - testing PyAudio...")
        test_pyaudio_with_fixes()

if __name__ == "__main__":
    main()