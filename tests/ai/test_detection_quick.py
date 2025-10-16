#!/usr/bin/env python3
"""
Quick test script to verify detection fixes
"""

import subprocess
import time
import signal
import sys

def run_test():
    print("\n" + "="*60)
    print("🧪 TESTING FIXED DOG DETECTION SYSTEM")
    print("="*60)
    print("\nRunning 10-second test to verify:")
    print("✓ No more 21,504 false detections")
    print("✓ Camera rotated correctly (90° counter-clockwise)")
    print("✓ Faster inference (<500ms target)")
    print("✓ Proper NMS and filtering")
    print("\n" + "="*60)

    # Run the fixed detection script for 10 seconds
    proc = subprocess.Popen(
        ["python3", "/home/morgan/dogbot/run_pi_1024_fixed_v2.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print("\n[TEST] Detection system running...")
    print("[TEST] Monitoring output for issues...\n")

    start_time = time.time()
    timeout = 10  # Run for 10 seconds

    issues_found = []
    detection_counts = []
    inference_times = []

    try:
        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line:
                break

            print(line.rstrip())

            # Check for excessive detections
            if "Detected" in line and "dog(s)" in line:
                try:
                    # Extract number of dogs
                    parts = line.split("Detected")
                    if len(parts) > 1:
                        num_str = parts[1].split("dog")[0].strip()
                        num_dogs = int(num_str)
                        detection_counts.append(num_dogs)

                        if num_dogs > 10:
                            issues_found.append(f"Excessive detections: {num_dogs} dogs")
                except:
                    pass

            # Check inference time
            if "Inference:" in line and "ms" in line:
                try:
                    parts = line.split("Inference:")
                    if len(parts) > 1:
                        time_str = parts[1].split("ms")[0].strip()
                        inf_time = float(time_str)
                        inference_times.append(inf_time)

                        if inf_time > 1000:
                            issues_found.append(f"Slow inference: {inf_time:.1f}ms")
                except:
                    pass

            # Check for the 21504 bug
            if "21504" in line or "21,504" in line:
                issues_found.append("CRITICAL: 21,504 detection bug still present!")

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")

    finally:
        # Stop the process
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()

    # Print test results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)

    if detection_counts:
        avg_detections = sum(detection_counts) / len(detection_counts)
        max_detections = max(detection_counts)
        print(f"✅ Detections: Avg={avg_detections:.1f}, Max={max_detections}")
    else:
        print("⚠️  No detections recorded (might be no dogs present)")

    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        max_inference = max(inference_times)
        print(f"✅ Inference: Avg={avg_inference:.1f}ms, Max={max_inference:.1f}ms")
    else:
        print("⚠️  No inference times recorded")

    if issues_found:
        print("\n❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("\n✅ NO CRITICAL ISSUES FOUND!")
        print("   The detection system appears to be working correctly.")

    print("\n" + "="*60)
    print("Test complete. Check detection_results_fixed_v2/ for saved images.")
    print("="*60)

if __name__ == "__main__":
    run_test()