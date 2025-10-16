#!/usr/bin/env python3
"""
Quick threshold test - just modify the confidence threshold and run a short test
"""

import json
import tempfile
import shutil
import os

def test_threshold(threshold):
    """Test with a specific confidence threshold by temporarily modifying the code"""

    print(f"\n=== Testing confidence threshold: {threshold} ===")

    # Read the current file
    with open('run_pi_1024x768.py', 'r') as f:
        content = f.read()

    # Find and replace the confidence threshold
    old_line = 'if conf < 0.25:  # Confidence threshold'
    new_line = f'if conf < {threshold}:  # Confidence threshold - TEMP TEST'

    if old_line not in content:
        print("Could not find confidence threshold line to modify")
        return

    # Create temporary modified version
    modified_content = content.replace(old_line, new_line)

    # Backup original
    shutil.copy('run_pi_1024x768.py', 'run_pi_1024x768.py.backup')

    try:
        # Write modified version
        with open('run_pi_1024x768.py', 'w') as f:
            f.write(modified_content)

        print(f"Modified threshold to {threshold}, running test...")

        # Run short test
        os.system(f'timeout 15 python3 test_pose_headless.py --duration 10 2>/dev/null | grep -E "(detections|Found|ISSUE|confidence)"')

    finally:
        # Restore original
        shutil.move('run_pi_1024x768.py.backup', 'run_pi_1024x768.py')
        print(f"Restored original file")

def main():
    """Test different thresholds"""

    print("=== QUICK CONFIDENCE THRESHOLD TEST ===")

    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]

    for threshold in thresholds:
        test_threshold(threshold)
        print("-" * 50)

    print("\n=== ANALYSIS ===")
    print("Look for thresholds that give:")
    print("- More than 0 detections")
    print("- Reasonable bbox sizes")
    print("- Not too many false positives")

if __name__ == "__main__":
    main()