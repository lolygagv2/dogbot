#!/usr/bin/env python3
"""
Test Dog vs No Dog Detection
Compare what gets detected with and without a dog present
"""

import os
import time
import shutil

def modify_threshold(threshold):
    """Temporarily modify confidence threshold"""

    with open('run_pi_1024x768.py', 'r') as f:
        content = f.read()

    old_line = 'if conf < 0.25:  # Confidence threshold'
    new_line = f'if conf < {threshold}:  # Confidence threshold - TEST'

    if old_line not in content:
        # Check if already modified
        if 'TEST' in content:
            old_line = [line for line in content.split('\n') if 'if conf <' in line and 'TEST' in line][0]
            new_line = f'                if conf < {threshold}:  # Confidence threshold - TEST'
        else:
            print("Could not find threshold line to modify")
            return False

    modified_content = content.replace(old_line, new_line)

    # Backup and write
    shutil.copy('run_pi_1024x768.py', 'run_pi_1024x768.py.backup')
    with open('run_pi_1024x768.py', 'w') as f:
        f.write(modified_content)
    return True

def restore_original():
    """Restore original file"""
    if os.path.exists('run_pi_1024x768.py.backup'):
        shutil.move('run_pi_1024x768.py.backup', 'run_pi_1024x768.py')

def run_detection_test(description, duration=15):
    """Run detection test and capture key stats"""

    print(f"\n=== {description} ===")
    print("Running detection test...")

    # Run test and capture output
    cmd = f'timeout {duration+5} python3 test_pose_headless.py --duration {duration} 2>/dev/null'
    output = os.popen(cmd).read()

    # Parse output for key metrics
    lines = output.split('\n')

    detection_counts = []
    confidence_mentions = []
    bbox_info = []

    for line in lines:
        if 'Found' in line and 'detections after NMS' in line:
            try:
                count = int(line.split('Found ')[1].split(' detections')[0])
                detection_counts.append(count)
            except:
                pass

        if 'confidence' in line.lower():
            confidence_mentions.append(line.strip())

        if 'WARNING' in line and '48D' in line:
            bbox_info.append(line.strip())

    # Calculate stats
    if detection_counts:
        avg_detections = sum(detection_counts) / len(detection_counts)
        max_detections = max(detection_counts)
        frames_with_detections = len([c for c in detection_counts if c > 0])
        total_frames = len(detection_counts)
    else:
        avg_detections = max_detections = frames_with_detections = total_frames = 0

    print(f"Results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with detections: {frames_with_detections} ({frames_with_detections/max(1,total_frames)*100:.1f}%)")
    print(f"  Average detections per frame: {avg_detections:.1f}")
    print(f"  Max detections in single frame: {max_detections}")

    if confidence_mentions:
        print(f"  Confidence issues: {len(confidence_mentions)}")

    if bbox_info:
        print(f"  Bbox dimension warnings: {len(bbox_info)}")

    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'avg_detections': avg_detections,
        'max_detections': max_detections,
        'detection_rate': frames_with_detections/max(1,total_frames)
    }

def main():
    """Run comparative test"""

    print("=== DOG vs NO-DOG DETECTION COMPARISON ===")
    print("\nThis will test detection with lowered threshold (0.005)")
    print("to see difference between background false positives and real dog detections\n")

    # Set low threshold for testing
    if not modify_threshold(0.005):
        print("Failed to modify threshold")
        return

    try:
        # Test 1: No dog present
        input("==> FIRST: Make sure NO DOG is in camera view. Press Enter when ready...")
        no_dog_results = run_detection_test("NO DOG IN VIEW", duration=15)

        # Test 2: Dog present
        input("\n==> NOW: Put the dog in camera view. Press Enter when dog is visible...")
        dog_results = run_detection_test("DOG IN VIEW", duration=15)

        # Analysis
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)

        print(f"\nNO DOG:")
        print(f"  Detection rate: {no_dog_results['detection_rate']:.1%}")
        print(f"  Avg detections: {no_dog_results['avg_detections']:.1f}")
        print(f"  Max detections: {no_dog_results['max_detections']}")

        print(f"\nWITH DOG:")
        print(f"  Detection rate: {dog_results['detection_rate']:.1%}")
        print(f"  Avg detections: {dog_results['avg_detections']:.1f}")
        print(f"  Max detections: {dog_results['max_detections']}")

        # Interpretation
        print(f"\nINTERPRETATION:")

        if dog_results['detection_rate'] > no_dog_results['detection_rate'] * 1.5:
            print("✅ Model appears to detect dogs better than background")
            print("   -> Threshold 0.005 may be appropriate")
        elif dog_results['detection_rate'] > no_dog_results['detection_rate']:
            print("⚠️  Model detects dogs slightly better than background")
            print("   -> May need even lower threshold or model is marginal")
        else:
            print("❌ Model shows no improvement with dog present")
            print("   -> Model may not be trained for dogs or is broken")

        if no_dog_results['detection_rate'] > 0.8:
            print("⚠️  High false positive rate - detecting too much background")

        print(f"\nRECOMMENDATION:")
        if dog_results['detection_rate'] > 0.3 and dog_results['detection_rate'] > no_dog_results['detection_rate']:
            print("Use threshold 0.005 and test detection quality manually")
        else:
            print("Model appears unsuitable for dog detection - consider different model")

    finally:
        restore_original()
        print("\nOriginal threshold restored")

if __name__ == "__main__":
    main()