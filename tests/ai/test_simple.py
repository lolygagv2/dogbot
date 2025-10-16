#!/usr/bin/env python3
"""
Simple HailoRT test to isolate issues
"""

import sys

try:
    print("🔍 Testing HailoRT 4.21 import...")
    from hailo_platform.pyhailort.pyhailort import VDevice
    print("✅ Import successful")

    print("🔍 Creating VDevice...")
    vdevice = VDevice()
    print("✅ VDevice created")

    print("🔍 Testing device info...")
    print(f"Device info: {vdevice}")
    print("✅ Device info retrieved")

    print("🎉 HailoRT 4.21 basic functionality working!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)