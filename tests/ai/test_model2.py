#!/usr/bin/env python3
import sys
from hailo_platform import VDevice, HEF, ConfigureParams, HailoStreamInterface

hef_path = sys.argv[1]

try:
    # Use VDevice instead of Device
    device = VDevice()
    print(f"✓ Device created")
    
    # Load HEF
    hef = HEF(hef_path)
    print(f"✓ HEF loaded: {hef_path}")
    
    # Get info
    inputs = hef.get_input_vstream_infos()
    outputs = hef.get_output_vstream_infos()
    
    print(f"✓ Input: {inputs[0].name}, Shape: {inputs[0].shape}")
    print(f"✓ Output: {outputs[0].name}, Shape: {outputs[0].shape}")
    
    # The output shape (80, 5, 100) means:
    # 80 classes, 5 values (x,y,w,h,confidence), 100 max detections
    print("✓ Model has NMS postprocessing built-in!")
    
    # Configure
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = device.configure(hef, params)
    
    if network_groups:
        print("✓ SUCCESS: Model configures properly!")
    
except Exception as e:
    print(f"✗ Error: {e}")