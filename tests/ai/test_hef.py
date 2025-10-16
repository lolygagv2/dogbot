#!/usr/bin/env python3
import sys
from hailo_platform import Device, HEF, ConfigureParams, HailoStreamInterface

hef_path = sys.argv[1]

try:
    # Create device
    device = Device()
    print(f"Device created")
    
    # Load HEF
    hef = HEF(hef_path)
    print(f"HEF loaded: {hef_path}")
    
    # Get info
    print(f"Input layers: {hef.get_input_vstream_infos()[0].name}")
    print(f"Output layers: {[o.name for o in hef.get_output_vstream_infos()]}")
    
    # Try to configure
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = device.configure(hef, params)
    
    if network_groups:
        print("SUCCESS: Model configured!")
        # Clean up
        device.release()
    else:
        print("FAILED: Could not configure model")
        
except Exception as e:
    print(f"Error: {e}")