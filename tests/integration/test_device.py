import hailo_platform

try:
    device = hailo_platform.VDevice()
    print("Device initialized successfully")
    temp = device.get_chip_temperature()
    print(f"Temperature: {temp}°C")
except Exception as e:
    print(f"Error: {e}")
