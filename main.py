import time

def detect_dog_behavior():
    print("🔍 Scanning for dog behavior...")

def trigger_servo():
    print("🎯 Treat launched via servo!")

def play_sound():
    print("🔊 Playing reward sound...")

def log_event(event_type, data=""):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {event_type}: {data}")

def main():
    print("🐶 Dogbot main loop starting...")
    try:
        while True:
            detect_dog_behavior()
            trigger_servo()
            play_sound()
            log_event("treat_dispensed", "sit_pose_detected")
            time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 Exiting cleanly.")

if __name__ == "__main__":
    main()