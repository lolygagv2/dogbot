#!/usr/bin/env python3
"""
Xbox Controller Health Check Script
Monitors Xbox controller responsiveness and restarts if needed
"""

import os
import sys
import time
import psutil
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/morgan/dogbot/xbox_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_xbox_controller_process():
    """Check if Xbox controller process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'xbox_hybrid_controller.py' in str(proc.info['cmdline']):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def check_joystick_device():
    """Check if joystick device exists"""
    return os.path.exists('/dev/input/js0')

def test_servo_controller():
    """Quick test of servo controller"""
    try:
        # Import and test servo controller
        sys.path.append('/home/morgan/dogbot')
        from core.hardware.servo_controller import ServoController
        servo = ServoController()
        # Just check if we can initialize - don't move servos
        return True
    except Exception as e:
        logger.error(f"Servo controller test failed: {e}")
        return False

def restart_xbox_controller():
    """Restart Xbox controller process"""
    logger.info("Restarting Xbox controller...")

    # Kill existing process
    pid = check_xbox_controller_process()
    if pid:
        try:
            subprocess.run(['sudo', 'kill', '-9', str(pid)], check=True)
            time.sleep(2)
        except subprocess.CalledProcessError:
            pass

    # Start new process
    try:
        subprocess.Popen([
            '/home/morgan/dogbot/env_new/bin/python',
            '/home/morgan/dogbot/xbox_hybrid_controller.py'
        ], cwd='/home/morgan/dogbot')
        logger.info("Xbox controller restarted successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to restart Xbox controller: {e}")
        return False

def main():
    """Main health check loop"""
    logger.info("Xbox Controller Health Check Started")

    while True:
        try:
            # Check 1: Process running?
            pid = check_xbox_controller_process()
            if not pid:
                logger.warning("Xbox controller process not found")
                restart_xbox_controller()
                time.sleep(10)
                continue

            # Check 2: Joystick device exists?
            if not check_joystick_device():
                logger.warning("Joystick device /dev/input/js0 not found")
                time.sleep(5)
                continue

            # Check 3: Servo controller working?
            if not test_servo_controller():
                logger.warning("Servo controller test failed - restarting")
                restart_xbox_controller()
                time.sleep(10)
                continue

            logger.info(f"Health check OK - PID: {pid}")
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            logger.info("Health check stopped by user")
            break
        except Exception as e:
            logger.error(f"Health check error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()