
# Initialize PCA9685
kit = ServoKit(channels=16, address=0x5F)  # Your documented address

class TreatDispenser:
    def __init__(self):
        # Your servo assignments
        self.carousel_servo = 0  # PCA channel 0
        self.winch_servo = 1    # PCA channel 1  
        self.camera_pan = 2
        self.camera_tilt = 3
        
    def dispense_treat(self):
        """Rotate carousel to next position"""
        # Rotate to dispense position
        kit.servo[self.carousel_servo].angle = 180
        time.sleep(0.5)
        
        # Return to home
        kit.servo[self.carousel_servo].angle = 90
        time.sleep(0.5)
        
    def launch_treat(self):
        """If using launch mechanism"""
        # Pull winch
        kit.servo[self.winch_servo].angle = 180
        time.sleep(1)
        
        # Release
        kit.servo[self.winch_servo].angle = 0
