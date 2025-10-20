#!/usr/bin/env python3
"""
Dog behavior data collector - Records and organizes training data
"""

import cv2
import os
import json
from datetime import datetime
from picamera2 import Picamera2
import time

class DogDataCollector:
    def __init__(self):
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        self.camera.configure(config)
        
        # Directory structure
        self.base_dir = "/home/morgan/dogbot/dog_dataset"
        self.setup_directories()
        
        # Metadata tracking
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "samples": []
        }
        
    def setup_directories(self):
        """Create organized folder structure"""
        self.dirs = {
            'elsa_sitting': f"{self.base_dir}/images/elsa/sitting",
            'elsa_lying': f"{self.base_dir}/images/elsa/lying",
            'elsa_standing': f"{self.base_dir}/images/elsa/standing",
            'elsa_spinning': f"{self.base_dir}/images/elsa/spinning",
            'bezik_sitting': f"{self.base_dir}/images/bezik/sitting",
            'bezik_lying': f"{self.base_dir}/images/bezik/lying",
            'bezik_standing': f"{self.base_dir}/images/bezik/standing",
            'bezik_spinning': f"{self.base_dir}/images/bezik/spinning",
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def collect_video_samples(self, dog_name, behavior, duration=10):
        """Record video clips for temporal behaviors"""
        print(f"\n📹 Recording {dog_name} - {behavior}")
        print(f"Get {dog_name} to {behavior} for {duration} seconds")
        print("Press ENTER when ready...")
        input()
        
        output_dir = f"{self.base_dir}/videos/{dog_name}/{behavior}"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"{output_dir}/{dog_name}_{behavior}_{timestamp}.mp4"
        
        # Record video
        self.camera.start()
        self.camera.start_recording(video_path)
        
        for i in range(duration, 0, -1):
            print(f"Recording... {i} seconds remaining")
            time.sleep(1)
        
        self.camera.stop_recording()
        self.camera.stop()
        
        print(f"✅ Saved video: {video_path}")
        return video_path
    
    def collect_image_burst(self, dog_name, behavior, count=30):
        """Capture rapid image sequence"""
        print(f"\n📸 Capturing {count} images of {dog_name} - {behavior}")
        print(f"Get {dog_name} to {behavior}")
        print("Press ENTER to start burst capture...")
        input()
        
        save_dir = self.dirs[f"{dog_name}_{behavior}"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.camera.start()
        
        captured = []
        for i in range(count):
            frame = self.camera.capture_array()
            filename = f"{dog_name}_{behavior}_{timestamp}_{i:03d}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            
            captured.append(filepath)
            print(f"Captured {i+1}/{count}", end='\r')
            time.sleep(0.1)  # 10 FPS capture
        
        self.camera.stop()
        print(f"\n✅ Saved {count} images to {save_dir}")
        return captured

def main():
    collector = DogDataCollector()
    
    print("🐕 DogBot Training Data Collector")
    print("="*50)
    
    behaviors = ['sitting', 'lying', 'standing', 'spinning']
    dogs = ['elsa', 'bezik']
    
    for dog in dogs:
        print(f"\n\n📌 Now collecting data for: {dog.upper()}")
        print("="*50)
        
        for behavior in behaviors:
            print(f"\n--- Behavior: {behavior} ---")
            
            # Collect based on behavior type
            if behavior == 'spinning':
                # Need video for temporal behavior
                collector.collect_video_samples(dog, behavior, duration=5)
            else:
                # Static poses can use image bursts
                collector.collect_image_burst(dog, behavior, count=20)
            
            print("\nPress ENTER for next behavior or 'q' to skip...")
            if input().lower() == 'q':
                continue
    
    print("\n✅ Data collection complete!")
    print(f"Data saved to: {collector.base_dir}")

if __name__ == "__main__":
    main()