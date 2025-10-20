#!/usr/bin/env python3
"""
Flask Web Server for DogBot
Run this ON the Raspberry Pi to serve the web interface
"""

from flask import Flask, render_template_string, Response, jsonify, request
from flask_cors import CORS
import cv2
import json
import threading
import time
from datetime import datetime
import logging

# Import your DogBot system
from main import DogBotAI

# Setup Flask
app = Flask(__name__)
CORS(app)

# Global variables
dogbot = None
video_frame = None
frame_lock = threading.Lock()

# HTML template (your web interface)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<!-- Your HTML from the web interface artifact goes here -->
<!-- I'll include it automatically when serving -->
</html>
"""

def generate_frames():
    """Generate video frames for streaming"""
    global video_frame
    
    while True:
        with frame_lock:
            if video_frame is None:
                # Create a placeholder frame if no video
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Feed", (200, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                frame = video_frame.copy()
                
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

def update_video_loop():
    """Continuously update the global video frame"""
    global video_frame, dogbot
    
    while True:
        try:
            if dogbot and hasattr(dogbot, 'camera'):
                frame = dogbot.camera.capture_frame()
                
                if frame is not None and dogbot.current_detection:
                    # Draw detection box
                    det = dogbot.current_detection
                    x, y, w, h = det.bbox
                    
                    # Color based on behavior
                    colors = {
                        'sitting': (0, 255, 0),
                        'lying': (255, 0, 0),
                        'standing': (0, 255, 255),
                        'playing': (255, 0, 255),
                        'idle': (128, 128, 128)
                    }
                    color = colors.get(det.behavior.value, (255, 255, 255))
                    
                    # Draw box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{det.behavior.value} ({det.confidence:.2f})"
                    cv2.putText(frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                with frame_lock:
                    video_frame = frame
                    
        except Exception as e:
            print(f"Video update error: {e}")
            
        time.sleep(0.03)

@app.route('/')
def index():
    """Serve the main web interface"""
    # Read the HTML file you created
    with open('/home/morgan/dogbot/web_interface.html', 'r') as f:
        html = f.read()
    return html

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current robot status"""
    global dogbot
    
    status = {
        'online': dogbot is not None,
        'behavior': 'idle',
        'confidence': 0,
        'fps': 30,
        'treatCount': 0,
        'lastTreat': 'Never'
    }
    
    if dogbot and dogbot.current_detection:
        det = dogbot.current_detection
        status['behavior'] = det.behavior.value
        status['confidence'] = det.confidence
        status['detection'] = {
            'x': det.bbox[0],
            'y': det.bbox[1],
            'width': det.bbox[2],
            'height': det.bbox[3]
        }
        
    return jsonify(status)

@app.route('/api/command', methods=['POST'])
def handle_command():
    """Handle commands from web interface"""
    global dogbot
    
    data = request.json
    command = data.get('command')
    
    if not dogbot:
        return jsonify({'error': 'Robot not initialized'}), 500
    
    try:
        if command == 'move':
            direction = data.get('direction')
            if direction == 'forward':
                dogbot.add_command({'type': 'move_forward', 'duration': 1.0})
            elif direction == 'backward':
                dogbot.add_command({'type': 'move_backward', 'duration': 1.0})
            elif direction == 'left':
                dogbot.add_command({'type': 'turn_left', 'angle': 45})
            elif direction == 'right':
                dogbot.add_command({'type': 'turn_right', 'angle': 45})
            elif direction == 'stop':
                dogbot.motor_controller.stop()
                
        elif command == 'servo':
            servo = data.get('servo')
            angle = data.get('angle')
            if servo == 'pan':
                dogbot.servo_controller.set_pan_angle(angle)
            elif servo == 'tilt':
                dogbot.servo_controller.set_tilt_angle(angle)
                
        elif command == 'center_servos':
            dogbot.servo_controller.center_all()
            
        elif command == 'dispense_treat':
            dogbot.dispense_treat()
            
        elif command == 'rotate_carousel':
            dogbot.servo_controller.rotate_carousel(1)
            
        elif command == 'led_pattern':
            pattern = data.get('pattern')
            dogbot.led_controller.set_pattern(pattern)
            
        elif command == 'play_sound':
            sound = data.get('sound')
            dogbot.audio_controller.play_sound(sound)
            
        elif command == 'camera_param':
            param = data.get('param')
            value = data.get('value')
            dogbot.camera.set_parameter(param, value)
            
        elif command == 'auto_adjust_camera':
            dogbot.camera.auto_adjust()
            
        elif command == 'snapshot':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/morgan/dogbot/snapshots/web_{timestamp}.jpg"
            dogbot.camera.save_snapshot(filename)
            
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/servo_info')
def get_servo_info():
    """Get current servo positions"""
    global dogbot
    
    if dogbot and hasattr(dogbot, 'servo_controller'):
        info = dogbot.servo_controller.get_servo_info()
        return jsonify(info)
    
    return jsonify({'pan': {'angle': 0}, 'tilt': {'angle': 0}})

@app.route('/api/camera_params')
def get_camera_params():
    """Get current camera parameters"""
    global dogbot
    
    if dogbot and hasattr(dogbot, 'camera'):
        params = dogbot.camera.get_parameters()
        return jsonify(params)
    
    return jsonify({})

def start_dogbot():
    """Initialize and start the DogBot system"""
    global dogbot
    
    print("Starting DogBot system...")
    dogbot = DogBotAI()
    dogbot.start()
    print("DogBot system started!")

def main():
    """Main entry point"""
    # Start DogBot in a separate thread
    dogbot_thread = threading.Thread(target=start_dogbot)
    dogbot_thread.daemon = True
    dogbot_thread.start()
    
    # Wait a bit for DogBot to initialize
    time.sleep(3)
    
    # Start video update thread
    video_thread = threading.Thread(target=update_video_loop)
    video_thread.daemon = True
    video_thread.start()
    
    # Start Flask server
    print("\n" + "="*50)
    print("DogBot Web Server Starting")
    print("="*50)
    print("\nAccess the web interface at:")
    print("  From this Pi: http://localhost:5000")
    print("  From your PC: http://raspberrypi.local:5000")
    print("  Or use IP:   http://[YOUR_PI_IP]:5000")
    print("\nPress Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()