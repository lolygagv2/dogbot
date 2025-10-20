#!/usr/bin/env python3
"""
DogBot Hailo Inference with TAPPAS
Real-time detection using Hailo-8 with proper view handling
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import sys
import time
import json
from pathlib import Path
from threading import Thread
from queue import Queue
import argparse

# For Hailo Model Zoo if needed
try:
    from hailo_model_zoo.core.postprocessing.detection_postprocessing import DetectionPostProcessing
    HAS_MODEL_ZOO = True
except ImportError:
    HAS_MODEL_ZOO = False
    print("Model Zoo not available for post-processing")

class HailoDogBotDetector:
    """Real-time dog behavior detection using Hailo with TAPPAS"""
    
    def __init__(self, hef_path, input_source=0, display=True):
        """
        Initialize the detector
        Args:
            hef_path: Path to compiled .hef model
            input_source: Camera index (0) or video file path
            display: Show video output
        """
        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
            
        self.input_source = input_source
        self.display = display
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Detection parameters
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Class names for your 10-class model
        self.class_names = {
            0: "elsa_sitting",
            1: "elsa_lying",
            2: "elsa_standing",
            3: "elsa_spinning",
            4: "elsa_playing",
            5: "bezik_sitting",
            6: "bezik_lying",
            7: "bezik_standing",
            8: "bezik_spinning",
            9: "bezik_playing"
        }
        
        # Behavior tracking
        self.behavior_history = {
            "elsa": [],
            "bezik": []
        }
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Create GStreamer pipeline with Hailo elements"""
        
        # Build pipeline string based on input source
        if isinstance(self.input_source, int):
            # Camera input
            source = f"v4l2src device=/dev/video{self.input_source} ! " \
                    "video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1"
        else:
            # Video file input
            source = f"filesrc location={self.input_source} ! decodebin"
        
        # Full pipeline with Hailo inference
        pipeline_str = f"""
            {source} ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format=RGB,width=640,height=640 !
            videoconvert !
            hailonet hef-path={self.hef_path} is-active=true !
            queue !
            hailofilter name=filter so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_post.so 
                        function-name=yolov8_postprocess config-file-path=/dev/null !
            queue !
            hailooverlay name=overlay !
            videoconvert !
            fpsdisplaysink name=display sync=false text-overlay=true
        """
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            return pipeline
        except Exception as e:
            print(f"Pipeline creation failed: {e}")
            # Fallback to simpler pipeline
            return self._create_simple_pipeline()
    
    def _create_simple_pipeline(self):
        """Create a simpler pipeline without post-processing"""
        pipeline = Gst.Pipeline()
        
        # Create elements
        if isinstance(self.input_source, int):
            source = Gst.ElementFactory.make("v4l2src", "source")
            source.set_property("device", f"/dev/video{self.input_source}")
        else:
            source = Gst.ElementFactory.make("filesrc", "source")
            source.set_property("location", self.input_source)
            decoder = Gst.ElementFactory.make("decodebin", "decoder")
            pipeline.add(decoder)
        
        convert = Gst.ElementFactory.make("videoconvert", "convert")
        scale = Gst.ElementFactory.make("videoscale", "scale")
        hailonet = Gst.ElementFactory.make("hailonet", "inference")
        hailonet.set_property("hef-path", str(self.hef_path))
        
        # For receiving raw tensors
        appsink = Gst.ElementFactory.make("appsink", "sink")
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_new_sample)
        
        # Add elements to pipeline
        pipeline.add(source)
        pipeline.add(convert)
        pipeline.add(scale)
        pipeline.add(hailonet)
        pipeline.add(appsink)
        
        # Link elements
        source.link(convert)
        convert.link(scale)
        scale.link(hailonet)
        hailonet.link(appsink)
        
        return pipeline
    
    def _on_new_sample(self, sink):
        """Handle new inference results"""
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        
        # Extract inference results
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        
        # Get tensor data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            data = np.frombuffer(map_info.data, dtype=np.float32)
            self._process_detections(data)
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _process_detections(self, raw_output):
        """Process raw model output"""
        # Reshape based on YOLOv8 output format
        # Expected: [1, 84, 8400] -> [1, num_classes+4, num_anchors]
        try:
            output = raw_output.reshape(1, 84, -1)
            
            # Extract boxes and scores
            boxes = output[0, :4, :].T  # [num_anchors, 4]
            scores = output[0, 4:, :].T  # [num_anchors, 80]
            
            # Apply NMS
            detections = self._apply_nms(boxes, scores)
            
            # Update behavior tracking
            self._update_behaviors(detections)
            
            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
        except Exception as e:
            print(f"Detection processing error: {e}")
    
    def _apply_nms(self, boxes, scores):
        """Apply Non-Maximum Suppression"""
        detections = []
        
        for class_id in range(10):  # Your 10 classes
            class_scores = scores[:, class_id]
            
            # Filter by confidence
            mask = class_scores > self.conf_threshold
            if not np.any(mask):
                continue
            
            class_boxes = boxes[mask]
            class_scores = class_scores[mask]
            
            # Simple NMS implementation
            indices = self._nms(class_boxes, class_scores, self.iou_threshold)
            
            for idx in indices:
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': float(class_scores[idx]),
                    'bbox': class_boxes[idx].tolist()
                })
        
        return detections
    
    def _nms(self, boxes, scores, threshold):
        """Simple NMS implementation"""
        # Convert to corner format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _update_behaviors(self, detections):
        """Update behavior tracking"""
        current_behaviors = {"elsa": None, "bezik": None}
        
        for det in detections:
            dog_name = det['class_name'].split('_')[0]
            behavior = det['class_name'].split('_')[1]
            
            current_behaviors[dog_name] = {
                'behavior': behavior,
                'confidence': det['confidence'],
                'timestamp': time.time()
            }
        
        # Update history
        for dog, behavior in current_behaviors.items():
            if behavior:
                self.behavior_history[dog].append(behavior)
                # Keep last 30 detections (1 second at 30fps)
                if len(self.behavior_history[dog]) > 30:
                    self.behavior_history[dog].pop(0)
    
    def run(self):
        """Start the detection pipeline"""
        print(f"Starting Hailo inference with model: {self.hef_path}")
        print(f"Input source: {self.input_source}")
        
        # Set pipeline to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start pipeline")
            return
        
        # Create main loop
        loop = GLib.MainLoop()
        
        # Handle messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message, loop)
        
        try:
            print("Running... Press Ctrl+C to stop")
            loop.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def _on_message(self, bus, message, loop):
        """Handle pipeline messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        elif t == Gst.MessageType.INFO:
            info, debug = message.parse_info()
            print(f"Info: {info}, {debug}")
    
    def stop(self):
        """Stop the pipeline"""
        self.pipeline.set_state(Gst.State.NULL)
        print(f"Final FPS: {self.fps:.2f}")
        print(f"Total frames processed: {self.frame_count}")
    
    def get_status(self):
        """Get current detection status"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'behaviors': {
                dog: self.behavior_history[dog][-1] if self.behavior_history[dog] else None
                for dog in ['elsa', 'bezik']
            }
        }


def main():
    parser = argparse.ArgumentParser(description='DogBot Hailo Inference')
    parser.add_argument('--hef', required=True, help='Path to HEF model file')
    parser.add_argument('--input', default=0, help='Camera index or video file')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Convert input to int if it's a camera index
    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input
    
    # Create and run detector
    detector = HailoDogBotDetector(
        hef_path=args.hef,
        input_source=input_source,
        display=not args.no_display
    )
    
    detector.conf_threshold = args.conf
    detector.run()


if __name__ == "__main__":
    main()
