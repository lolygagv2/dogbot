#!/usr/bin/env python3
"""
Hailo Detector - High-performance AI detection using Hailo-8L
Consolidates Hailo implementations from multiple files
"""

import numpy as np
import logging
from typing import Dict, List, Any
from .base_detector import BaseDetector

try:
    from hailo_platform.pyhailort import pyhailort
    HAILO_AVAILABLE = True
    HAILO_ERROR = None
except ImportError as e:
    HAILO_AVAILABLE = False
    HAILO_ERROR = str(e)

class HailoDetector(BaseDetector):
    """Hailo-8L accelerated dog detection using HailoRT platform"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger('HailoDetector')

        # Hailo components
        self.device = None
        self.hef = None
        self.network_group = None
        self.infer_model = None
        self.input_vstream_infos = None
        self.output_vstream_infos = None

        # Model configuration
        self.model_path = config.get('hailo_model_path', '/home/morgan/dogbot/ai/models/dogdetector_14.hef')
        self.input_shape = (640, 640, 3)

        # Initialize if available
        if HAILO_AVAILABLE:
            self._initialize()
        else:
            self.logger.error(f"Hailo platform not available: {HAILO_ERROR}")
            self.logger.error("HailoRT Python bindings are not installed!")
            self.logger.error("Install with: sudo apt install python3-hailo-platform")
            self.logger.error("Or install the wheel package if available")

    def _initialize(self):
        """Initialize Hailo device and load model"""
        try:
            self.logger.info(f"Loading Hailo model: {self.model_path}")

            # Initialize Hailo device (use VDevice for configure capability)
            self.device = pyhailort.VDevice()

            # Load HEF model
            self.hef = pyhailort.HEF(self.model_path)

            # Configure network group
            configure_params = pyhailort.ConfigureParams.create_from_hef(self.hef, interface=pyhailort.HailoStreamInterface.PCIe)
            network_groups = self.device.configure(self.hef, configure_params)
            if not network_groups:
                raise RuntimeError("Failed to configure network groups")
            self.network_group = network_groups[0]

            # Get input and output stream info
            self.input_vstream_infos = self.hef.get_input_vstream_infos()
            self.output_vstream_infos = self.hef.get_output_vstream_infos()

            # Get input shape from the model
            input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.input_shape = input_vstream_info.shape

            # Create inference model with proper configuration
            self.infer_model = self.device.create_infer_model(self.model_path)
            self.infer_model.set_batch_size(1)  # Single image inference

            # Configure output streams to AUTO format (let Hailo decide)
            for output_name in self.infer_model.output_names:
                output_stream = self.infer_model.output(output_name)
                output_stream.set_format_type(pyhailort.FormatType.AUTO)  # Let Hailo auto-configure
                self.logger.info(f"Configured output {output_name} as AUTO format")

            self.logger.info("Hailo InferModel ready for inference")

            self.initialized = True
            self.logger.info(f"Hailo detector initialized successfully - Input shape: {self.input_shape}")

        except Exception as e:
            self.logger.error(f"Hailo initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.initialized = False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using Hailo acceleration with corrected input format

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries
        """
        if not self.initialized or not self.infer_model:
            return []

        try:
            # Preprocess frame to UINT8 format (as expected by model)
            input_data = self._preprocess_frame(frame)
            if input_data is None:
                return []

            outputs = {}
            configured_model = None
            try:
                # Configure model for this frame
                configured_model = self.infer_model.configure()
                bindings = configured_model.create_bindings()

                # Get names
                input_names = self.infer_model.input_names
                output_names = self.infer_model.output_names

                self.logger.debug(f"Available outputs: {output_names}")
                self.logger.debug(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

                # Set input data (UINT8 format)
                if input_names:
                    bindings.input(input_names[0]).set_buffer(input_data)

                # Run inference (with timeout in milliseconds)
                configured_model.run([bindings], 1000)

                # Get output data from configured bindings using get_buffer_view
                for output_name in output_names:
                    try:
                        output_binding = bindings.output(output_name)
                        # Use get_buffer_view() since HailoRT expects it as a view
                        outputs[output_name] = output_binding.get_buffer_view()
                        self.logger.debug(f"Successfully got output view: {output_name} shape: {outputs[output_name].shape if hasattr(outputs[output_name], 'shape') else 'N/A'}")

                    except Exception as e:
                        self.logger.warning(f"Failed to get output view {output_name}: {e}")
                        continue

                self.logger.debug(f"Raw outputs collected: {list(outputs.keys())}")

            finally:
                # Always clean up the configured model after each inference
                if configured_model:
                    try:
                        configured_model.shutdown()
                    except:
                        pass

            # Postprocess outputs
            detections = self._postprocess_outputs(outputs, frame.shape)

            # Filter detections
            detections = self.filter_detections(detections)

            return detections

        except Exception as e:
            self.logger.error(f"Hailo detection failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo input - expects UINT8 format"""
        import cv2

        if self.input_shape is None:
            self.logger.error("Input shape not available")
            return None

        # Get input dimensions (height, width, channels)
        height, width, channels = self.input_shape

        # Resize to model input size
        resized = cv2.resize(frame, (width, height))

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Keep as UINT8 (0-255 range) as expected by the model
        return rgb_frame.astype(np.uint8)

    def _postprocess_outputs(self, output_dict: Dict[str, np.ndarray], original_shape: tuple) -> List[Dict[str, Any]]:
        """Postprocess Hailo outputs to detection format - handles both raw and NMS formats"""
        detections = []

        try:
            if not isinstance(output_dict, dict):
                self.logger.error(f"Expected dict output, got: {type(output_dict)}")
                return []

            self.logger.debug(f"Processing outputs: {list(output_dict.keys())}")

            # Check if we have NMS postprocessed output
            nms_output_keys = [k for k in output_dict.keys() if 'nms_postprocess' in k.lower()]
            if nms_output_keys:
                # Handle Hailo NMS postprocessed output
                nms_output = output_dict[nms_output_keys[0]]
                self.logger.debug(f"Processing NMS output: {nms_output_keys[0]}, type: {type(nms_output)}")

                # Hailo NMS output is usually a structured array or list of detections
                if hasattr(nms_output, '__len__') and len(nms_output) > 0:
                    orig_h, orig_w = original_shape[:2]

                    # Handle different NMS output formats
                    if hasattr(nms_output, 'dtype') and nms_output.dtype.names:
                        # Structured array format
                        for detection in nms_output:
                            if hasattr(detection, 'confidence') and detection.confidence > self.confidence_threshold:
                                detections.append({
                                    'bbox': [int(detection.x), int(detection.y), int(detection.w), int(detection.h)],
                                    'confidence': float(detection.confidence),
                                    'class_id': int(detection.class_id),
                                    'center': (int(detection.x + detection.w/2), int(detection.y + detection.h/2))
                                })
                    else:
                        # Try to iterate through NMS results
                        for i, detection in enumerate(nms_output):
                            if i > 100:  # Limit processing
                                break
                            try:
                                # Assume [x, y, w, h, confidence, class_id] format
                                if len(detection) >= 6:
                                    x, y, w, h, conf, class_id = detection[:6]
                                    if conf > self.confidence_threshold:
                                        detections.append({
                                            'bbox': [int(x), int(y), int(w), int(h)],
                                            'confidence': float(conf),
                                            'class_id': int(class_id),
                                            'center': (int(x + w/2), int(y + h/2))
                                        })
                            except:
                                continue

                self.logger.debug(f"Processed {len(detections)} detections from NMS output")
                return detections

            # Fallback to raw tensor processing
            output_tensor = list(output_dict.values())[0]
            if not isinstance(output_tensor, np.ndarray):
                self.logger.error(f"Unexpected output type: {type(output_tensor)}")
                return []

            self.logger.debug(f"Processing raw tensor shape: {output_tensor.shape}")

            # Handle raw YOLOv8 output format
            if len(output_tensor.shape) == 2:
                num_detections, num_features = output_tensor.shape

                # Extract components
                boxes = output_tensor[:, :4]  # x_center, y_center, width, height
                confidences = output_tensor[:, 4]  # objectness scores
                class_scores = output_tensor[:, 5:]  # class scores

                # Convert to detection format
                orig_h, orig_w = original_shape[:2]
                model_h, model_w, _ = self.input_shape

                scale_x = orig_w / model_w
                scale_y = orig_h / model_h

                for i in range(num_detections):
                    conf = confidences[i]
                    if conf < 0.1:  # Low threshold for initial filtering
                        continue

                    # Get best class
                    class_scores_i = class_scores[i]
                    class_id = np.argmax(class_scores_i)
                    class_conf = class_scores_i[class_id]
                    final_conf = conf * class_conf

                    if final_conf < self.confidence_threshold:
                        continue

                    # Convert box coordinates
                    cx, cy, w, h = boxes[i]
                    cx *= scale_x
                    cy *= scale_y
                    w *= scale_x
                    h *= scale_y

                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    w = int(w)
                    h = int(h)

                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': float(final_conf),
                        'class_id': int(class_id),
                        'center': (int(cx), int(cy))
                    })
            else:
                self.logger.error(f"Unexpected output shape: {output_tensor.shape}")

        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return detections

    def is_available(self) -> bool:
        """Check if Hailo detector is available"""
        return HAILO_AVAILABLE and self.initialized

    def cleanup(self):
        """Cleanup Hailo resources"""
        try:
            # Clean up device
            if self.device:
                try:
                    self.device.release()
                except:
                    pass

            self.logger.info("Hailo detector cleaned up")

        except Exception as e:
            self.logger.error(f"Hailo cleanup error: {e}")

        finally:
            self.device = None
            self.hef = None
            self.network_group = None
            self.infer_model = None
            self.input_vstream_infos = None
            self.output_vstream_infos = None
            self.initialized = False