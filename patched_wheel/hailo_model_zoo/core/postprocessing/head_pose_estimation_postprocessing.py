from math import cos, sin

import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="head_pose_estimation")
def head_pose_estimation_postprocessing(endnodes, device_pre_post_layers, **kwargs):
    if device_pre_post_layers is not None and device_pre_post_layers["softmax"]:
        probs = endnodes
    else:
        probs = [tf.nn.softmax(x) for x in endnodes]
    idx_tensor = list(range(66))
    pitch_predicted = tf.reduce_sum(probs[0] * idx_tensor, 1) * 3 - 99
    roll_predicted = tf.reduce_sum(probs[1] * idx_tensor, 1) * 3 - 99
    yaw_predicted = tf.reduce_sum(probs[2] * idx_tensor, 1) * 3 - 99
    return {
        "pitch": pitch_predicted,
        "roll": roll_predicted,
        "yaw": yaw_predicted,
    }


@VISUALIZATION_FACTORY.register(name="head_pose_estimation")
def visualize_head_pose_result(net_output, img, **kwargs):
    img = img[0]
    pitch, roll, yaw = net_output["pitch"][0], net_output["roll"][0], net_output["yaw"][0]
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    height, width = img.shape[0:2]
    tdx = width / 2
    tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = 100 * (cos(yaw) * cos(roll)) + tdx
    y1 = 100 * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    x2 = 100 * (-cos(yaw) * sin(roll)) + tdx
    y2 = 100 * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = 100 * (sin(yaw)) + tdx
    y3 = 100 * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
    return np.array(img, np.uint8)
