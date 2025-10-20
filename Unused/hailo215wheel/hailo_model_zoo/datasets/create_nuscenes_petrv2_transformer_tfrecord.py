#!/usr/bin/env python

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "models_files/nuscenes/2024-08-21/nuscenes_val.tfrecord",
    "calib": "models_files/nuscenes/2024-08-21/nuscenes_calib.tfrecord",
}

SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type=cv2.IMREAD_UNCHANGED):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack([cv2.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = {
            "mean": np.zeros(num_channels, dtype=np.float32),
            "std": np.ones(num_channels, dtype=np.float32),
            "to_rgb": False,
        }

        return results


class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to cv2.IMREAD_UNCHANGED.
    """

    def __init__(
        self,
        sweeps_num=5,
        to_float32=False,
        pad_empty_sweeps=False,
        sweep_range=(3, 27),
        sweeps_id=None,
        color_type=cv2.IMREAD_UNCHANGED,
        sensors=SENSORS,
        test_mode=True,
        prob=1.0,
    ):
        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results["img"]
        img_timestamp = results["img_timestamp"]
        lidar_timestamp = results["timestamp"]
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for _ in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results["filename"].append(results["filename"][j])
                    results["lidar2img"].append(np.copy(results["lidar2img"][j]))
                    results["intrinsics"].append(np.copy(results["intrinsics"][j]))
                    results["extrinsics"].append(np.copy(results["extrinsics"][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1]
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results["sweeps"]):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results["sweeps"]))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1]

            for idx in choices:
                sweep_idx = min(idx, len(results["sweeps"]) - 1)
                sweep = results["sweeps"][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results["sweeps"][sweep_idx - 1]
                results["filename"].extend([sweep[sensor]["data_path"] for sensor in self.sensors])

                img = np.stack(
                    [cv2.imread(sweep[sensor]["data_path"], self.color_type) for sensor in self.sensors], axis=-1
                )

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]["timestamp"] / 1e6 for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results["lidar2img"].append(sweep[sensor]["lidar2img"])
                    results["intrinsics"].append(sweep[sensor]["intrinsics"])
                    results["extrinsics"].append(sweep[sensor]["extrinsics"])
        results["img"] = sweep_imgs_list
        results["timestamp"] = timestamp_imgs_list

        return results


def _get_input_dict(info):
    input_dict = {
        "sample_idx": info["token"],
        "pts_filename": info["lidar_path"],
        "sweeps": info["sweeps"],
        "timestamp": info["timestamp"] / 1e6,
    }

    input_dict.update(
        {
            "lidar2ego_rotation": info["lidar2ego_rotation"],
            "lidar2ego_translation": info["lidar2ego_translation"],
            "ego2global_rotation": info["ego2global_rotation"],
            "ego2global_translation": info["ego2global_translation"],
        }
    )

    image_paths = []
    lidar2img_rts = []
    intrinsics = []
    extrinsics = []
    img_timestamp = []
    for _, cam_info in info["cams"].items():
        img_timestamp.append(cam_info["timestamp"] / 1e6)
        image_paths.append(cam_info["data_path"])
        # obtain lidar to image transformation matrix
        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        intrinsics.append(viewpad)
        extrinsics.append(lidar2cam_rt)  ### The extrinsics mean the transformation from lidar to camera.
        # If anyone want to use the extrinsics as sensor to lidar,
        #  please use np.linalg.inv(lidar2cam_rt.T) and modify the
        #  ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
        lidar2img_rts.append(lidar2img_rt)

    input_dict.update(
        {
            "img_timestamp": img_timestamp,
            "img_filename": image_paths,
            "lidar2img": lidar2img_rts,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }
    )

    input_dict["img_fields"] = []
    input_dict["bbox3d_fields"] = []
    input_dict["pts_mask_fields"] = []
    input_dict["pts_seg_fields"] = []
    input_dict["bbox_fields"] = []
    input_dict["mask_fields"] = []
    input_dict["seg_fields"] = []

    transforms = [
        LoadMultiViewImageFromFiles(to_float32=True),
        LoadMultiViewImageFromMultiSweepsFiles(
            sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3, 27]
        ),
    ]

    for t in transforms:
        input_dict = t(input_dict)

    return input_dict


def _create_tfrecord(filenames, data_infos, name, num_images):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    print("Creating TFRecord...")
    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, data_input_path in enumerate(progress_bar):
            info_dict = _get_input_dict(data_infos[i])

            mlvl_feats = np.expand_dims(np.load(data_input_path)["input_layer1"], axis=0)
            coords_3d = np.expand_dims(np.load(data_input_path)["input_layer2"], axis=0)

            timestamp = info_dict["timestamp"]
            token = info_dict["sample_idx"]
            lidar2ego_rotation = info_dict["lidar2ego_rotation"]
            lidar2ego_translation = info_dict["lidar2ego_translation"]
            ego2global_rotation = info_dict["ego2global_rotation"]
            ego2global_translation = info_dict["ego2global_translation"]

            bs, feats_height, feats_width, mlvl_feats_ch = mlvl_feats.shape
            bs_coords, coords_height, coords_width, coords_3d_ch = coords_3d.shape
            assert bs_coords == bs
            assert feats_height == coords_height
            assert feats_width == coords_width
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "feats_height": _int64_feature(feats_height),
                        "feats_width": _int64_feature(feats_width),
                        "coords_height": _int64_feature(coords_height),
                        "coords_width": _int64_feature(coords_width),
                        "mlvl_feats_ch": _int64_feature(mlvl_feats_ch),
                        "coords_3d_ch": _int64_feature(coords_3d_ch),
                        "mlvl_feats": _bytes_feature(mlvl_feats.tobytes()),
                        "coords_3d": _bytes_feature(coords_3d.tobytes()),
                        "ind": _int64_feature(i),
                        "timestamp": _float_list_feature(timestamp),
                        "token": _bytes_feature(str.encode(token)),
                        "lidar2ego_rotation": _float_list_feature(lidar2ego_rotation),
                        "lidar2ego_translation": _float_list_feature(lidar2ego_translation),
                        "ego2global_rotation": _float_list_feature(ego2global_rotation),
                        "ego2global_translation": _float_list_feature(ego2global_translation),
                    }
                )
            )
            writer.write(example.SerializeToString())
    print(f"Tfrecord file created at {tfrecords_filename}")
    return i + 1


def run(ann_path, data_input_dir, name, num_images):
    print(f"Loading {ann_path}...")
    with open(ann_path, "rb") as f:
        infos = pickle.load(f)

    data_infos = sorted(infos["infos"], key=lambda e: e["timestamp"])
    files = sorted(Path(data_input_dir).iterdir(), key=os.path.getmtime)
    assert len(files) == len(data_infos), "mismatch between transformer input data and annotations"

    images_num = _create_tfrecord(files, data_infos, name, num_images)
    print("Done converting {} images".format(images_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE), choices=TF_RECORD_TYPE)
    parser.add_argument("--ann-file", help="annotation file path", type=str, default="")
    parser.add_argument(
        "--data-input-dir",
        help=(
            "transformer input data folder containing NPZs. "
            "Each with 'input_layer1' and 'input_layer2' keys "
            "holding backbone and 3d coords postional embedding data, "
            "respectively"
        ),
        type=str,
        default="",
    )
    parser.add_argument("--num-images", type=int, default=int(1e6), help="Limit num images")
    args = parser.parse_args()

    run(args.ann_file, args.data_input_dir, args.type, args.num_images)

""" Usage example
python create_nuscenes_petrv2_transformer_tfrecord.py val
--ann-file <annotations_pkl_file_path>
--data-input-dir <transformer_input_data_folder>
--num-images 6019
"""
