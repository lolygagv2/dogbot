#!/usr/bin/env python

import argparse

try:
    import clip
except ModuleNotFoundError:
    raise ModuleNotFoundError("Module 'clip' not found. Please run: pip install clip-by-openai") from None
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

CLASSES = [
    "apple",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm",
]
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]
TF_RECORD_TYPE = ["val", "calib"]
CLASS_TOKEN_LOC = {
    "RN50": "models_files/cifar100/2023-03-09/class_token_resnet50.npy",
    "RN50x4": "models_files/cifar100/2023-03-09/class_token_resnet50x4.npy",
    "ViT-B/16": "models_files/cifar100/2023-03-09/class_token_vit_b_16.npy",
    "ViT-B/32": "models_files/cifar100/2023-03-09/class_token_vit_b_32.npy",
    "ViT-L/14": "models_files/cifar100/2023-09-05/class_token_vit_l_14.npy",
}
TF_RECORD_LOC = {
    "val": "models_files/cifar100/2023-03-09/cifar100_val.tfrecord",
    "calib": "models_files/cifar100/2023-03-09/cifar100_calib.tfrecord",
}


def _generate_class_tokens(model, classnames, templates):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.format(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cpu()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.mean(dim=0)
        class_embeddings = class_embeddings.detach().cpu().numpy()
        zeroshot_weights.append(class_embeddings)
    zeroshot_weights = np.stack(zeroshot_weights, axis=0)
    return zeroshot_weights


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _create_tfrecord(name, images, labels):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(len(images))
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i in range(len(images)):
            progress_bar.update(1)
            img_numpy = images[i]
            height = img_numpy.shape[0]
            width = img_numpy.shape[1]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "label_index": _int64_feature(labels[i][0]),
                        "image_numpy": _bytes_feature(np.array(img_numpy, np.uint8).tostring()),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def _create_class_tokens(model_name):
    model, _ = clip.load(model_name)
    tfrecords_filename = path_resolver.resolve_data_path(CLASS_TOKEN_LOC[model_name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)
    class_tokens = _generate_class_tokens(model, CLASSES, TEMPLATES)
    np.save(tfrecords_filename, class_tokens)
    print(f"Class tokens are saved in {tfrecords_filename}")


def run(type, model_name):
    _create_class_tokens(model_name)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    if type == "calib":
        images_num = _create_tfrecord(type, x_train, y_train)
    elif type == "val":
        images_num = _create_tfrecord(type, x_test, y_test)
    else:
        raise Exception(f"TFrecord type {type} is not supported.")
    print(f"Done converting {images_num} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create TFRecord of CIFAR100 dataset to use in CLIP models.\n\t"
            "Note: If you use a GPU and encounter an Out of Memory (OOM) error, "
            "consider setting CUDA_VISIBLE_DEVICES=99 to use CPU only."
        ),
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("type", help="TFRecord of which dataset to create", type=str, choices=TF_RECORD_TYPE)
    parser.add_argument(
        "--model", help="CLIP model to use", type=str, default="RN50", choices=list(CLASS_TOKEN_LOC.keys())
    )
    args = parser.parse_args()
    run(args.type, args.model)

"""
-----------------------------------------------------------------
CMD used to create a cifar100.tfrecord of the CIFAR100 dataset:
python create_clip_cifar100_tfrecord.py val
-----------------------------------------------------------------
"""
