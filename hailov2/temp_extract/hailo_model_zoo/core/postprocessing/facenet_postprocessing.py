import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="face_verification")
def facenet_postprocessing(endnodes, device_pre_post_layers, **kwargs):
    embeddings = tf.nn.l2_normalize(endnodes, -1, 1e-10, name="embeddings")
    return {"predictions": embeddings}


@VISUALIZATION_FACTORY.register(name="face_verification")
def visualize_face_result(embeddings1, embeddings2, filenames, **kwargs):
    matplotlib.use("TkAgg")
    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(np.array(embeddings1 + embeddings2))
    for i, c in enumerate(["r", "g", "b", "y"]):
        if i >= int(len(filenames) / 2):
            break
        img_name = " ".join([filenames[2 * i].decode(), filenames[2 * i + 1].decode()])
        plt.scatter(reduced[i, 0], reduced[i, 1], c=c, label=img_name)
        plt.scatter(reduced[i + len(embeddings1), 0], reduced[i + len(embeddings1), 1], c=c)
    plt.legend()
    plt.show()
