import tensorflow as tf
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0]))
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1]))
    x = (2 * x) - 1.0
    return x


def load_data():
    """
    x: [n_samples, 2]
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize x and y
    x = normalize(x)
    return x