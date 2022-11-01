import tensorflow as tf
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    x = x / x.max()
    x = (2 * x) - 1.0
    return x


def load_data():
    """
    x: [n_samples, 2]
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize x and y
    x = x_train.reshape(-1, 784)
    x = normalize(x)

    # asserts
    assert x.shape == (60000, 784), x.shape
    assert x.min() < -1.0, x.min()
    assert x.max() > 1.0, x.max()
    return x