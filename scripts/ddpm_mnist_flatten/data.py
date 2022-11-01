import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(x: np.ndarray) -> np.ndarray:
    x = x / x.max()
    x = (2 * x) - 1.0
    return x


def show_data(x):
    # show data
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(10):
        ax[i].imshow(x[i]/255.0, cmap='gray')
        ax[i].axis('off')
    plt.savefig("results/data.png")
    plt.close()


def load_data():
    """
    x: [n_samples, 2]
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape, x_train.max(), x_train.min())
    show_data(x_train)

    # normalize x and y
    x = x_train.reshape(-1, 784)
    x = normalize(x)

    # asserts
    assert x.shape == (60000, 784), x.shape
    assert x.min() < -1.0, x.min()
    assert x.max() > 1.0, x.max()
    return x