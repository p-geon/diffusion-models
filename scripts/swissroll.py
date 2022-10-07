import numpy as np
from sklearn.datasets import make_swiss_roll
from utils import visualize


def normalize(x: np.ndarray) -> np.ndarray:
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0]))
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1]))
    x = (2 * x) - 1.0
    return x


def load_swissroll():
    """
    x: [n_samples, 2]
    """
    x, color = make_swiss_roll(n_samples=1000, noise=0.0, random_state=None)
    x = np.delete(x, obj=1, axis=-1) # delete Y axis

    # normalize x and y
    x = normalize(x)
    visualize(x, color, savename="t=0")
    return x