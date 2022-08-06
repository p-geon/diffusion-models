import tensorflow as tf
import numpy as np


class DPM(object):
    def __init__(self):
        pass



def main():
    # load mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


if(__name__ == '__main__'):
    main()