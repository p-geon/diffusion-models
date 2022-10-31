import tensorflow as tf


def create_dpm():
    x = x_in = tf.keras.layers.Input(shape=[784])
    t = t_in = tf.keras.layers.Input(shape=[1])

    x = tf.keras.layers.Concatenate()([x, t])

    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(784)(x)
    y = x
    return tf.keras.Model(inputs=[x_in, t_in], outputs=y)
