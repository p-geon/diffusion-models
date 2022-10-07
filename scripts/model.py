import tensorflow as tf


def create_dpm():
    x = x_in = tf.keras.layers.Input(shape=[2])
    t = t_in = tf.keras.layers.Input(shape=[1])

    t = tf.keras.layers.Dense(10, activation='relu')(t)
    t = tf.keras.layers.Dense(10, activation='relu')(t)

    x = tf.keras.layers.Concatenate()([x, t])

    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(100, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(2)(x)
    y = x
    return tf.keras.Model(inputs=[x_in, t_in], outputs=y)
