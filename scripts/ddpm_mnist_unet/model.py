import tensorflow as tf

import math
def sinusoidal_embedding(x):
    # from: https://keras.io/examples/generative/ddim/
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    embedding_dims = 32

    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def scaler_to_2d(x: tf.Tensor, size: int) -> tf.Tensor:
    x = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)
    x = tf.tile(x, multiples=[1, size, size, 1])
    return x


def block(x: tf.Tensor, ch: int) -> tf.Tensor:
    bn_args = {
        'center': False,
        'scale': False,
    }

    x = tf.keras.layers.Conv2D(ch, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)
    x = tf.keras.layers.Conv2D(ch, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)
    return x


def create_dpm():
    x = x_in = tf.keras.layers.Input(shape=[28, 28, 1])
    t = t_in = tf.keras.layers.Input(shape=[1])

    x = tf.keras.layers.Concatenate()([x, scaler_to_2d(t, 28)]) # [batch, 28, 28, 1+1]
    x = block(x, ch=32)
    x_1 = x # [batch, 28, 28, 32]

    x = tf.keras.layers.Concatenate()([x, scaler_to_2d(t, 28)])
    #x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = block(x, ch=64)
    x_2 = x # [batch, 14, 14, 64]

    x = tf.keras.layers.Concatenate()([x, scaler_to_2d(t, 14)])
    #x = tf.keras.layers.MaxPool2D(2)(x) # [batch, 7, 7, 64]
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x) # [batch, 7, 7, 64]
    x = block(x, ch=128)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x) # [batch, 14, 14, 128]
    x = block(x, ch=64)
    x = tf.keras.layers.Concatenate()([x, x_2])

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x) # [batch, 28, 28, 64]
    x = block(x, ch=32)

    x = tf.keras.layers.Concatenate()([x, x_1])
    x = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), 
        kernel_initializer='zeros'
    )(x) # [batch, 28, 28, 1]
    # noise の予測なので tanh はいらない
    y = x
    return tf.keras.Model(inputs=[x_in, t_in], outputs=y)
