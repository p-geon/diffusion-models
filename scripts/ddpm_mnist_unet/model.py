import tensorflow as tf


def scaler_to_2d(x: tf.Tensor) -> tf.Tensor:
    x = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)
    x = tf.tile(x, multiples=[1, 28, 28, 1])
    return x


def block(x: tf.Tensor, ch: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(ch, 3, padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def create_dpm():
    x = x_in = tf.keras.layers.Input(shape=[28, 28, 1])
    t = t_in = tf.keras.layers.Input(shape=[1])
    t = scaler_to_2d(t)

    x = tf.keras.layers.Concatenate()([x, t]) # [batch, 28, 28, 1+1]
    x = block(x, ch=32)
    x = block(x, ch=32)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = block(x, ch=64)
    x = block(x, ch=64)
    x = tf.keras.layers.MaxPool2D(2)(x) # [batch, 7, 7, 64]
    x = block(x, ch=128)
    x = block(x, ch=128)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = block(x, ch=64)
    x = block(x, ch=64)
    x = tf.keras.layers.UpSampling2D(2)(x) # [batch, 28, 28, 64]
    x = block(x, ch=32)
    x = block(x, ch=32)
    x = tf.keras.layers.Conv2D(1, 3, padding='same')(x) # [batch, 28, 28, 1]
    # noise の予測なので tanh はいらない
    y = x
    return tf.keras.Model(inputs=[x_in, t_in], outputs=y)
