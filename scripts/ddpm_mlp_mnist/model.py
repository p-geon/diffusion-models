import tensorflow as tf


def create_dpm():
    bn_args = {
        'center': False, 
        'scale': False,
    }

    x = x_in = tf.keras.layers.Input(shape=[784])
    t = t_in = tf.keras.layers.Input(shape=[1])

    x = tf.keras.layers.Concatenate()([x, t])
    x = x_1 = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)

    x = tf.keras.layers.Concatenate()([x, t])
    x = x_2 = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)
    
    x = tf.keras.layers.Concatenate()([x, t])
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)

    x = tf.keras.layers.Concatenate()([x, x_2])
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization(**bn_args)(x)
    x = tf.keras.activations.swish(x)

    x = tf.keras.layers.Concatenate()([x, x_1])

    x = tf.keras.layers.Dense(784, kernel_initializer='zeros')(x)
    # noise の予測なので tanh はいらない
    y = x
    return tf.keras.Model(inputs=[x_in, t_in], outputs=y)
