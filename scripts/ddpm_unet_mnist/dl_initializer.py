import os
import random
import logging
import numpy as np


def tf_initializer(seed: int=42, print_debug=True, mixed_precision: bool=False) -> None:
    """ Silence every warning of notice from tensorflow.
    checked TF versions
    - 2.7.0-gpu
    """
    # IN: https://github.com/LucaCappelletti94/silence_tensorflow/blob/master/silence_tensorflow/silence_tensorflow.py
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    
    if(print_debug): print(f'[{__name__}] tf.__version__: {tf.__version__}')
    if(print_debug): print(f'[{__name__}] GPU: {tf.test.is_gpu_available()}')

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if(print_debug): print(f'[{__name__}] GPU devices: {gpu_devices}')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
        if(print_debug): print(f'[{__name__}] memory growth: {device} DONE')

    '''set seeds
    '''
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if(print_debug): print(f'[{__name__}] set random seed: {seed}')

    if(mixed_precision):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f'[{__name__}] mixed precision: {tf.keras.mixed_precision.global_policy()}')
