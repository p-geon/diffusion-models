import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm


def visualize(x, c, savename):
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=t)
    plt.savefig('results/swissroll_0-1.png')

    plt.figure()
    plt.scatter(X[:, 1], X[:, 2], c=t)
    plt.savefig('results/swissroll_1-2.png')
    """

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=c)
    plt.savefig(f'results/{savename}.png')



def forward_process(x: np.ndarray, beta: float):
    q = np.random.normal(size=x.shape)
    x_t = np.sqrt(1-beta) * x + np.sqrt(beta) * q
    return x_t


def normalize(x: np.ndarray) -> np.ndarray:
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0]))
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1]))
    x = (2 * x) - 1.0
    return x


def create_beta(n_T: int):
    beta_1 = 10e-4
    beta_T = 0.02
    return np.linspace(beta_1, beta_T, n_T)


def create_dpm():
    # 2 layers multi-layer perceptron
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


def main():
    # params
    t = 1000
    betas = create_beta(n_T=t)
    batch_size = 10
    epochs = 1000


    # data
    x, c = make_swiss_roll(n_samples=1000, noise=0.0, random_state=None)
    x = np.delete(x, obj=1, axis=-1) # delete Y axis

    # normalize x and y
    x = normalize(x)
    visualize(x, c, savename="t=0")

    # define model
    ddpm = create_dpm()
    ddpm.compile(optimizer='adam', loss='mse')
    tf.keras.utils.plot_model(ddpm, to_file='results/model.png', show_shapes=True)

    # train
    for i in range(epochs):
        x_ts = np.zeros(shape=[t, 2])

        # forward process
        x_t = x
        for i in tqdm(range(t)):
            x_t_plus = forward_process(x_t[i], beta=betas[i])
            print(x_t_plus.shape)

            x_ts[t] = x_t_plus
            x_t = x_t_plus

        # reverse process
        loss = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            for i in tqdm(range(1, t)):
                epsilon_t = ddpm([x_ts[i], i/t])
                loss += tf.keras.losses.MeanSquaredError(x_ts[i-1], x_ts[i] - epsilon_t) 
               

if(__name__ == '__main__'):
    main()