import numpy as np
import matplotlib.pyplot as plt


def visualize(x, c, savename):
    plt.figure(figsize=(10, 10))
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    figure = plt.scatter(x[:, 0], x[:, 1], c=c)
    plt.savefig(f'results/{savename}.png')
    return figure


def create_beta(n_steps: int):
    beta_1 = 10e-4
    beta_T = 0.02
    return np.linspace(beta_1, beta_T, n_steps)


def create_alpha(n_steps: int, betas: np.ndarray):
    alpha_s = np.zeros(shape=[n_steps])
    for i in range(n_steps):
        alpha_s[i] = 1 - betas[i]

    alpha_t_bar = np.zeros(shape=[n_steps])
    for i in range(n_steps):
        alpha_t_bar[i] = np.prod(alpha_s[:i+1])
    alpha_t_bar = alpha_t_bar[:, np.newaxis] # add pixel-axis

    if(False): # show graph
        plt.figure()
        plt.plot(alpha_s)
        plt.plot(alpha_t_bar)
        plt.savefig('results/alpha_t_bar.png')
    return alpha_s, alpha_t_bar

