from dl_initializer import tf_initializer; tf_initializer(seed=42, print_debug=False) # call this before importing tensorflow
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt

from model import create_dpm
from utils import create_beta, create_alpha, visualize
from swissroll import load_swissroll


def get_params():
    return EasyDict({
        "batch_size": 10,
        "epochs": 10000,
        "lr": 0.001,
        'n_diffusion_steps': 1000,
    })


class DPM:
    def __init__(self):
        self.params = get_params()
        self.betas = create_beta(n_steps=self.params.n_diffusion_steps)
        self.alpha_t, self.alpha_t_bar = create_alpha(n_steps=self.params.n_diffusion_steps, betas=self.betas)
        self.define_model()

    
    def define_model(self):
        # define model
        self.model = create_dpm()
        self.model.compile()
        tf.keras.utils.plot_model(self.model, to_file='results/model.png', show_shapes=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(self.params.lr)
    

    def __call__(self):
        # data
        x = load_swissroll()

        n_iters_per_epoch = x.shape[0] // self.params.batch_size

        # train
        for epoch in tqdm(range(self.params.epochs)):
            # training
            for i in range(n_iters_per_epoch):
                # random sampling
                bs = self.params.batch_size
                np.random.shuffle(x)
                x_0 = x[bs*i:bs*(i+1), :]

                # forward process
                t = np.random.randint(low=0, high=1000)
                x_0_alpha = self.alpha_t_bar[t] * x_0
                x_t = x_0_alpha + np.sqrt(1 - self.alpha_t_bar[t]) * np.random.normal(size=x_0.shape)

                t_mat = np.ones(shape=[self.params.batch_size, 1]) * t/self.params.n_diffusion_steps

                # casting for tensorflow
                x_0_alpha = x_0_alpha.astype(np.float32)
                x_t = x_t.astype(np.float32)
                t_mat = t_mat.astype(np.float32)

                self.train_step(x_0_alpha, x_t, t_mat)

            print(f"epoch: {epoch}, loss: {self.train_loss.result():.4f}")
            self.train_loss.reset_states()

            # sampling
            if(epoch%100==0):
                n_samples = 1000
                x_T = np.random.normal(size=[n_samples, 2])
                x_t = x_T.astype(np.float32)
                for t in reversed(range(self.params.n_diffusion_steps)):
                    z = np.random.normal(size=[n_samples, 2]) if(t > 0) else np.zeros(shape=[n_samples, 2])

                    t_mat = np.ones(shape=[n_samples, 1]) * t/self.params.n_diffusion_steps
                    t_mat = t_mat.astype(np.float32)

                    pred = self.model([x_t, t_mat])

                    frac_a = 1 - self.alpha_t[t]
                    frac_b = np.sqrt(1 - self.alpha_t_bar[t])
                    denoise_x = frac_a/frac_b * pred # ここの係数がおかしい

                    sigma_t = 1
                    x_t = 1/np.sqrt(1 - self.alpha_t[t]) * (x_T - denoise_x) + sigma_t * z


                    if(t%100==0):
                        visualize(x_t, None, savename=f"epoch_{epoch}-{t}")
                        if(False):
                            print("z", z) # [10, 2]
                            print("t_mat", t_mat) # [10, 1], t依存.
                            print("pred", pred) # [10, 2], float32
                            print('denoise_x', denoise_x) # [10, 2], inf or -inf
                            print('x_t', x_t) # [10, 2], inf or -inf


    @tf.function
    def train_step(self, x_0_alpha, x_t, t):
        # reverse process
        with tf.GradientTape() as tape:
            #noise_epsilon = tf.random.normal(shape=[self.params.batch_size, 2])
            noise_epsilon = x_t - x_0_alpha
            pred = self.model([x_t, t])
            pred_noise = x_t - pred
            loss = self.loss_object(noise_epsilon, pred_noise)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        self.train_loss(loss)
        

if(__name__ == '__main__'):
    DPM().__call__()