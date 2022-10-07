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
        self.alpha_t_bar = create_alpha(n_steps=self.params.n_diffusion_steps, betas=self.betas)
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
        if(epoch%10==0):
            


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