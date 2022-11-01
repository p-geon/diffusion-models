from typing import Tuple, List
from dl_initializer import tf_initializer; tf_initializer(seed=42, print_debug=False) # call this before importing tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import create_dpm
from utils import create_beta, create_alpha, visualize, show_images
from data import load_data


def get_params():
    return EasyDict({
        "batch_size": 500,
        "epochs": 20001,
        "lr": 1e-3,
        'weight_decay': 1e-4,
        'n_diffusion_steps': 1000,
        'sampling_per_epochs': 50,
    })


class DPM:
    def __init__(self):
        self.params = get_params()
        self.betas = create_beta(n_steps=self.params.n_diffusion_steps)
        self.alpha_t, self.alpha_t_bar = create_alpha(n_steps=self.params.n_diffusion_steps, betas=self.betas)
        self.define_model()
        self.losses = []
        self.generated_t0 = []

    
    def define_model(self):
        # define model
        self.model = create_dpm()
        self.model.compile()
        tf.keras.utils.plot_model(self.model, to_file='results/model.png', show_shapes=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=self.params.lr, 
            weight_decay=self.params.weight_decay
        )
    

    def __call__(self):
        x = load_data()
        n_iters_per_epoch = x.shape[0] // self.params.batch_size

        for epoch in tqdm(range(self.params.epochs)):
            np.random.shuffle(x)
            for i in range(n_iters_per_epoch):
                x_0 = x[self.params.batch_size*i:self.params.batch_size*(i+1), :] # create batch

                # training
                x_t, t_mat, noise_eps = self.forward_process(x_0)
                self.train_step(x_t, t_mat, noise_eps)

            if(epoch%self.params.sampling_per_epochs==0):
                print(f"epoch: {epoch}, loss: {self.train_loss.result():.4f}")
                self.sampling(epoch)
                plt.figure()
                plt.ylim(0, 0.3)
                #plt.yscale('log')
                plt.grid(which='major',color='black',linestyle='-')
                plt.grid(which='minor',color='black',linestyle='-')
                plt.plot(self.losses)
                plt.savefig("results/loss.png")
    
            self.losses.append(self.train_loss.result())
            self.train_loss.reset_states()


        self.create_scattering_animation()


    def forward_process(self, x_0: np.ndarray) -> Tuple[np.ndarray]:
        t = np.random.randint(low=1, high=1000)
        noise_eps = np.random.normal(size=x_0.shape)

        x_0_alpha = np.sqrt(self.alpha_t_bar[t]) * x_0
        x_t = x_0_alpha + np.sqrt(1 - self.alpha_t_bar[t]) * noise_eps

        t_mat = np.ones(shape=[self.params.batch_size, 1]) * t/self.params.n_diffusion_steps

        return x_t.astype(np.float32), t_mat.astype(np.float32), noise_eps.astype(np.float32)


    @tf.function
    def train_step(self, x_t, t, noise_eps):
        with tf.GradientTape() as tape:
            pred_noise = self.model([x_t, t])
            loss = self.loss_object(noise_eps, pred_noise)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        self.train_loss(loss)
        

    def sampling(self, epoch: int, n_samples: int=64, n_dim: int=784) -> None:
        x_T = np.random.normal(size=[n_samples, n_dim])
        x_t = x_T.astype(np.float32)
        for t in reversed(range(self.params.n_diffusion_steps)): # t=999, 998, ..., 0
            # reverse process
            z = np.random.normal(size=[n_samples, n_dim]) if(t > 0) else np.zeros(shape=[n_samples, n_dim])

            t_mat = np.ones(shape=[n_samples, 1]) * t/self.params.n_diffusion_steps
            t_mat = t_mat.astype(np.float32)

            pred = self.model([x_t, t_mat])

            # [algo2 4]
            frac_a = 1 - self.alpha_t[t]
            frac_b = np.sqrt(1 - self.alpha_t_bar[t])
            pred_noise_eps = frac_a/frac_b * pred
            
            scale = 1/np.sqrt(self.alpha_t[t])
            sigma_t = np.sqrt(self.betas[t])
            x_t = scale * (x_t - pred_noise_eps) + sigma_t * z

            if(t==0):#t%100==0 or t==999):
                print("x_t", x_t.shape)
                imgs = x_t.numpy().reshape(-1, 28, 28)
                imgs = (imgs + 1.0) / 2.0
                imgs = np.clip(imgs, 0, 1)
                show_images(imgs, savename=f"sampling_{epoch}_{t}")


    def create_scattering_animation(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.grid()
        #plt.grid()
        ims = []
        for i in range(len(self.generated_t0)):
            scatter = ax.scatter(self.generated_t0[i][:, 0], self.generated_t0[i][:, 1], s=10, c='b')
            text = ax.text(-1.95, -1.95, f'epoch: {str(i*self.params.sampling_per_epochs).zfill(4)}', fontsize=15)
            ims.append([scatter, text])
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000, repeat=True)
        ani.save('results/scattering.gif', writer='imagemagick')


if(__name__ == '__main__'):
    DPM().__call__()