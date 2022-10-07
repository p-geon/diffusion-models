from dl_initializer import tf_initializer; tf_initializer(seed=42, print_debug=False) # call this before importing tensorflow
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import create_dpm
from utils import create_beta, create_alpha, visualize
from swissroll import load_swissroll


def get_params():
    return EasyDict({
        "batch_size": 10,
        "epochs": 501,
        "lr": 2e-4,
        'n_diffusion_steps': 1000,
    })


class DPM:
    def __init__(self):
        self.params = get_params()
        self.betas = create_beta(n_steps=self.params.n_diffusion_steps)
        self.alpha_t, self.alpha_t_bar = create_alpha(n_steps=self.params.n_diffusion_steps, betas=self.betas)
        self.define_model()
        self.losses = []
        self.animations = {
            'epoch': [],
            'step': [],
        }

    
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
            np.random.shuffle(x)
            for i in range(n_iters_per_epoch):
                # create batch
                bs = self.params.batch_size
                x_0 = x[bs*i:bs*(i+1), :]

                # forward process
                x_t, t_mat, noise_eps = self.forward_process(x_0)

                self.train_step(x_t, t_mat, noise_eps)


            self.losses.append(self.train_loss.result())
            self.train_loss.reset_states()

            if(epoch%500==0):
                print(f"epoch: {epoch}, loss: {self.train_loss.result():.4f}")
                self.sampling(epoch)
                plt.figure()
                plt.plot(self.losses)
                plt.savefig("results/loss.png")

        ani = animation.ArtistAnimation(plt.figure(), self.animations['epoch'], interval=100)
        ani.save("results/epoch.gif")
        ani = animation.ArtistAnimation(plt.figure(), self.animations['step'], interval=100)
        ani.save("results/step.gif")



    def forward_process(self, x_0: np.ndarray) -> tuple:
        t = np.random.randint(low=1, high=1000)
        noise_eps = np.random.normal(size=x_0.shape)

        x_0_alpha = np.sqrt(self.alpha_t_bar[t]) * x_0
        x_t = x_0_alpha + np.sqrt(1 - self.alpha_t_bar[t]) * noise_eps

        t_mat = np.ones(shape=[self.params.batch_size, 1]) * t/self.params.n_diffusion_steps

        # casting for tensorflow
        x_t = x_t.astype(np.float32)
        t_mat = t_mat.astype(np.float32)
        noise_eps = noise_eps.astype(np.float32)
        return x_t, t_mat, noise_eps


    @tf.function
    def train_step(self, x_t, t, noise_eps):
        # reverse process
        with tf.GradientTape() as tape:
            pred_noise = self.model([x_t, t])
            loss = self.loss_object(noise_eps, pred_noise)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        self.train_loss(loss)
        

    def sampling(self, epoch: int, n_samples: int=1000) -> None:
        x_T = np.random.normal(size=[n_samples, 2])
        x_t = x_T.astype(np.float32)
        for t in reversed(range(self.params.n_diffusion_steps)): # t=999, 998, ..., 0
            z = np.random.normal(size=[n_samples, 2]) if(t > 0) else np.zeros(shape=[n_samples, 2])

            t_mat = np.ones(shape=[n_samples, 1]) * t/self.params.n_diffusion_steps
            t_mat = t_mat.astype(np.float32)

            pred = self.model([x_t, t_mat])

            # algo2 4:
            frac_a = 1 - self.alpha_t[t]
            frac_b = np.sqrt(1 - self.alpha_t_bar[t])
            pred_noise_eps = frac_a/frac_b * pred
            
            scale = 1/np.sqrt(self.alpha_t[t])
            sigma_t = np.sqrt(self.betas[t])
            x_t = scale * (x_t - pred_noise_eps) + sigma_t * z


            if(t%100==0 or t==999):
                figure = visualize(x_t, None, savename=f"epoch_{epoch}-{t}")
                if(t==0):
                    self.animations['epoch'].append(figure)
                if(epoch==500):
                    self.animations['step'].append(figure)


                if(False):
                    print("z", z) # [10, 2]
                    print("t_mat", t_mat) # [10, 1], t依存.
                    print("pred", pred) # [10, 2], float32
                    print('denoise_x', denoise_x) # [10, 2], inf or -inf
                    print('x_t', x_t) # [10, 2], inf or -inf


if(__name__ == '__main__'):
    DPM().__call__()