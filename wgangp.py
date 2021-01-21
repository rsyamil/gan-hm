"""Credit to
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
for helpful reference to calculate the gp
"""
import sys

import util
import dataloader

import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv1D, UpSampling1D
from keras.layers import AveragePooling1D, MaxPooling1D
from keras.layers.merge import _Merge

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History, EarlyStopping

from keras.utils import plot_model
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects
from functools import partial

import string
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.cm as cm
from matplotlib.colors import Normalize

BATCH_SIZE = 128

def RMSE(x, y):
    return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))
    
class RandomWeightedAverage(_Merge):
    """from reference"""
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
class WGanGP:

    def __init__(self, M, D, z_dim=64, name=[]):
        self.name = name
        
        self.M = M 
        self.D = D
        
        self.mx = M.shape[1]
        self.my = M.shape[2]
        self.mz = M.shape[3]
        
        self.dx = D.shape[1]
        
        self.z_dim = z_dim
        
        self.critic_iter = 5
        
        self.generator = self.get_generator()
        self.critic = self.get_critic()
        
        self.generator_model = []
        self.critic_model = []
        
        self.noise = np.random.normal(0, 1, (25, self.z_dim))
        
        self.get_wgangp()
        
    def get_generator(self):
    
        noise = Input(shape=(self.z_dim, )) 
    
        _ = Dense(64*4*4, input_dim=self.z_dim)(noise)
        _ = Reshape((4, 4, 64))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(16, (3, 3))(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        generated_image = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(_)
        
        return Model(noise, generated_image)
        
    def get_critic(self):
    
        input_image = Input(shape=(self.mx, self.my, self.mz)) 
        
        _ = Conv2D(16, (3, 3), padding='same')(input_image)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(32, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(64, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Flatten()(_)

        score = Dense(1)(_)

        return Model(input_image, score)
        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """from reference"""
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)
        
    def get_wgangp(self):
    
        #critic compute graph
        self.critic.trainable = True
        self.generator.trainable = False
        
        real_img = Input(shape=(self.mx, self.my, self.mz)) 
        noise = Input(shape=(self.z_dim,))
        
        fake_img = self.generator(noise)
        score_fake = self.critic(fake_img)
        score_real = self.critic(real_img)
        
        interpolate_img = RandomWeightedAverage()([real_img, fake_img])
        score_interpolate = self.critic(interpolate_img)
        
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolate_img)
        partial_gp_loss.__name__ = 'gp'
        
        self.critic_model = Model([real_img, noise], [score_real, score_fake, score_interpolate])
        self.critic_model.compile(optimizer=RMSprop(lr=5e-5), 
                                    loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                                    loss_weights=[1, 1, 10])
        self.critic_model.summary()
        plot_model(self.critic_model, to_file='wgangp_critic.png')
        
        #generator compute graph
        self.critic.trainable = False
        self.generator.trainable = True
        
        noise_g = Input(shape=(self.z_dim,))
        fake_img_g = self.generator(noise_g)
        score_fake_g = self.critic(fake_img_g)
        
        self.generator_model = Model(noise_g, score_fake_g)
        self.generator_model.compile(optimizer=RMSprop(lr=5e-5), loss=self.wasserstein_loss)
        self.generator_model.summary()
        plot_model(self.generator_model, to_file='wgangp_generator.png')
    
    def save_generated_images(self, i):

        images = self.generator.predict(self.noise)
        util.plot_tile(images, "images/"+self.name+"/"+str(i))
    
    def train_wgangp(self, totalEpoch=300, batch_size=128, load=False, checkpoint=50):
             
        if not load:
        
            d_loss = 0
            d_losses = np.zeros([totalEpoch, 4])
            g_losses = np.zeros([totalEpoch, 1])
            
            real_label = (-1.0)*np.ones((batch_size, 1))
            fake_label = np.ones((batch_size, 1))
            dummy_label = np.zeros((batch_size, 1))
            
            for i in range(totalEpoch):
                for j in range(self.critic_iter):
                
                    real_images = self.M[np.random.randint(0, self.M.shape[0], batch_size)]
                    noise = np.random.normal(0, 1, [batch_size, self.z_dim])
                    
                    d_loss = self.critic_model.train_on_batch([real_images, noise], [real_label, fake_label, dummy_label])

                d_losses[i, :] = d_loss
                
                g_loss = self.generator_model.train_on_batch(noise, real_label)          
                g_losses[i, :] = g_loss
                
                print ("%d [D weighted total loss: %f] [G loss: %f]" % (i, d_loss[0], g_loss))
                util.plotAllLosses(d_losses, g_losses, name="wgangp_losses")
                
                if i % checkpoint == 0:
                    self.save_generated_images(i)
                    np.save("losses/"+self.name+"_d_losses.npy", np.array(d_losses))
                    np.save("losses/"+self.name+"_g_losses.npy", np.array(g_losses))

            self.critic_model.save('wgangp_critic.h5')
            self.generator_model.save('wgangp_generator.h5')
        else:
            print("Trained model loaded")
            self.critic_model = load_model('wgangp_critic.h5')
            self.generator_model = load_model('wgangp_generator.h5')
            

if __name__ == "__main__":

    if sys.argv[1] == False:
        load = False
    else:
        load = True
        
    #load data
    dataset = dataloader.DataLoader(verbose=True)
    x_train, x_test, y_train, y_test, y_reg_train, y_reg_test = dataset.load_data()

    #load trained architecture, to retrain set "load=False"
    wass_gan_gp = WGanGP(x_train, y_reg_train, z_dim=100, name="wgangp")
    wass_gan_gp.train_wgangp(totalEpoch=20000, batch_size=BATCH_SIZE, load=False, checkpoint=100)

            
        
