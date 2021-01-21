import sys
import random
import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline

from scipy.ndimage.filters import gaussian_filter1d as gf

gan_d_losses = np.load("gan_d_losses.npy")
gan_g_losses = np.load("gan_g_losses.npy")

print(gan_d_losses.shape)
print(gan_g_losses.shape)

wgan_d_losses = np.load("wgan_d_losses.npy")
wgan_g_losses = np.load("wgan_g_losses.npy")

print(wgan_d_losses.shape)
print(wgan_g_losses.shape)

wgangp_d_losses = np.load("wgangp_d_losses.npy")
wgangp_g_losses = np.load("wgangp_g_losses.npy")

print(wgangp_d_losses.shape)
print(wgangp_g_losses.shape)

#plot gan losses
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Discriminator', color='green')
ax1.plot(gf(gan_d_losses[:, 0], sigma=200), label='d_loss', c = 'green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.legend()
ax1.set_xlim([0, 18000])
ax1.grid(False)

ax2 = ax1.twinx() 
ax2.set_ylabel('Generator', color='red') 
ax2.plot(gf(gan_g_losses[:,0], sigma=200), label='g_loss', c = 'red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend()
ax1.set_xlim([0, 18000])
ax2.grid(False)

plt.title("GAN")
fig.tight_layout() 
fig.savefig("gan_losses"+'.png')
plt.close(fig)

#plot wgan losses
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Discriminator', color='green')
ax1.plot(gf(wgan_d_losses[:, 0], sigma=200), label='d_loss', c = 'green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.legend()
ax1.set_xlim([0, 18000])
ax1.grid(False)

ax2 = ax1.twinx() 
ax2.set_ylabel('Generator', color='red') 
ax2.plot(gf(wgan_g_losses[:,0], sigma=200), label='g_loss', c = 'red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend()
ax1.set_xlim([0, 18000])
ax2.grid(False)

plt.title("WGAN")
fig.tight_layout() 
fig.savefig("wgan_losses"+'.png')
plt.close(fig)

#plot wgangp losses
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Discriminator', color='green')
ax1.plot(gf(wgangp_d_losses[:, 0], sigma=200), label='d_loss', c = 'green')
ax1.tick_params(axis='y', labelcolor='green')
ax1.legend()
ax1.set_xlim([0, 18000])
ax1.grid(False)

ax2 = ax1.twinx() 
ax2.set_ylabel('Generator', color='red') 
ax2.plot(gf(wgangp_g_losses[:,0], sigma=200), label='g_loss', c = 'red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend()
ax1.set_xlim([0, 18000])
ax2.grid(False)

plt.title("WGANGP")
fig.tight_layout() 
fig.savefig("wgangp_losses"+'.png')
plt.close(fig)

