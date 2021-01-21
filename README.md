# gans

Keras implementation of [DCGAN](https://github.com/rsyamil/gans/blob/main/gan.py), [WGAN](https://github.com/rsyamil/gans/blob/main/wgan.py) and [WGANGP](https://github.com/rsyamil/gans/blob/main/wgangp.py). To load trained model or train:

`python3 <gan.py|wgan.py|wgangp.py> <True|False>`

~2 hours training time on NVIDIA RTX 2080 Ti. Generated images when the models are trained for 20k epochs:

![imgs](/readme/gan_wgan_wgangp_images.gif)

Comparison of the generator and discriminator/critic losses:

![losses](/readme/losses_comp.png)
