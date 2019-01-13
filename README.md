# University-of-Antwerp
 12 layers CNN is used to take generated noisy and clear images of vacuum environment and trains how to denoise generated noisy image and reconstruct image most like to ground truth image. Python code implementation of this network is available online here using Keras package. 
After training we save the model and then read noisy images from O2 and 5% H2 environments and then reconstruct less noisy images possible.
We have aboute one millions parameters that are learnable.
