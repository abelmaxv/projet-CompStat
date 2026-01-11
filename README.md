# Project for Computational Statistics course (master MVA)

This project consisted in studying the paper [Hamiltonian Variational Auto-Encoder](https://arxiv.org/abs/1805.11328) which introduces an low-variance unbiased estimator for the gradient of the ELBO in VAE. This repository implements a [FLAX NNX](https://flax.readthedocs.io/en/stable/#) reproduction of their experiment on a Gaussian model. We add a variant of the model, IWHVAE, where the estimateur is given by the mean on L trajectories. 

