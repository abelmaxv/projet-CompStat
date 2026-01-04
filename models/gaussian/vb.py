import jax.numpy as jnp
from flax import nnx
import jax.random as jr
from typing import NamedTuple

class VBOutput(NamedTuple):
    z: jnp.ndarray
    mu_z: jnp.ndarray
    log_sigma_z: jnp.ndarray
    Delta: jnp.ndarray
    log_sigma: jnp.ndarray


class VB(nnx.Module):

    def __init__(self, dim: int, param_init: dict):
        self.dim = int(dim)
        
        # Model's parameters
        self.Delta = nnx.Param(param_init["Delta"])
        self.log_sigma = nnx.Param(param_init["log_sigma"])
        self.mu_z = nnx.Param(param_init["mu_z"])
        self.log_sigma_z = nnx.Param(param_init["log_sigma_z"])

    def sample_latent(self, rngs : nnx.Rngs)->jnp.ndarray:
        """Sample from the latent space according to q_phi(z|D)

        Args:
            rngs (nnx.Rngs): random number generator

        Returns:
            jnp.ndarray: sample z in the latent space
        """
        # reparametrization trick
        return self.mu_z[...] + jnp.exp(self.log_sigma_z[...])*jr.normal(rngs.param(), shape = (self.dim,))



    def __call__(self, rngs) -> VBOutput:
        """Returns the parameters of the VAE to compute ELBO

        Returns:
            VBOutput: parameters of the model
        """
        z = self.sample_latent(rngs)
        return VBOutput(
            z = z,
            Delta=self.Delta[...],
            log_sigma=self.log_sigma[...],
            mu_z = self.mu_z[...],
            log_sigma_z= self.log_sigma_z[...]
        )
