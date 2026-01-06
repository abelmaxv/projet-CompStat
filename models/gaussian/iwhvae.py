from .hvae import HVAE, HVAEOutput
import jax.numpy as jnp
from flax import nnx

class IWHVAEOutput(HVAEOutput):
    z0: jnp.ndarray
    rho0: jnp.ndarray
    zK: jnp.ndarray
    rhoK: jnp.ndarray
    Delta: jnp.ndarray
    log_sigma: jnp.ndarray

class IWHVAE(HVAE):
    def __init__(self, dim: int, K: int, L : int, param_init: dict, tempering : bool = True ):
        super().__init__(dim, K, param_init, tempering)
        self.L = L

    def __call__(self, n_data: int, x_bar: jnp.ndarray, rngs: nnx.Rngs) -> HVAEOutput:
        """Returns a named tuple with all elements for elbo computation by the trainer class

        Args:
            n_data (int): number of data points
            x_bar (jnp.ndarray):  barycenter of the dataset
            rngs (nnx.Rngs): random number generator

        Returns:
            HVAEOutput: elements for elbo computation
        """
        z0, rho0, zK, rhoK, _ = self.his(n_data, x_bar, rngs)
        return IWHVAEOutput(
            z0=z0, rho0=rho0, zK=zK, rhoK=rhoK,
            Delta=self.Delta[...],
            log_sigma=self.log_sigma[...]
        )
