from flax import nnx
import jax 
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from models.gaussian.vb import VB


class VB_trainer():
    def __init__(self, model: VB, optimizer: nnx.Optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def compute_loss(self, model: VB, data: jnp.ndarray, rngs: nnx.Rngs) -> jnp.ndarray:
        """Compute the loss which is the inverse of the ELBO for VB

        Args:
            model (VB): model to train
            data (jnp.ndarray): dataset
            rngs (nnx.Rngs): random number generator

        Returns:
            jnp.ndarray: inverse of the ELBO
        """
        dim = model.dim
        out = model(rngs)

        # log p(D,z_K)
        term1 = jnp.sum(jax.vmap(lambda x : multivariate_normal.logpdf(x, out.z + out.Delta, jnp.diag(jnp.exp(2*out.log_sigma))))(data))+ multivariate_normal.logpdf(out.z, jnp.zeros(dim), jnp.eye(dim))
        # log N(z|\mu_z, \Sigma_z)
        term2 = multivariate_normal.logpdf(out.z, out.mu_z, jnp.diag(jnp.exp(2*out.log_sigma_z))) 

        elbo = term1 - term2

        # Return the oposite of ELBO to maximize ELBO
        return -elbo

    @nnx.jit(static_argnames='self')
    def train_step(
        self,
        model: VB,
        optimizer: nnx.Optimizer,
        data: jax.Array,
        rngs: nnx.Rngs
    ) -> tuple[jax.Array, nnx.State]:
        """Compute one step of optimizing the loss function with respect to the model's parameters

        Args:
            model (HVAE): model to train
            optimizer (nnx.Optimizer): optimizer used to train 
            data (jax.Array): dataset
            rngs (nnx.Rngs): random number generator

        Returns:
            tuple[jax.Array, nnx.State]: value and gradients of the loss function at the current step
        """
        val_and_grad_fn = nnx.value_and_grad(self.compute_loss)
        val, grads = val_and_grad_fn(model, data, rngs)
        optimizer.update(model, grads)
        return val, grads
 
