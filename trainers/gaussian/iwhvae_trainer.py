from flax import nnx
import jax 
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from models.gaussian.iwhvae import IWHVAE


class IWHVAE_trainer():
    def __init__(self, model: IWHVAE, optimizer: nnx.Optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def compute_loss(self, model: IWHVAE, data: jnp.ndarray, rngs: nnx.Rngs) -> jnp.ndarray:
        """Compute the loss which is the inverse of the ELBO (given by eq (5) in "Hamiltonian Variational Autoencoders")

        Args:
            model (IWHVAE): model to train
            data (jnp.ndarray): dataset
            rngs (nnx.Rngs): random number generator

        Returns:
            jnp.ndarray: inverse of the ELBO
        """

        n_data = data.shape[0]
        x_bar = jnp.mean(data, axis=0)
        dim = model.dim
        L = model.L

        keys = jax.random.split(jax.random.key(0), L)
        graphdef, state = nnx.split(model)

        def run_model(key):
            # Re-create the Rngs object inside vmap for this specific trace
            rngs = nnx.Rngs(key) 
            m = nnx.merge(graphdef, state)
            return m(n_data, x_bar, rngs)


        outs = jax.vmap(run_model)(keys)

        def compute_single_elbo(single_out):
            """Computes ELBO of HVAE for a single prediction object."""
            
            # 1. Log p(D | z_K, Delta, sigma) + Log p(z_K)
            # We vmap over the data points for this specific 'single_out'
            log_probs = jax.vmap(lambda x: multivariate_normal.logpdf(
                x, 
                single_out.zK + single_out.Delta, 
                jnp.diag(jnp.exp(2 * single_out.log_sigma))
            ))(data)
            
            term1 = jnp.sum(log_probs) + multivariate_normal.logpdf(
                single_out.zK, jnp.zeros(dim), jnp.eye(dim)
            )

            # 2. Log priors/posteriors for rho and z
            term2 = multivariate_normal.logpdf(single_out.rhoK, jnp.zeros(dim), jnp.eye(dim))
            term3 = multivariate_normal.logpdf(single_out.z0, jnp.zeros(dim), jnp.eye(dim))
            term4 = multivariate_normal.logpdf(single_out.rho0, jnp.zeros(dim), jnp.eye(dim))

            return term1 + term2 - term3 - term4

        estimates = jnp.exp(jax.vmap(compute_single_elbo)(outs))

        elbo = jnp.log(jnp.mean(estimates))
        # Return the oposite of ELBO to maximize ELBO
        return -elbo

    @nnx.jit(static_argnames='self')
    def train_step(
        self,
        model: IWHVAE,
        optimizer: nnx.Optimizer,
        data: jax.Array,
        rngs: nnx.Rngs
    ) -> tuple[jax.Array, nnx.State]:
        """Compute one step of optimizing the loss function with respect to the model's parameters

        Args:
            model (IWHVAE): model to train
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
 
