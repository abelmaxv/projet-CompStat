from flax import nnx
import jax 
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal


class HVAE_trainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def compute_loss(self, model, data, rngs ):
        n_data = data.shape[0]
        x_bar = jnp.mean(data, axis=0)
        dim = model.dim

        out = model(n_data, x_bar, rngs)

         # log p(D,z_K)
        term1 = jnp.sum(jax.vmap(lambda x : multivariate_normal.logpdf(x, out.zK + out.Delta, jnp.diag(jnp.exp(2*out.log_sigma))))(data))
        # log N(\rho_K| 0, I)
        term2 = multivariate_normal.logpdf(out.rhoK, jnp.zeros(shape = (dim,)), jnp.eye(dim))
        # l/2*log(\beta_0) 
        term3 = 0.5*dim*jnp.log(out.beta0)
        # log N(z_0| 0, I)
        term4 = multivariate_normal.logpdf(out.z0, jnp.zeros(shape = (dim,)), jnp.eye(dim))
        # log N(\rho_0| 0, \beta_0^{-1}I)
        term5 = multivariate_normal.logpdf(out.rho0, jnp.zeros(shape = (dim,)), 1/out.beta0*jnp.eye(dim))

        elbo = term1 + term2 + term3 - term4 - term5

        # Return the oposite of ELBO to maximize ELBO
        return -elbo


    def train_step(self, data, rngs):
        grad_fn = nnx.grad(self.compute_loss)
        grads = grad_fn(self.model, data, rngs)
        self.optimizer.update(self.model, grads)
 
