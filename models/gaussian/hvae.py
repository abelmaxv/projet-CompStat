import jax 
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal

MAX_EPS = 0.5


class HVAE(nnx.Module):

    def __init__(self, dim, K, param_init):
        self.dim = dim
        self.K = K
        
        # Model's parameters
        self.Delta = nnx.Param(param_init["Delta"])
        self.log_sigma = nnx.Param(param_init["log_sigma"])
        self.logit_eps = nnx.Param(param_init["logit_eps"])
        self.logit_beta0 = nnx.Param(param_init["logit_beta0"])

        
    def his(self, n_data, x_bar, rngs):
        """Algorithm 1 in "Hamiltonian Variational Autoencoders", integrate a trajectory of hamiltonian dynamics

        Args:
            n_data (int): number of data points
            x_bar (jnp.array): barycenter of the dataset
            rngs (nnx.rnglib.Rngs): Random number generator

        Returns:
            tuple : z0, rho0, z_K, rho_K
        """
        logit_beta0 = self.logit_beta0[...]
        logit_eps = self.logit_eps[...]
        epsilon = jax.nn.sigmoid(logit_eps)*MAX_EPS
        beta0 = jax.nn.sigmoid(logit_beta0)

        # Initialize values
        z0 = jr.normal(rngs.params(), shape = (self.dim,))
        gamma0 = jr.normal(rngs.params(), shape = (self.dim,))
        rho0 = gamma0/jnp.sqrt(beta0)



        def his_step(carry, x):
            # Unwrap carry : 
            rho = carry["rho"]
            z = carry["z"]
            beta = carry["beta"]
            iteration = carry["iteration"]
            
            # Leapfrog integration :
            rho_tilde = rho - 0.5*epsilon*self.grad_U(z, n_data, x_bar)
            z = z + epsilon*rho_tilde
            rho_prime = rho_tilde - 0.5*epsilon*self.grad_U(z, n_data, x_bar)
            # Quadratic tempering
            beta_new = 1/((1-1/jnp.sqrt(beta0))*iteration**2/self.K**2+1/jnp.sqrt(beta0))**2
            rho = jnp.sqrt(beta/beta_new)*rho_prime

            carry = {
                "rho" : rho, 
                "z" : z, 
                "beta" : beta_new, 
                "iteration" : iteration +1
            }
            return carry, z
        
        init = {
            "rho" : rho0,
            "z" : z0, 
            "beta" : beta0, 
            "iteration" : 1
        }
        # Integrate Hamiltonian equation using jax.lax.scan
        carry, _ = jax.lax.scan(his_step, init, jnp.arange(self.K))
        zK, rhoK = carry["z"], carry["rho"] 
        return z0, rho0, zK, rhoK


    def grad_U(self, z, n_data, x_bar):
        """Computes the gradient of the energy of the Hamiltonian system (page 7 "Hamiltonian Variational Autoencoders")

        Args:
            z (jnp.array): state of the system in the latent space
            n_data (int): number of data points
            x_bar (jnp.array): barycenter of the dataset

        Returns:
            jnp.array : gradient of the energy at the current state
        """
        Delta = self.Delta[...]
        log_sigma = self.log_sigma[...]
        
        grad_U = z + n_data*(z+Delta - x_bar)/jnp.exp(2*log_sigma)
        return grad_U


    def elbo(self, data, rngs):
        """Computes the estimation of the ELBO

        Args:
            data (jnp.array): dataset
            rngs (nnx.rnglib.Rngs): random number generator

        Returns:
            jnp.array: estimator of the ELBO
        """
        n_data = data.shape[0]
        x_bar = jnp.mean(data, axis = 0)
        beta0 = jax.nn.sigmoid(self.logit_beta0.get_value())
        Delta = self.Delta[...]
        log_sigma = self.log_sigma[...]
        z0, rho0, zK, rhoK  = self.his(n_data, x_bar, rngs)

        # log p(D,z_K)
        term1 = jnp.sum(jax.vmap(lambda x : multivariate_normal.logpdf(x, zK + Delta, jnp.diag(jnp.exp(2*log_sigma))))(data))
        # log N(\rho_K| 0, I)
        term2 = multivariate_normal.logpdf(rhoK, jnp.zeros(shape = (self.dim,)), jnp.eye(self.dim))
        # l/2*log(\beta_0) 
        term3 = 0.5*self.dim*jnp.log(beta0)
        # log N(z_0| 0, I)
        term4 = multivariate_normal.logpdf(z0, jnp.zeros(shape = (self.dim,)), jnp.eye(self.dim))
        # log N(\rho_0| 0, \beta_0^{-1}I)
        term5 = multivariate_normal.logpdf(rho0, jnp.zeros(shape = (self.dim,)), 1/beta0*jnp.eye(self.dim))

        return term1 + term2 + term3 - term4 - term5








if __name__ == "__main__":
# 1. Setup dimensions and dummy data
    latent_dim = 2
    num_samples = 10
    key = jr.PRNGKey(42)
    
    # Generate dummy data (N samples of dimension latent_dim)
    dummy_data = jr.normal(key, (num_samples, latent_dim))
    
    # 2. Initialize parameters with correct shapes (using jnp arrays)
    # Delta and log_sigma should match the latent dimension
    init_params = {
        "Delta": jnp.zeros(latent_dim), 
        "log_sigma": jnp.zeros(latent_dim), 
        "logit_eps": jnp.array(0.0), 
        "logit_beta0": jnp.array(0.0)
    }
    
    # 3. Instantiate model and NNX RNGs
    # K=5 leapfrog steps for a quick check
    model = HVAE(dim=latent_dim, K=5, param_init=init_params)
    rngs = nnx.Rngs(42)
    
    print("--- Model Structure ---")
    nnx.display(model)
    
    # 4. Test ELBO computation


    def test_step(model, data, rngs):
        return model.elbo(data, rngs)
        
    elbo_val = test_step(model, dummy_data, rngs)
    print(f"\n--- Compilation Successful ---")
    print(f"ELBO value: {elbo_val}")
    