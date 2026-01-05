import jax 
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.scipy.stats import multivariate_normal
from typing import NamedTuple

MAX_EPS = 0.5

class HVAEOutput(NamedTuple):
    z0: jnp.ndarray
    rho0: jnp.ndarray
    zK: jnp.ndarray
    rhoK: jnp.ndarray
    Delta: jnp.ndarray
    log_sigma: jnp.ndarray


class HVAE(nnx.Module):

    def __init__(self, dim: int, K: int, param_init: dict, tempering : bool = True ):
        self.dim = dim
        self.tempering = tempering
        self.K = K
        
        # Model's parameters
        self.Delta = nnx.Param(param_init["Delta"])
        self.log_sigma = nnx.Param(param_init["log_sigma"])
        self.logit_eps = nnx.Param(param_init["logit_eps"])
        if tempering : 
            self.logit_beta0 = nnx.Param(param_init["logit_beta0"])

   
    def his(
        self, 
        n_data: int, 
        x_bar: jnp.ndarray, 
        rngs: nnx.Rngs
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Algorithm 1 in "Hamiltonian Variational Autoencoders", integrate a trajectory of hamiltonian dynamics

            Args:
                n_data (int): number of data points
                x_bar (jnp.ndarray): barycenter of the dataset
                rngs (nnx.Rngs): Random number generator

            Returns:
                tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: z0, rho0, z_K, rho_K
            """
        logit_eps = self.logit_eps[...]
        epsilon = jax.nn.sigmoid(logit_eps)*MAX_EPS
        if self.tempering: 
            logit_beta0 = self.logit_beta0[...]
            beta0 = jax.nn.sigmoid(logit_beta0)
        else : 
            beta0 = jnp.array(1.)

        # Initialize values
        key_z, key_gamma = jr.split(rngs.params())
        z0 = jr.normal(key_z, (self.dim,))
        gamma0 = jr.normal(key_gamma, (self.dim,))
        rho0 = gamma0/jnp.sqrt(beta0)



        def his_step(carry, x):
            # Unwrap carry : 
            rho = carry["rho"]
            z = carry["z"]
            sqrt_beta = carry["sqrt_beta"]
            iteration = carry["iteration"]
            
            # Leapfrog integration :
            rho_tilde = rho - 0.5*epsilon*self.grad_U(z, n_data, x_bar)
            z = z + epsilon*rho_tilde
            rho_prime = rho_tilde - 0.5*epsilon*self.grad_U(z, n_data, x_bar)
            # Quadratic tempering
            if self.tempering : 
                sqrt_beta_new = 1/((1-1/jnp.sqrt(beta0))*iteration**2/self.K**2+1/jnp.sqrt(beta0))
                rho = sqrt_beta/sqrt_beta_new*rho_prime
            else : 
                sqrt_beta_new = jnp.array(1.)
                rho = rho_prime

            carry = {
                "rho" : rho, 
                "z" : z, 
                "sqrt_beta" : sqrt_beta_new, 
                "iteration" : iteration +1
            }
            return carry, z
        
        init = {
            "rho" : rho0,
            "z" : z0, 
            "sqrt_beta" : jnp.sqrt(beta0), 
            "iteration" : 1
        }
        # Integrate Hamiltonian equation using jax.lax.scan
        carry, zs = jax.lax.scan(his_step, init, jnp.arange(self.K))
        zK, rhoK = carry["z"], carry["rho"] 
        return z0, rho0, zK, rhoK, zs

    def U(self, z : jnp.ndarray, data : jnp.ndarray)->jnp.ndarray:
        """Computes the energy of the model at a given position

        Args:
            z (jnp.ndarray): letent variable
            data (jnp.ndarray): dataset

        Returns:
            jnp.ndarray: energy U(z|D)
        """
        energy = jnp.sum(jax.vmap(lambda x : multivariate_normal.logpdf(x, z + self.Delta[...], jnp.diag(jnp.exp(2*self.log_sigma[...]))))(data))+ multivariate_normal.logpdf(z, jnp.zeros(self.dim), jnp.eye(self.dim))
        return energy


    def grad_U(self, z, n_data, x_bar):
        """Computes the gradient of the energy of the Hamiltonian system (page 7 "Hamiltonian Variational Autoencoders")

        Args:
            z (jnp.ndarray): state of the system in the latent space
            n_data (int): number of data points
            x_bar (jnp.ndarray): barycenter of the dataset

        Returns:
            jnp.array : gradient of the energy at the current state
        """
        Delta = self.Delta[...]
        log_sigma = self.log_sigma[...]
        
        grad_U = z + n_data*(z + Delta - x_bar)/jnp.exp(2*log_sigma)
        return grad_U


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
        return HVAEOutput(
            z0=z0, rho0=rho0, zK=zK, rhoK=rhoK,
            Delta=self.Delta[...],
            log_sigma=self.log_sigma[...]
        )








if __name__ == "__main__":
    latent_dim = 2
    num_samples = 10
    key = jr.PRNGKey(42)
    
    dummy_data = jr.normal(key, (num_samples, latent_dim))

    init_params = {
        "Delta": jnp.zeros(latent_dim), 
        "log_sigma": jnp.zeros(latent_dim), 
        "logit_eps": jnp.array(0.0), 
        "logit_beta0": jnp.array(0.0)
    }
    

    model = HVAE(dim=latent_dim, K=5, param_init=init_params)
    rngs = nnx.Rngs(42)
    
    print("--- Model Structure ---")
    nnx.display(model)
    


    def test_step(model, data, rngs):
        return model.elbo(data, rngs)
        
    elbo_val = test_step(model, dummy_data, rngs)
    print(f"\n--- Compilation Successful ---")
    print(f"ELBO value: {elbo_val}")
    