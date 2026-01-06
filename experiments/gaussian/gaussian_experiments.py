import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from flax import nnx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax 
from models.gaussian.vb import VB
from trainers.gaussian.vb_trainer import VB_trainer
from models.gaussian.hvae import HVAE
from trainers.gaussian.hvae_trainer import HVAE_trainer
from models.gaussian.iwhvae import IWHVAE
from trainers.gaussian.iwhvae_trainer import IWHVAE_trainer
from tqdm import tqdm
import json

def generate_data(key, dim, n_data, n_test, Delta_gt, sigma_gt, ):
    def generate_one_dataset(key):
        subkey1, subkey2 = jr.split(key)
        z = jr.normal(subkey1, shape = (dim,))
        noise = jr.normal(subkey2, shape=(n_data, dim))
        data = (z + Delta_gt) + sigma_gt * noise
        return data
    keys = jr.split(key, n_test)
    datasets = jax.vmap(generate_one_dataset)(keys)
    return datasets



def main(dim, n_data, n_test, n_iter, rngs):
    # Generate syntetic data
    num_range = jnp.arange(-(dim-1)/2, (dim+1)/2, dtype=jnp.float32)
    Delta_gt = num_range/5
    if dim == 1 : 
        sigma_gt = jnp.array([1.])
    else : 
        sigma_gt = 36/(10*(dim-1)**2) * num_range**2 + 0.1 

    datasets = generate_data(rngs.param(), dim, n_data, n_test, Delta_gt, sigma_gt)

    # Initialization parameters
    init_eps = 0.005 * jnp.ones(dim)
    max_eps = 0.5 * jnp.ones(dim)
    init_logit_eps = jnp.log(init_eps/(max_eps - init_eps))

    init_logit_beta0 = 1/jnp.log(4.)

    init_Delta = jnp.zeros(shape =(dim,))
    init_log_sigma = 3* jnp.ones(shape = (dim,))

    init_mu_z = jnp.zeros(shape = (dim,))
    init_log_sigma_z = jnp.ones(shape = (dim,))

    # Create model :
    init_params_vb = {
        "Delta" : init_Delta,
        "log_sigma" : init_log_sigma,
        "mu_z" : init_mu_z,
        "log_sigma_z" : init_log_sigma_z
    }
    init_params_hvae = {
        "Delta" : init_Delta,
        "log_sigma" : init_log_sigma,
        "logit_eps" : init_logit_eps,
        "logit_beta0" : init_logit_beta0
    }
    VB_model = VB(dim = dim, param_init=init_params_vb)
    HVAE1_model = HVAE(dim = dim, K = 1, param_init = init_params_hvae)
    HVAE10_model = HVAE(dim = dim, K = 10, param_init = init_params_hvae)
    IWHVAE_model = IWHVAE(dim = dim, K = 1, L = 10, param_init = init_params_hvae)

    # Create optimizers and trainers for VB, HVAE1, and HVAE10
    VB_optimizer = nnx.Optimizer(VB_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    VB_trainer_instance = VB_trainer(VB_model, VB_optimizer)

    HVAE1_optimizer = nnx.Optimizer(HVAE1_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    HVAE1_trainer_instance = HVAE_trainer(HVAE1_model, HVAE1_optimizer)

    HVAE10_optimizer = nnx.Optimizer(HVAE10_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    HVAE10_trainer_instance = HVAE_trainer(HVAE10_model, HVAE10_optimizer)

    IWHVAE_optimizer = nnx.Optimizer(IWHVAE_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    IWHVAE_trainer_instance = HVAE_trainer(IWHVAE_model, IWHVAE_optimizer)

    # Initialize accumulators for mean errors of all models
    out = {
        "Delta_gt" : Delta_gt, 
        "sigma_gt" : sigma_gt,
        "VB": {
            "Delta" : [],
            "sigma" : [],
        },
        "HVAE1": {
            "Delta" : [],
            "sigma" : [],
        },
        "HVAE10": {
            "Delta" : [],
            "sigma" : [],
        },
        "IWHVAE" : {
            "Delta" : [], 
            "sigma" : [],
        }
    }

    for i in range(n_test):
        print(f"Executing test {i}/{n_test} of dimension {dim}")

        # Train VB model
        for j in tqdm(range(n_iter), desc="VB"):
            val, grad = VB_trainer_instance.train_step(VB_model, VB_optimizer, datasets[i], rngs)
        out["VB"]["Delta"].append(VB_model.Delta[...])
        out["VB"]["sigma"].append(jnp.exp(VB_model.log_sigma[...]))

        # Train HVAE1 model
        for j in tqdm(range(n_iter), desc="HVAE1"):
            val, grad = HVAE1_trainer_instance.train_step(HVAE1_model, HVAE1_optimizer, datasets[i], rngs)
        out["HVAE1"]["Delta"].append(HVAE1_model.Delta[...])
        out["HVAE1"]["sigma"].append(jnp.exp(HVAE1_model.log_sigma[...]))

        # Train HVAE10 model
        for j in tqdm(range(n_iter), desc="HVAE10"):
            val, grad = HVAE10_trainer_instance.train_step(HVAE10_model, HVAE10_optimizer, datasets[i], rngs)
        out["HVAE10"]["Delta"].append(HVAE10_model.Delta[...])
        out["HVAE10"]["sigma"].append(jnp.exp(HVAE10_model.log_sigma[...]))

        # Train IWHVAE model
        for j in tqdm(range(n_iter), desc="IWHVAE"):
            val, grad = IWHVAE_trainer_instance.train_step(IWHVAE_model, IWHVAE_optimizer, datasets[i], rngs)
        out["IWHVAE"]["Delta"].append(IWHVAE_model.Delta[...])
        out["IWHVAE"]["sigma"].append(jnp.exp(IWHVAE_model.log_sigma[...]))
    return out

def to_json_ready(obj):
    """Recursively converts JAX/NumPy arrays to Python lists."""
    if isinstance(obj, dict):
        return {k: to_json_ready(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_ready(x) for x in obj]
    elif hasattr(obj, "tolist"):  # Matches JAX arrays and NumPy arrays
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj


if __name__ == "__main__":

    n_test = 1
    n_data = 10000
    n_iter = 30000
    dimensions = [1, 2, 3, 5, 11, 25, 51, 101, 201, 301]

    rngs = nnx.Rngs(0)

    results = {
        "dimensions" : dimensions,
        "Delta_gt" : [], 
        "sigma_gt" : [], 
        "VB": {
        },
        "HVAE1": {
        },
        "HVAE10": {
        },
        "IWHVAE":{
        }
    }

    for dim in dimensions : 
        out = main(dim, n_data, n_test, n_iter, rngs)
        results["Delta_gt"].append(out["Delta_gt"])
        results["sigma_gt"].append(out["sigma_gt"])
        for key in results.keys():
            results["VB"][str(dim)] = out["VB"]
            results["HVAE1"][str(dim)] = out["HVAE1"]
            results["HVAE10"][str(dim)] = out["HVAE10"]
            results["IWHVAE"][str(dim)] = out["IWHVAE"]

    with open('experiments/gaussian/results.json', 'w') as f:
        json.dump(to_json_ready(results), f)
