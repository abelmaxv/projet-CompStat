import sys
import os
from pathlib import Path
from unittest import result

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

    # Create model :
    init_params_vb = {
        "Delta" : jnp.zeros(shape = (dim,)),
        "log_sigma" : jnp.ones(shape = (dim,)),
        "mu_z" : jnp.zeros(shape = (dim,)),
        "log_sigma_z" : jnp.zeros(shape = (dim,))
    }
    init_params_hvae = {
        "Delta" : jnp.zeros(shape = (dim,)),
        "log_sigma" : jnp.ones(shape = (dim,)),
        "logit_eps" : jnp.zeros(shape = (dim,)),
        "logit_beta0" : jnp.zeros(shape = (1,))
    }
    VB_model = VB(dim = dim, param_init=init_params_vb)
    HVAE1_model = HVAE(dim = dim, K = 1, param_init = init_params_hvae)
    HVAE10_model = HVAE(dim = dim, K = 10, param_init = init_params_hvae)

    # Create optimizers and trainers for VB, HVAE1, and HVAE10
    VB_optimizer = nnx.Optimizer(VB_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    VB_trainer_instance = VB_trainer(VB_model, VB_optimizer)

    HVAE1_optimizer = nnx.Optimizer(HVAE1_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    HVAE1_trainer_instance = HVAE_trainer(HVAE1_model, HVAE1_optimizer)

    HVAE10_optimizer = nnx.Optimizer(HVAE10_model, optax.rmsprop(learning_rate=1e-3), wrt=nnx.Param)
    HVAE10_trainer_instance = HVAE_trainer(HVAE10_model, HVAE10_optimizer)

    # Initialize accumulators for mean errors of all models
    out = {
        "Delta_gt" : Delta_gt, 
        "Sigma_gt" : sigma_gt,
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
    return out




if __name__ == "__main__":

    n_test = 10
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
        }
    }

    for dim in dimensions : 
        out = main(dim, n_data, n_test, n_iter, rngs)
        for key in results.keys():
            results["VB"][str(dim)] = out["VB"]
            results["HVAE1"][str(dim)] = out["HVAE1"]
            results["HVAE10"][str(dim)] = out["HVAE10"]

    with open('experiments/gaussian/results.json', 'w') as f:
        json.dump(results, f)
