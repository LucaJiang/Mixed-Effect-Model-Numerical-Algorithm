from math import e
import os, time, argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoDiagonalNormal, AutoDelta


def lmm(Z, X, Y):
    """
    Z: fixed effects, (n, c)
    X: random effects, (n, p)
    Y: response, (n, )
    Formula: Y = Z*omega + X*beta + epsilon
    Distribution:
            beta ~ N(0, sigma_beta^2 I_p)
            epsilon ~ N(0, sigma_epsilon^2 I_n)
    Inference:
        omega_post_mean: posterior mean of omega
        sigma_beta2_post: posterior variance of beta
        sigma_epsilon2_post: posterior variance of epsilon
    """
    n, p = X.shape
    _, c = Z.shape
    # sample from distribution
    omega_prior = jnp.linalg.inv(Z.T @ Z) @ Z.T @ Y
    omega = numpyro.sample(
        "omega",
        dist.Normal(omega_prior, jnp.ones((c, 1))),
    )
    sigma_beta2 = numpyro.sample("sigma_beta2", dist.HalfNormal(1))
    beta = numpyro.sample("beta", dist.Normal(0, sigma_beta2**0.5).expand([p, 1]))
    sigma_epsilon2 = numpyro.sample("sigma_epsilon2", dist.HalfNormal(1))
    # deterministic statement
    # print("omega: ", omega.shape)
    # print("omega_prior: ", omega_prior.shape)
    # print("beta: ", beta.shape)
    # print("Z: ", Z.shape)
    # print("X: ", X.shape)
    mu = numpyro.deterministic("y_hat", Z @ omega + X @ beta)
    numpyro.sample("Y", dist.Normal(mu, sigma_epsilon2**0.5), obs=Y)


def run_inference(Z, X, Y, rng_key, method, args):
    if method == "mcmc":
        kernel = NUTS(lmm)
        mcmc = MCMC(
            kernel,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(rng_key, Z, X, Y)
        # mcmc.print_summary()
        return mcmc.get_samples()
    elif method == "svi" or method == "map":
        guide = AutoDelta(lmm) if method == "svi" else AutoDiagonalNormal(lmm)
        optimizer = numpyro.optim.Adam(0.001)
        svi = SVI(lmm, guide, optimizer, Trace_ELBO(), Z=Z, X=X, Y=Y)
        svi_result = svi.run(rng_key, args.maxiter)
        return svi_result
    else:
        raise NotImplementedError


def main(args):
    # load data
    data_name = "fake_data"  # 2,2,0.01
    # data_name = "XYZ_MoM"
    data = pd.read_table("data/" + data_name + ".txt", sep="\t", header=0).values

    y = jnp.array(data[:, 0]).reshape(-1, 1)
    Z = jnp.array(data[:, 1:31])
    X = jnp.array(data[:, 31:])

    # method = "mcmc"
    # method = "svi"
    method = "map"

    # record time
    start_time = time.time()
    rng_key = random.PRNGKey(1)
    result = run_inference(Z, X, y, rng_key, method, args)
    end_time = time.time()
    print(
        "Calculation %s finished, time elapsed %.2f s."
        % (method, end_time - start_time)
    )

    # get posterior samples
    if method == "mcmc":
        # get posterior mean of omega
        omega_post_mean = result["omega"].mean(axis=1)
        # get posterior variance of beta
        sigma_beta2_post = result["sigma_beta2"]
        # get posterior variance of epsilon
        sigma_epsilon2_post = result["sigma_epsilon2"]
        # plot
        plt.plot(omega_post_mean, label="omega")
        plt.plot(sigma_beta2_post, label="sigma_beta2")
        plt.plot(sigma_epsilon2_post, label="sigma_epsilon2")
        plt.xlabel("Iterations")
        plt.ylabel("posterior value")
        plt.legend()
        plt.title("MCMC with NUTS: %.3f s" % (end_time - start_time))
        plt.tight_layout()
        plt.savefig("lmm_numpyro_mcmc.png")
        plt.show()

    elif method == "svi" or method == "map":
        params = result.params
        omega_post_mean = params["omega_auto_loc"]
        sigma_beta2_post = params["sigma_beta2_auto_loc"]
        sigma_epsilon2_post = params["sigma_epsilon2_auto_loc"]
        elbo = result.losses
        plt.plot(elbo, label="ELBO")
        plt.plot(omega_post_mean, label="omega")
        plt.plot(sigma_beta2_post, label="sigma_beta2")
        plt.plot(sigma_epsilon2_post, label="sigma_epsilon2")
        plt.xlabel("Iterations")
        plt.ylabel("ELBO")
        plt.legend()
        title = (
            lambda method, time: "SVI with AutoDelta: %.3f s" % time
            if method == "svi"
            else "MAP with AutoDiagonalNormal: %.3f s" % time
        )
        plt.title(title(method, end_time - start_time))
        plt.tight_layout()
        plt.savefig("lmm_numpyro_%s.png" % method)
        plt.show()

    # print results
    print("omega_post_mean: %.4f" % omega_post_mean.mean())
    print("sigma_beta2_post: %.4f" % sigma_beta2_post.mean())
    print("sigma_epsilon2_post: %.4f" % sigma_epsilon2_post.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Zero-Inflated Poisson Regression")
    parser.add_argument("--seed", nargs="?", default=42, type=int)
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=20, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--maxiter", nargs="?", default=5000, type=int)
    parser.add_argument("--train-size", nargs="?", default=0.8, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
