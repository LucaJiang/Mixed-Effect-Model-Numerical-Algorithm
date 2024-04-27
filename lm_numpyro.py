import os, time, argparse
import jax

import numpy as np
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.optim as optim
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoDelta


def generate_data(n, p, sigma_beta2, sigma_epsilon2):
    """
    Y=X*beta+epsilon
    """
    X = np.random.randn(n, p)
    # mu = np.random.randn(p, 1)
    beta = np.random.randn(p, 1) * sigma_beta2**0.5  # + mu
    epsilon = np.random.randn(n, 1) * sigma_epsilon2**0.5
    Y = X @ beta + epsilon
    print("beta true mean: ", beta.mean())
    return dict(X=X, Y=Y, beta=beta, epsilon=epsilon)


def lm(X, Y):
    """
    X: random effects, (n, p)
    Y: response, (n, )
    Formula: Y = X*beta + epsilon
    Distribution:
            beta_i ~iid N(mu_i, sigma_beta^2)
            epsilon ~iid N(0, sigma_epsilon^2)
    Inference:
        beta_post_mean: posterior mean of beta
        sigma_beta2_post: posterior variance of beta
        sigma_epsilon2_post: posterior variance of epsilon
    """
    n, p = X.shape
    # beta
    # mu = numpyro.sample("mu", dist.Normal(0, 1).expand([p, 1]))
    sigma_beta2 = numpyro.sample("sigma_beta2", dist.HalfNormal(1))
    # beta = numpyro.sample(
    #     "beta",
    #     dist.MultivariateNormal(jnp.zeros((p,)), jnp.sqrt(sigma_beta2) * jnp.eye(p)),
    # )
    beta = numpyro.sample("beta", dist.Normal(0, jnp.sqrt(sigma_beta2)).expand([p, 1]))
    # beta = numpyro.sample("beta", dist.Normal(mu, jnp.sqrt(sigma_beta2)))
    # epsilon
    sigma_epsilon2 = numpyro.sample("sigma_epsilon2", dist.HalfNormal(0.5))
    # print("beta: ", beta.shape)
    # print("X: ", X.shape)
    # deterministic statement
    y_hat = numpyro.deterministic("y_hat", X @ beta)
    numpyro.sample("Y", dist.Normal(y_hat, jnp.sqrt(sigma_epsilon2)), obs=Y)


def lm_simple(X, Y):
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1], 1]))
    mu = numpyro.deterministic("mu", X @ beta)
    numpyro.sample("Y", dist.Normal(mu, 1), obs=Y)


def run_inference(X, Y, rng_key, method, args):
    model = lm
    if method == "mcmc":
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(rng_key, X, Y)
        # mcmc.print_summary()
        return mcmc.get_samples()
    elif method == "svi" or method == "map":
        guide = AutoDelta(model) if method == "map" else AutoNormal(model)
        optimizer = optim.Adam(0.001)
        svi = SVI(model, guide, optimizer, Trace_ELBO(), X=X, Y=Y)
        svi_result = svi.run(rng_key, args.maxiter)
        return svi_result
    else:
        raise NotImplementedError


def plot_results(method, result, start_time, end_time, args):
    if method == "mcmc":
        # get posterior mean of beta
        beta_post_mean = result["beta"].mean(axis=1)
        # get posterior variance of beta
        sigma_beta2_post = result["sigma_beta2"]
        # get posterior variance of epsilon
        sigma_epsilon2_post = result["sigma_epsilon2"]
        title = "MCMC with NUTS: %.3f s" % (end_time - start_time)
        # plot
        plt.figure(figsize=(6, 4))
        plt.plot(beta_post_mean, label="beta")
        plt.plot(sigma_beta2_post, "r", label="sigma_beta2")
        plt.hlines(
            args.sigma_beta2,
            0,
            len(beta_post_mean),
            colors="r",
            linestyles="dashed",
            label="sigma_beta2_true",
        )
        plt.plot(sigma_epsilon2_post, "y", label="sigma_epsilon2")
        plt.hlines(
            args.sigma_epsilon2,
            0,
            len(sigma_epsilon2_post),
            colors="y",
            linestyles="dashed",
            label="sigma_epsilon2_true",
        )
        plt.xlabel("Iterations")
        plt.ylabel("posterior value")
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        file_name = "img/lm_numpyro_%s.png" % method
        plt.savefig(file_name)
    else:
        params = result.params
        beta_post_mean = params["beta_auto_loc"]
        sigma_beta2_post = params["sigma_beta2_auto_loc"]
        sigma_epsilon2_post = params["sigma_epsilon2_auto_loc"]
        elbo = result.losses
        plt.figure(figsize=(6, 4))
        plt.plot(elbo)
        plt.xlabel("Iterations")
        plt.ylabel("ELBO")
        plt.title("ELBO during training")
        plt.tight_layout()
        plt.savefig("img/lm_numpyro_%s_elbo.png" % method)

        title = (
            "SVI with AutoNormal: %.3f s" % (end_time - start_time)
            if method == "svi"
            else "MAP with AutoDelta: %.3f s" % (end_time - start_time)
        )

    plt.show()

    # print results
    print("beta_post_mean: %.4f" % beta_post_mean.mean())
    print(
        "sigma_beta2_post: %.4f, true value: %.4f"
        % (sigma_beta2_post.mean(), args.sigma_beta2)
    )
    print(
        "sigma_epsilon2_post: %.4f, true value: %.4f"
        % (sigma_epsilon2_post.mean(), args.sigma_epsilon2)
    )


def main(args):
    # load data

    data = generate_data(
        args.num_samples, args.dim, args.sigma_beta2, args.sigma_epsilon2
    )

    y = jnp.array(data["Y"])
    X = jnp.array(data["X"])

    # method = "mcmc"
    # method = "svi"
    method = "map"

    # record time
    start_time = time.time()
    rng_key = random.PRNGKey(args.seed)
    result = run_inference(X, y, rng_key, method, args)
    end_time = time.time()
    plot_results(method, result, start_time, end_time, args)
    print(
        "Calculation %s finished, time elapsed %.2f s."
        % (method, end_time - start_time)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear Regression")
    parser.add_argument("--seed", nargs="?", default=42, type=int)
    parser.add_argument("-n", "--num-samples", nargs="?", default=100, type=int)
    parser.add_argument("-p", "--dim", nargs="?", default=10, type=int)
    parser.add_argument("--sigma-beta2", nargs="?", default=2, type=float)
    parser.add_argument("--sigma-epsilon2", nargs="?", default=0.01, type=float)
    parser.add_argument("--num-warmup", nargs="?", default=20, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--maxiter", nargs="?", default=5000, type=int)
    parser.add_argument("--train-size", nargs="?", default=0.8, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    np.random.seed(args.seed)

    main(args)
