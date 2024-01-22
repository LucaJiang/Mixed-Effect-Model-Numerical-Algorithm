# Use EM + MFVI to solve linear mixed model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set_style("whitegrid")


def lmm_em_mfvi(y, X, Z, tol=1e-3, max_iter=10, verbose=True):
    """
    Input:
    y: n x 1, response
    X: n x p, random effect
    Z: n x c, fixed effect
    tol: float, tolerance for convergence, default 1e-3
    max_iter: int, maximum number of iterations, default 10
    verbose: bool, print information, default True

    Output:
    elbo_list: elbo in each iteration
    omega_list: value x (iter + 1), record omega in each iteration
    sigma_beta2_list: record sigma_beta^2 in each iteration
    sigma_e2_list: record sigma_e^2 in each iteration
    beta_post_mean: posterior mean of beta

    Notation:
    omega: c x 1, fixed effect coefficient
    beta: p x 1, random effect coefficient

    Formula and prior distribution:
    y = Z * omega + X * beta + e
    e ~ N(0, sigma_e^2 * I_n)
    beta ~ N(0, sigma_beta^2 * I_p)

    Initialization:
    omega = (Z^T * Z)^-1 * Z^T * y
    sigma_beta^2 = var(y - Z * omega) / 2
    sigma_e^2 = var(y - Z * omega) / 2

    Implementation:
    E-step:
    sigma_j^2 = (X_j^T * X_j) / sigma_e^2 + 1 / sigma_beta^2
    mu_j = sigma_j^2 / sigma_e^2 * X_j^T * (y - X * (mu - mu_j) - X_j * mu_j)

    ELBO:
    elbo = -n / 2 * log(2 * pi * sigma_e^2) - 1 / (2 * sigma_e^2) * (y - Z * omega - X * mu)^T * (y - Z * omega - X * mu) - p / 2 * log(2 * pi * sigma_beta^2) - 1 / (2 * sigma_beta^2) * mu^T * mu + sum(log(2 * pi * sigma_j^2)) / 2

    M-step:
    omega = (Z^T * Z)^-1 * Z^T * (y - X * mu)
    sigma_beta^2 = (sum(sigma_j^2) + mu^T * mu) / p
    sigma_e^2 = (y - Z * omega - X * mu)^T * (y - Z * omega - X * mu) / n
    """
    n, p = X.shape
    n_, c = Z.shape
    assert n == n_, "X and Z must have same number of rows"
    assert y.shape == (n, 1), "y must be a column vector with length n"

    # Initialize parameters
    mu = np.random.randn(p, 1)
    omega = np.linalg.inv(Z.T @ Z) @ Z.T @ y
    y_z_omega = y - Z @ omega
    sigma_beta2 = np.var(y_z_omega) / 2
    sigma_e2 = sigma_beta2
    sigma_j2 = np.ones((p, 1))

    # For accelerate calculation
    XTX = X.T @ X
    ZTZinvZT = np.linalg.inv(Z.T @ Z) @ Z.T

    def cal_mu():
        ## update mu one by one
        r = y_z_omega - X @ mu
        for j in range(p):
            Xj = X[:, j]
            rj = r + (Xj * mu[j]).reshape(-1, 1)
            mu[j] = sigma_j2[j] / sigma_e2 * (Xj @ rj)
            r = rj - (Xj * mu[j]).reshape(-1, 1)
        return mu

    def cal_sigma_beta2():
        return (np.sum(sigma_j2) + (mu.T @ mu).item()) / p

    def cal_sigma_e2():
        return (
            np.linalg.norm(y_z_omega - X @ mu) ** 2
            # + np.trace(XTX @ np.diagflat(sigma_j2))
            + np.sum(np.diag(XTX) * sigma_j2)  # tr(XTX @ sigma_j2)
        ) / n

    def cal_elbo():
        # return 0
        # Q_function = (
        #     -n / 2 * np.log(2 * np.pi * sigma_e2)
        #     - sigma_e2_new * n / (2 * sigma_e2)
        #     - p / 2 * np.log(2 * np.pi * sigma_beta2)
        #     - sigma_beta2_new * p / (2 * sigma_beta2)
        # )
        # return Q_function + np.sum(np.log(2 * np.pi * sigma_j2)) / 2
        return -n / 2 * np.log(2 * np.pi * sigma_e2) - sigma_e2_new * n / (2 * sigma_e2)

    # Record parameters in each iteration
    max_iter += 1
    elbo_list = np.zeros(max_iter)
    omega_list = np.zeros((c, max_iter))
    sigma_beta2_list = np.zeros(max_iter)
    sigma_e2_list = np.zeros(max_iter)

    # Initialize record
    iter = 0
    omega_list[:, iter] = omega.flatten()
    elbo_list[iter] = 0
    sigma_beta2_list[iter] = sigma_beta2
    sigma_e2_list[iter] = sigma_e2

    # EM algorithm
    print("EM algorithm with MFVI starts")
    convergence = False
    for iter in range(1, max_iter):
        # E step
        sigma_j2 = 1 / (np.diag(XTX) / sigma_e2 + 1 / sigma_beta2).reshape(-1, 1)
        mu = cal_mu()

        # ELBO
        sigma_beta2_new = cal_sigma_beta2()
        sigma_e2_new = cal_sigma_e2()
        elbo_list[iter] = cal_elbo()

        # M step
        sigma_beta2 = sigma_beta2_new
        sigma_e2 = sigma_e2_new
        omega = ZTZinvZT @ (y - X @ mu)
        y_z_omega = y - Z @ omega

        # Record parameters
        omega_list[:, iter] = omega.flatten()
        sigma_beta2_list[iter] = sigma_beta2
        sigma_e2_list[iter] = sigma_e2

        # Check convergence
        if np.abs(elbo_list[iter] - elbo_list[iter - 1]) < tol:
            convergence = True
            break

        # Print process information
        if verbose and iter % 10 == 0:
            print("iter: {}, elbo: {:.4e}".format(iter, elbo_list[iter]))
            print("beta: {:.4e}".format(np.mean(mu)))
            print("sigma_beta^2: {:.4e}".format(sigma_beta2))
            print("sigma_e^2: {:.4e}".format(sigma_e2))
            print("--------------------------------------")

    # Algorithm summary
    beta_post_mean = np.mean(mu)
    resident = np.linalg.norm(y - Z @ omega - X @ mu) ** 2 / n
    if not convergence:
        print("EM algorithm does not converge within {} iterations".format(iter))
    else:
        print("EM algorithm converges after {} iterations".format(iter))
    print("sigma_beta^2 = {:.4e}".format(sigma_beta2))
    print("sigma_e^2 = {:.4e}".format(sigma_e2))
    print("beta_post_mean = {:.4e}".format(beta_post_mean))
    print("omega_mean = {:.4e}".format(np.mean(omega)))
    print("resident = {:.4e}".format(resident))

    return (
        elbo_list[: iter + 1],
        omega_list[:, : iter + 1],
        sigma_beta2_list[: iter + 1],
        sigma_e2_list[: iter + 1],
        beta_post_mean,
    )


def visual_em(
    elbo_list, sigma_beta2_list, sigma_e2_list, omega_list, beta_post_mean, img_name
):
    # subplot
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # omit the first point
    axes[0, 0].plot(range(1, len(elbo_list)), elbo_list[1:])
    axes[0, 0].set_xlabel("Iteration")
    # set x axis is integer
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_title("ELBO")

    axes[0, 1].plot(sigma_beta2_list, label=r"$\sigma_\beta^2$")
    axes[0, 1].plot(sigma_e2_list, label=r"$\sigma_e^2$")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_title("Unknown Variance")
    axes[0, 1].legend()

    axes[1, 0].plot(omega_list.T)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_title(r"$\omega$")

    # hist
    # sns.distplot(omega_list[:, -1], ax=axes[1, 1], label=r'$\omega$')
    sns.histplot(omega_list[:, -1], ax=axes[1, 1], label=r"$\omega$")
    axes[1, 1].axvline(
        beta_post_mean,
        color="r",
        linestyle="--",
        label=rf"$\beta={beta_post_mean:.2e}$",
    )
    axes[1, 1].set_title("Effects")
    axes[1, 1].legend()

    plt.suptitle("EM Algorithm with MFVI for Linear Mixed Model")
    plt.tight_layout()
    plt.savefig("img/lmm_em_mfvi" + img_name + ".png")
    plt.show()


if __name__ == "__main__":
    # load data
    # data_name = "fake_data"
    data_name = "XYZ_MoM"
    data = pd.read_table("data/" + data_name + ".txt", sep="\t", header=0).values

    y = data[:, 0].reshape(-1, 1)
    Z = data[:, 1:31]
    X = data[:, 31:]

    # run EM algorithm
    start_time = time.time()
    # run EM algorithm
    (
        elbo_list,
        omega_list,
        sigma_beta2_list,
        sigma_e2_list,
        beta_post_mean,
    ) = lmm_em_mfvi(y, X, Z, tol=1e-3, max_iter=200)
    end_time = time.time()
    print(
        "Run time: %d min %.2f s"
        % ((end_time - start_time) // 60, (end_time - start_time) % 60)
    )

    visual_em(
        elbo_list,
        sigma_beta2_list,
        sigma_e2_list,
        omega_list,
        beta_post_mean,
        data_name,
    )
