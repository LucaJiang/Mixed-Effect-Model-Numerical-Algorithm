# Use EM to solve linear mixed model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def lmm_em(y, X, Z, tol=1e-10, max_iter=100):
    '''
    Input:
    y: n x 1, response
    X: n x p, random effect
    Z: n x c, fixed effect
    tol: float, tolerance for convergence, default 1e-10
    max_iter: int, maximum number of iterations, default 100

    Output:
    likelihood_list: likelihood in each iteration
    omega_list: value x iter, record omega in each iteration
    sigma_beta2_list: record sigma_beta^2 in each iteration
    sigma_e2_list: record sigma_e^2 in each iteration
    beta_post_mean: posterior mean of beta

    Notation:
    omega: c x 1, fixed effect coefficient
    beta: p x 1, random effect coefficient

    Formula:
    y = Z * omega + X * beta + e
    e ~ N(0, sigma_e^2 * I)
    beta ~ N(0, sigma_beta^2 * I)

    Sigma = sigma_beta^2 * X * X^T + sigma_e^2 * I
    omega = (Z^T * Sigma^-1 * Z)^-1 * Z^T * Sigma^-1 * y
    # beta|omega = (X^T * X)^-1 * X^T * (y - Z * omega)
    beta = sigma_beta^2 * X^T * Sigma^-1 * (y - Z * omega)
    e = y - Z * omega - X * beta

    # sigma_beta^2 = (beta^T * beta + sigma_beta^2 * trace(?)) / p
    # sigma_e^2 = (e^T * e + sigma_e^2 * trace(?)) / n

    sigma_beta^2 = (||beta||_2^2 -mean(beta)^2) / p
    sigma_e^2 = (||e||_2^2 - mean(e)^2) / n
    '''
    n, p = X.shape
    n_, c = Z.shape
    assert n == n_, 'X and Z must have same number of rows'
    assert y.shape == (n, 1), 'y must be a column vector with length n'

    # Initialize parameters
    beta = np.zeros((p, 1))
    omega = np.zeros((c, 1))
    sigma_beta2 = 1
    sigma_e2 = 1

    # tmp
    XXT = X @ X.T
    # H_beta = np.linalg.inv(XXT.T) @ X.T


    # Calculate likelihood
    def likelihood(omega, sigma_beta2, sigma_e2):
        Sigma = sigma_beta2 * XXT + sigma_e2 * np.eye(n)
        Sigma_inv = np.linalg.inv(Sigma)
        tmp = y - Z @ omega
        likelihood = -0.5 * np.log(np.linalg.det(Sigma) + tol) - \
            0.5 * tmp.T @ Sigma_inv @ tmp
        return likelihood

    # Record parameters in each iteration
    likelihood_list = np.zeros(max_iter)
    omega_list = np.zeros((c, max_iter))
    sigma_beta2_list = np.zeros(max_iter)
    sigma_e2_list = np.zeros(max_iter)

    # Initialize record
    iter = 0
    max_iter += 1
    omega_list[:, iter] = omega.flatten()
    likelihood_list[iter] = likelihood(omega, sigma_beta2, sigma_e2)
    sigma_beta2_list[iter] = sigma_beta2
    sigma_e2_list[iter] = sigma_e2

    # EM algorithm
    for iter in range(1, max_iter):
        # E step
        Sigma = sigma_beta2 * XXT + sigma_e2 * np.eye(n)
        Sigma_inv = np.linalg.inv(Sigma)
        # beta = H_beta @ (y - Z @ omega)
        beta = sigma_beta2 * X.T @ Sigma_inv @ (y - Z @ omega)
        e = y - Z @ omega - X @ beta

        # M step
        tmp = Z.T @ Sigma_inv @ Z
        omega = np.linalg.inv(tmp) @ Z.T @ Sigma_inv @ y

        # sigma_beta2 = (beta.T @ beta + np.trace(beta))[0, 0] / p
        # sigma_e2 = (e.T @ e + sigma_e2 * np.trace(Sigma_inv))[0, 0] / n

        beta_flat = beta.flatten()
        sigma_beta2 = (np.linalg.norm(beta_flat)**2 -
                       np.mean(beta_flat)**2) / p
        sigma_e2_flat = e.flatten()
        sigma_e2 = (np.linalg.norm(sigma_e2_flat)**2 -
                    np.mean(sigma_e2_flat)**2) / n

        # Calculate likelihood
        likelihood_list[iter] = likelihood(omega, sigma_beta2, sigma_e2)

        # Record parameters
        omega_list[:, iter] = omega.flatten()
        sigma_beta2_list[iter] = sigma_beta2
        sigma_e2_list[iter] = sigma_e2

        # Check convergence
        if np.linalg.norm(likelihood_list[iter] -
                          likelihood_list[iter - 1]) < tol:
            break
        if iter % 10 == 0:
            print('iter: {}, likelihood: {:.4e}'.format(iter,
                                                        likelihood_list[iter]))
    beta_post_mean = np.mean(beta)
    resident = np.linalg.norm(y - Z @ omega - X @ beta) / n

    if iter == max_iter - 1:
        print('EM algorithm does not converge within {} iterations'.format(iter))
    else:
        print('EM algorithm converges after {} iterations'.format(iter))
    print('sigma_beta^2 = {:.4e}'.format(sigma_beta2))
    print('sigma_e^2 = {:.4e}'.format(sigma_e2))
    print('beta_post_mean = {:.4e}'.format(beta_post_mean))
    print('resident = {:.4e}'.format(resident))

    return likelihood_list[:iter +1], omega_list[:,\
            :iter + 1], sigma_beta2_list[:iter + 1],\
            sigma_e2_list[:iter + 1], beta_post_mean


# load data
data = pd.read_table('data/XYZ_MoM.txt', sep='\t', header=0).values
y = data[:, 0].reshape(-1, 1)
Z = data[:, 1:31]
X = data[:, 31:]

# run EM algorithm
likelihood_list, omega_list, sigma_beta2_list, sigma_e2_list, beta_post_mean = lmm_em(y, X, Z)
# MAX_LENGTH = 200
# MAX_X_LENGTH = 100
# likelihood_list, omega_list, sigma_beta2_list, sigma_e2_list, beta_post_mean = lmm_em(
#     y[:MAX_LENGTH], X[:MAX_LENGTH, :MAX_X_LENGTH], Z[:MAX_LENGTH, :])

# subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(likelihood_list)
axes[0, 0].set_title('Likelihood')

axes[0, 1].plot(sigma_beta2_list, label=r'$\sigma_\beta^2$')
axes[0, 1].plot(sigma_e2_list, label=r'$\sigma_e^2$')
axes[0, 1].set_title('Unknown Variance')
axes[0, 1].legend()

axes[1, 0].plot(omega_list.T)
axes[1, 0].set_title(r'$\omega$')

# hist
sns.distplot(omega_list[:, -1], ax=axes[1, 1], label=r'$\omega$')
axes[1, 1].axvline(beta_post_mean,
                   color='r',
                   linestyle='--',
                   label=rf'$\beta={beta_post_mean:.4e}$')
axes[1, 1].set_title('Effects')
axes[1, 1].legend()

plt.suptitle('EM Algorithm for Linear Mixed Model')
plt.tight_layout()
plt.savefig('img/lmm_em.png')
plt.show()
