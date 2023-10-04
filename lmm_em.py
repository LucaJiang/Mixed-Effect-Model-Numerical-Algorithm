# Use EM to solve linear mixed model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def logdet_inv(A):
    # Return log determinant and inverse of matrix A
    eigvals, eigvecs = np.linalg.eigh(A)
    # Compute log determinant of A from eigenvalues
    log_det = np.sum(np.log(eigvals))
    # Compute inverse of eigenvalues
    inv_eigvals = 1 / eigvals
    # Compute inverse of A from eigenvalues and eigenvectors
    inv_A = np.dot(eigvecs, np.dot(np.diag(inv_eigvals), eigvecs.T))
    return log_det, inv_A


def lmm_em(y, X, Z, tol=1e-6, max_iter=10, verbose=True):
    '''
    Input:
    y: n x 1, response
    X: n x p, random effect
    Z: n x c, fixed effect
    tol: float, tolerance for convergence, default 1e-6
    max_iter: int, maximum number of iterations, default 10
    verbose: bool, print information, default True

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
    e ~ N(0, sigma_e^2 * I_n)
    beta ~ N(0, sigma_beta^2 * I_p)

    E-step:
    Gamma = (X^T * X / sigma_e^2 + I_p / sigma_beta^2)^-1
    mu = Gamma * X^T * (y - Z * omega) / sigma_e^2
    beta = mu
    # e = y - Z * omega - X * beta

    M-step:
    omega = (Z^T * Z)^-1 * Z^T * (y - X * mu)
    sigma_beta^2 = (trace(Gamma) + ||mu||^2) / p
    sigma_e^2 = (||y - Z * W||^2 + trace(Gamma * X^T * X) + ||X * mu||^2 - 2 * (y - Z * W)^T * X * mu) / n
    
    likelihood:
    l = - n / 2 * log(2 * pi) - 1 / 2 * log(|Sigma|) - 1 / 2 * (y - Z * omega - X * beta)^T * Sigma^-1 * (y - Z * omega - X * beta)
    '''
    n, p = X.shape
    n_, c = Z.shape
    assert n == n_, 'X and Z must have same number of rows'
    assert y.shape == (n, 1), 'y must be a column vector with length n'

    # Initialize parameters
    beta = np.random.randn(p, 1)
    omega = np.random.randn(c, 1)
    sigma_beta2 = 1
    sigma_e2 = 1

    # for accelerate calculation
    XXT = X @ X.T
    XTX = X.T @ X
    ZTZinvZT = np.linalg.inv(Z.T @ Z) @ Z.T

    # Calculate likelihood
    def likelihood(omega, Sigma):
        tmp = y - Z @ omega - X @ beta
        # Find log-det and inverse with eignvalues
        Sigma_logdet, Sigma_inv = logdet_inv(Sigma)
        # Sigma_logdet = np.linalg.slogdet(Sigma)[1]
        # Sigma_inv = np.linalg.inv(Sigma)
        likelihood = -n / 2 * np.log(
            2 * np.pi) - 0.5 * Sigma_logdet - 0.5 * tmp.T @ Sigma_inv @ tmp
        return likelihood

    # Record parameters in each iteration
    max_iter += 1
    likelihood_list = np.zeros(max_iter)
    omega_list = np.zeros((c, max_iter))
    sigma_beta2_list = np.zeros(max_iter)
    sigma_e2_list = np.zeros(max_iter)

    # Initialize record
    iter = 0
    omega_list[:, iter] = omega.flatten()
    Sigma = sigma_beta2 * XXT + sigma_e2 * np.eye(n)
    likelihood_list[iter] = likelihood(omega, Sigma)
    sigma_beta2_list[iter] = sigma_beta2
    sigma_e2_list[iter] = sigma_e2

    # EM algorithm
    for iter in range(1, max_iter):
        # E step
        Gamma = np.linalg.pinv(XTX / sigma_e2 + np.eye(p) / sigma_beta2)
        mu = Gamma @ X.T @ (y - Z @ omega) / sigma_e2
        beta = mu
        # e = y - Z @ omega - X @ beta

        # M step
        omega = ZTZinvZT @ (y - X @ mu)
        sigma_beta2 = (np.trace(Gamma) + np.linalg.norm(mu)**2) / p
        sigma_e2 = (np.linalg.norm(y - Z @ omega)**2 + np.trace(Gamma @ XTX) + np.linalg.norm(X @ mu)**2 - 2 * ((y - Z @ omega).T @ X @ mu)[0, 0]) / n

        # Update Sigma
        Sigma = sigma_beta2 * XXT + sigma_e2 * np.eye(n)
        # Calculate likelihood
        likelihood_list[iter] = likelihood(omega, Sigma)

        # Record parameters
        omega_list[:, iter] = omega.flatten()
        sigma_beta2_list[iter] = sigma_beta2
        sigma_e2_list[iter] = sigma_e2

        # Check convergence
        if np.abs(likelihood_list[iter] - likelihood_list[iter - 1]) < tol:
            break
        if verbose and iter % 5 == 0:
            print('iter: {}, likelihood: {:.4e}'.format(
                iter, likelihood_list[iter]))
    beta_post_mean = np.mean(beta)
    resident = np.linalg.norm(y - Z @ omega - X @ beta) / n

    if iter == max_iter - 1:
        print(
            'EM algorithm does not converge within {} iterations'.format(iter))
    else:
        print('EM algorithm converges after {} iterations'.format(iter))
    print('sigma_beta^2 = {:.4e}'.format(sigma_beta2))
    print('sigma_e^2 = {:.4e}'.format(sigma_e2))
    print('beta_post_mean = {:.4e}'.format(beta_post_mean))
    print('omega_mean = {:.4e}'.format(np.mean(omega)))
    print('resident = {:.4e}'.format(resident))

    return likelihood_list[:iter +1], omega_list[:,\
            :iter + 1], sigma_beta2_list[:iter + 1],\
            sigma_e2_list[:iter + 1], beta_post_mean


if __name__ == '__main__':
    # load data
    data_name = 'fake_data'
    # data_name = 'XYZ_MoM'
    data = pd.read_table('data/' + data_name + '.txt', sep='\t',
                         header=0).values

    y = data[:, 0].reshape(-1, 1)
    Z = data[:, 1:31]
    X = data[:, 31:]

    # run EM algorithm
    likelihood_list, omega_list, sigma_beta2_list, sigma_e2_list, beta_post_mean = lmm_em(
        y, X, Z)

    # run EM algorithm with limited data
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
    plt.savefig('img/lmm_em' + data_name + '.png')
    plt.show()
