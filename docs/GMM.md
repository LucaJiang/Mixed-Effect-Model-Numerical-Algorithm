# Generalized Method of Moments (GMM)

- [Generalized Method of Moments (GMM)](#generalized-method-of-moments-gmm)
  - [From MM to GMM](#from-mm-to-gmm)
  - [Population Moments Conditions](#population-moments-conditions)
  - [GMM Estimator](#gmm-estimator)
  - [Implementation](#implementation)
  - [References](#references)

## From MM to GMM

In the linear regression, $k+1$ moments conditions yield $k+1$ equations and thus $k+1$ parameter estimates. If there are more moments conditions than parameters to be estimated, the moments equations cannot be solved exactly. This case is called GMM (generalized method of moments).

In GMM, moments conditions are solved approximately. To this aim, single condition equations are **weighted**.

## Population Moments Conditions

**Definition:** Let $\theta_0$ be a true unknown vector parameter to be estimated, $v_t$ a vector of random variables, and $f(.)$ **a vector of functions**. Then, a population moment condition takes the form

$$
E\{f(v_t, \theta_0)\} = 0, t \in T.
$$

Often, $f(.)$ will contain linear functions only, then the problem essentially becomes one of linear regression. In other cases, $f(.)$ may still be products of errors and functions of observed variables, then the problem becomes one of non-linear regression. The definition is even more general.

## GMM Estimator

The basic idea behind GMM is to replace the theoretical expected value E[⋅] with its empirical analog—sample average:

$$
\hat{m}(\theta)\equiv n^{-1} \sum_{t=1}^n f(v_t, \theta)\quad \text{and} \quad \hat{m}(\theta_0) = 0.
$$

which is equivalent to minimizing a certain norm of $\hat{m}(\theta)$:

$$
\hat{\theta} = \arg \min_{\theta} \|\hat{m}(\theta)\|_W^2=\arg \min_{\theta} \hat{m}(\theta)'W\hat{m}(\theta),
$$

where $W$ is a positive definite matrix. The GMM estimator is the value of $\theta$ that minimizes the above expression.

**Definition:** The Generalized Method of Moments estimator based on these population moments conditions is the value of $\theta$ that minimizes

$$
Q_n(\theta) = \left\{n^{-1} \sum_{t=1}^n f(v_t, \theta)'\right\} W_n \left\{n^{-1} \sum_{t=1}^n f(v_t, \theta)\right\},
$$

where $W_n$ is a non-negative definite matrix that usually depends on the data but converges to a constant positive definite matrix as $n \to \infty$.

## Implementation

One difficulty with implementing the outlined method is that we cannot take $W=\Omega^{-1}$ because, by the definition of matrix $\Omega$, we need to know the value of $\theta_0$ in order to compute this matrix, and $\theta_0$ is precisely the quantity we do not know and are trying to estimate in the first place. In the case of $Y_t$ being iid we can estimate $W$ as
$$
\hat{W}_T(\hat{\theta})=\left(\frac{1}{T} \sum_{t=1}^T g\left(Y_t, \hat{\theta}\right) g\left(Y_t, \hat{\theta}\right)^{\top}\right)^{-1} .
$$

Several approaches exist to deal with this issue, the first one being the most popular:
- Two-step feasible GMM:
- Step 1: Take $W=$ I (the identity matrix) or some other positive-definite matrix, and compute preliminary GMM estimate $\hat{\theta}_{(1)}$. This estimator is consistent for $\theta_0$, although not efficient.
- Step 2: $\hat{W}_T\left(\hat{\theta}_{(1)}\right)$ converges in probability to $\Omega^{-1}$ and therefore if we compute $\hat{\theta}$ with this weighting matrix, the estimator will be asymptotically efficient.
- Iterated GMM. Essentially the same procedure as 2-step GMM, except that the matrix $\hat{W}_T$ is recalculated several times. That is, the estimate obtained in step 2 is used to calculate the weighting matrix for step 3 , and so on until some convergence criterion is met.
$$
\hat{\theta}_{(i+1)}=\arg \min _{\theta \in \Theta}\left(\frac{1}{T} \sum_{t=1}^T g\left(Y_t, \theta\right)\right)^{\top} \hat{W}_T\left(\hat{\theta}_{(i)}\right)\left(\frac{1}{T} \sum_{t=1}^T g\left(Y_t, \theta\right)\right)
$$

Asymptotically no improvement can be achieved through such iterations, although certain Monte-Carlo experiments suggest that finite-sample properties of this estimator are slightly better. [citation needed]
- Continuously updating GMM (CUGMM, or CUE). Estimates $\hat{\theta}$ simultaneously with estimating the weighting matrix $W$ :
$$
\hat{\theta}=\arg \min _{\theta \in \Theta}\left(\frac{1}{T} \sum_{t=1}^T g\left(Y_t, \theta\right)\right)^{\top} \hat{W}_T(\theta)\left(\frac{1}{T} \sum_{t=1}^T g\left(Y_t, \theta\right)\right)
$$

In Monte-Carlo experiments this method demonstrated a better performance than the traditional two-step GMM: the estimator has smaller median bias (although fatter tails), and the J-test for overidentifying restrictions in many cases was more reliable.

## References

- https://homepage.univie.ac.at/robert.kunst/gmm.pdf
- https://en.wikipedia.org/wiki/Generalized_method_of_moments
