# Expectation-Maximization (EM) algorithm for Linear Mixed-Effect Model (LMM)
- [Expectation-Maximization (EM) algorithm for Linear Mixed-Effect Model (LMM)](#expectation-maximization-em-algorithm-for-linear-mixed-effect-model-lmm)
  - [Model Description](#model-description)
  - [Statistical Inference in LMM](#statistical-inference-in-lmm)
    - [Statistical Inference in E-Step](#statistical-inference-in-e-step)
    - [Statistical Inference in M-Step](#statistical-inference-in-m-step)
    - [Complete Data Log-Likelihood](#complete-data-log-likelihood)
  - [EM Algorithm](#em-algorithm)
    - [E-Step](#e-step)
    - [M-Step](#m-step)
    - [The Pseudocode of EM Algorithm](#the-pseudocode-of-em-algorithm)
  - [Codes and Results](#codes-and-results)
  - [References](#references)

## Model Description
The relation between OLM and LMM can be found in [The Introduction of Linear Mixed-Effect Model](https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/lmm_model).

The linear mixed-effect model (LMM) is a statistical model that accounts for both fixed effects and random effects in a linear regression model. It is used for modeling data where observations are not independent or identically distributed.

Consider a dataset $\{\mathbf{y}, \mathbf{X},\mathbf{Z}\}$ with $n$ samples, where $\mathbf{y} \in \mathbb{R}^n$ is the vector of response variable, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the matrix of $p$ independent variables, and $\mathbf{Z} \in \mathbb{R}^{n \times c}$ is another matrix of $c$ variables. The linear mixed model builds upon a linear relationship from $\mathbf{y}$ to $\mathbf{X}$ and $\mathbf{Z}$ by
$$\begin{equation}
\mathbf{y} = \underbrace{\mathbf{Z}\mathbf{\omega}}_{\text {fixed}} + \underbrace{\mathbf{X}\mathbf{\beta}}_{\text {random}} + \underbrace{\mathbf{e}}_{\text {error}},
\end{equation}$$
where $\mathbf{\omega} \in \mathbb{R}^c$ is the vector of fixed effects, $\mathbf{\beta} \in \mathbb{R}^p$ is the vector of random effects with $\mathbf{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_p)$, and $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I}_n)$ is the independent noise term. 

Let $\mathbf{\Theta}$ denote the set of unknown parameters $\mathbf{\Theta} = \{\mathbf{\omega}, \sigma^2_\mathbf{\beta}, \sigma^2_e\}$. Under the framework of EM algorithm, we can treat $\mathbf{\beta}$ as a latent variable. Below is the directed acyclic graph below for our model.

<p align="center" width="100%">
    <img width="40%" src="https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/dag.bmp">
</p>

In the following sections, we will first find the posterior distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\hat{\mathbf{\Theta}}$ in the E-step. Then we will assume that $\mathbf{\beta}$ has been estimated by $\hat{\mathbf{\beta}}$ and find the ML estimator of $\mathbf{\Theta}$ which is used in M-step. The EM algorithm is an iterative algorithm that alternates between the E-step and the M-step until convergence. Finally, we will calculate and track the complete data log-likelihood $\ell_c(\mathbf{\Theta})$ to check the convergence of the algorithm.


## Statistical Inference in LMM
Question: Can we use $\mathbf{y}-\mathbf{Z}\mathbf{\omega}| \mathbf{\Theta}, \mathbf{\beta} \sim \mathcal{N}(\mathbf{X}\mathbf{\beta}, \sigma_\beta^2\mathbf{X}\mathbf{X}^T + \sigma_e^2\mathbf{I}_n)$?[^1] The following derivation are based on Ref 1.

[^1]: If so, $\omega = (\mathbf{Z}^T\Sigma^{-1} \mathbf{Z})^{-1} \mathbf{Z}^T\Sigma^{-1} \mathbf{y}$, where $\Sigma = \sigma_\beta^2\mathbf{X}\mathbf{X}^T + \sigma_e^2\mathbf{I}_n$. And $\mathbf{\beta}= \sigma_\beta^2\mathbf{X}^T\Sigma^{-1} (\mathbf{y}-\mathbf{Z}\mathbf{\omega})$, $\mathbb{V}(\mathbf{\omega})= (\mathbf{Z}\Sigma^{-1}\mathbf{X})^{-1}$, $\mathbb{V}(\mathbf{\beta})=\sigma_\beta^4 \mathbf{X}^T {\Sigma^{-1}-\Sigma^{-1}\mathbf{Z}(\mathbf{Z}\Sigma^{-1}\mathbf{Z})^{-1}\mathbf{Z}^T\Sigma^{-1}}\mathbf{X}$. More details in Ref 2 and 3.


### Statistical Inference in E-Step
The E-step aims to find the posterior distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\mathbf{\Theta}$. The complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell_c(\mathbf{\Theta}) &= \log p(\mathbf{y}, \mathbf{\beta}, \mathbf{\Theta})\\
&= \log p(\mathbf{\beta}| \mathbf{y}, \mathbf{\Theta}) + \log p(\mathbf{y}| \mathbf{\Theta})
\end{split}\end{equation}$$
where $p(\mathbf{y}| \mathbf{\Theta})$ dose not depend on $\mathbf{\beta}$. However the posterior distribution of $\mathbf{\beta}$ can not be obtained directly. Instead, by applying Bayes' rule, we have
$$\begin{equation}
\begin{split}
p(\mathbf{\beta}| \mathbf{y}, \mathbf{\Theta}) 
&= \frac{p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) p(\mathbf{\beta}| \mathbf{\Theta})}{p(\mathbf{y}| \mathbf{\Theta})}\\
&\propto p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) p(\mathbf{\beta}| \mathbf{\Theta})\\
&\propto \exp \left\{-\frac{1}{2}(\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T (\sigma_e^2\mathbf{I}_n)^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}) \right\} \\
&\quad \times \exp \left\{-\frac{1}{2} \mathbf{\beta}^T (\sigma_\mathbf{\beta}^2 \mathbf{I}_p)^{-1} \mathbf{\beta} \right\} \\
&\propto \exp \left\{-\frac{1}{2}\left( \mathbf{\beta} - \mathbf{\mu}\right)^T \Gamma^{-1} \left( \mathbf{\beta} - \mathbf{\mu}\right) \right\}
\end{split}\end{equation}$$

where $\Gamma = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$ and $\mathbf{\mu} = \Gamma \mathbf{X}^T (\mathbf{y}-\mathbf{Z}\mathbf{\omega})/\sigma_e^2$. And, the posterior distribution of $\mathbf{\beta}$ is $\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta} \sim \mathcal{N}(\mathbf{\mu}, \Gamma)$.

### Statistical Inference in M-Step
If $\mathbf{\beta}$ has been estimated, the conditional distribution of $\mathbf{y}$ become 
$$\mathbf{y}| \mathbf{\Theta}, \mathbf{\beta} \sim \mathcal{N}(\mathbf{Z}\mathbf{\omega}+ \mathbf{X}\mathbf{\beta}, \sigma_e^2\mathbf{I}_n)$$

Then the ML estimator of $\mathbf{\Theta}$ maximizes the complete data log-likelihood:
$$\begin{equation}
\begin{split}
\ell_c(\Theta) &= \log p(\mathbf{y}, \mathbf{\beta}, \mathbf{\Theta})\\
&= \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \Theta)\end{split}\end{equation}$$

The marginal log-likelihood is given by
$$\begin{equation}
\begin{split}
\log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) &= -\frac{n}{2} \log (2\pi) -\frac{1}{2} \log |\sigma_e^2\mathbf{I}_n| \\&\quad - \frac{1}{2} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T (\sigma_e^2\mathbf{I}_n)^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})\\
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2
\end{split}\end{equation}$$
and
$$\begin{equation}
\begin{split}
\log p(\mathbf{\beta}| \Theta) &= -\frac{p}{2} \log (2\pi) -\frac{p}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}
\end{split}\end{equation}$$

Maximizing (5) and (6) with respect to $\mathbf{\Theta}$ is equivalent to set the partial derivatives of $\ell_c(\Theta)$ with respect to $\mathbf{\Theta}$ to zero. Thus, we have
$$\begin{equation}
\begin{split}
\frac{\partial \ell_c}{\partial \mathbf{\omega}} &= \frac{1}{\sigma_e^2} \mathbf{Z}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}) =: 0\\
\Rightarrow \hat{\mathbf{\omega}} &= (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T (\mathbf{y} - \mathbf{X}\mathbf{\beta})\\
\frac{\partial \ell_c}{\partial \sigma_\beta^2} &= -\frac{p}{2\sigma_\beta^2} + \frac{1}{2\sigma_\beta^4} \mathbf{\beta}^T \mathbf{\beta} =: 0\\
\Rightarrow \hat{\sigma}_\beta^2 &= \frac{1}{p} \|\mathbf{\beta}\|^2 \\
\frac{\partial \ell_c}{\partial \sigma_e^2} &= -\frac{n}{2\sigma_e^2} + \frac{1}{2\sigma_e^4} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2 =: 0\\
\Rightarrow \hat{\sigma}_e^2 &= \frac{1}{n} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2\\
&= \frac{1}{n} \left( \|\mathbf{y}-\mathbf{Z}\mathbf{\omega}\|^2 + \|\mathbf{X}\mathbf{\beta}\|^2 - 2(\mathbf{y}-\mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\beta}\right)
% &= \frac{1}{n}\left( \|\mathbf{y}-\mathbf{Z}\mathbf{\omega}\|^2 + \text{tr}\left(\mathbf{\beta}^T\mathbf{X}^T\mathbf{X}\mathbf{\beta}\right) - 2(\mathbf{y}-\mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\beta}\right)
\end{split}\end{equation}$$

Since, the posterior distribution of $\mathbf{\beta}$ is $\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta} \sim \mathcal{N}(\mathbf{\mu}, \Gamma)$.

$$\begin{equation}
\begin{split}
p\hat{\sigma}_\beta^2 &=\mathbb{E}[\|\mathbf{\beta}\|^2|\mathbf{y}, \mathbf{\Theta}] = \text{tr}\left(\mathbb{V}[\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta}]\right) +\mathbb{E}[\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta}]^T \mathbb{E}[\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta}] = \text{tr}(\Gamma) + \|\mathbf{\mu}\|^2,\\
n\hat{\sigma}_e^2 &= \mathbb{E}[\|\mathbf{y}-\mathbf{Z}\mathbf{\omega}-\mathbf{X}\mathbf{\beta}\|^2|\mathbf{y}, \mathbf{\Theta}]\\
&= \|\mathbf{y}-\mathbf{Z}\mathbf{\omega}\|^2 + \text{tr}\left(\Gamma \mathbf{X}^T\mathbf{X}\right)+ \mathbf{\mu}^T\mathbf{X}^T\mathbf{X}\mathbf{\mu} - 2(\mathbf{y}-\mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu}
\end{split}\end{equation}$$

Note that $\mathbb{E}[(\mathbf{X}\mathbf{\beta})^T (\mathbf{X}\mathbf{\beta})] = \text{tr}(\mathbf{X}\Gamma \mathbf{X}^T) + (\mathbf{X}\mathbf{\mu})^T (\mathbf{X}\mathbf{\mu})$.

### Complete Data Log-Likelihood
The complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell_c(\mathbf{\Theta}) &= \log p(\mathbf{y}, \mathbf{\beta}, \mathbf{\Theta})\\
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2\\
&\quad -\frac{p}{2} \log (2\pi) -\frac{p}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}
\end{split}\end{equation}$$

-------------

Since
$$\begin{equation} \mathbf{y}|\mathbf{\Theta}\sim \mathcal{N}(\mathbf{Z}\mathbf{\omega} + \mathbf{X}\mathbf{\beta}, \mathbf{\Sigma})\end{equation}$$
where $\mathbf{\Sigma} = \mathbf{\sigma}_\beta^2 \mathbf{X}\mathbf{X}^T + \mathbf{\sigma}_e^2 \mathbf{I}_n$, the  complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell_c(\Theta|\mathbf{y}) &= \log p(\mathbf{\Theta}|\mathbf{y})\\
&= \log f_\mathbf{y}(\mathbf{y}|\mathbf{\Theta})\\
&= -\frac{n}{2} \log (2\pi) -\frac{1}{2} \log |\mathbf{\Sigma}| - \frac{1}{2} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T \mathbf{\Sigma}^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})
\end{split}\end{equation}$$

## EM Algorithm
### E-Step
The E-step is to compute the expectation of the complete data log-likelihood with respect to the conditional distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\Theta^{(t)} $:
$$\begin{equation}
\begin{split}
Q(\Theta| \mathbf{\Theta}^{(t)}) &= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta}^{(t)}} \left[ \log p(\mathbf{y}, \mathbf{\beta}| \Theta) \right] \\
&= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta}^{(t)}} \left[ \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \Theta) \right]
\end{split}\end{equation}$$

Thus, the E-step is to compute the following expectations:

$$\begin{equation}
\hat{\mathbf{\beta}}^{(t+1)}= \mathbf{\mu}^{(t)}
\end{equation}$$
<!-- \quad \text{and} \quad \hat{\mathbf{e}}^{(t+1)} = \mathbf{y} - \mathbf{Z}\mathbf{\omega}^{(t)} - \mathbf{X}\hat{\beta}^{(t+1)} -->
where $\Gamma = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$ and $\mathbf{\mu} = \Gamma \mathbf{X}^T (\mathbf{y}-\mathbf{Z}\mathbf{\omega})/\sigma_e^2$.


### M-Step
The M-step is to maximize the expectation of the complete data log-likelihood with respect to $\mathbf{\Theta}$:
$$\begin{equation}
\mathbf{\Theta}^{(t+1)} = \arg\max_\mathbf{\Theta} Q(\mathbf{\Theta}|\mathbf{\Theta}^{(t)})
\end{equation}$$
which is equivalent to maximizing (2) and according to the results in (8), we have
$$\begin{equation}
\begin{split}
\hat{\mathbf{\omega}}^{(t+1)} &= (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T (\mathbf{y} - \mathbf{X}\mathbf{\mu}^{(t+1)})\\
\hat{\mathbf{\sigma}}_\beta^{2(t+1)} &= \frac{1}{p} \left(\text{tr}(\Gamma^{(t+1)}) + \| \mathbf{\mu}^{(t+1)}\|^2 \right)\\
\hat{\mathbf{\sigma}}_e^{2(t+1)} &= \frac{1}{n}\left( \|\mathbf{y}-\mathbf{Z}\mathbf{\omega}^{(t+1)}\|^2 + \text{tr}\left(\Gamma^{(t+1)}\mathbf{X}^T\mathbf{X}\right)+ \|\mathbf{X}\mathbf{\mu}^{(t+1)} \|^2 - 2(\mathbf{y}-\mathbf{Z}\mathbf{\omega}^{(t+1)})^T \mathbf{X}\mathbf{\mu}^{(t+1)}\right)
\end{split}\end{equation}$$
Then the complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell_c(\mathbf{\Theta}) &= -\frac{n}{2} \log (2\pi) - \frac{1}{2} \log |\mathbf{\Sigma}^{(t+1)}| - \frac{1}{2} (\mathbf{y} - \mathbf{Z}\mathbf{\omega}^{(t+1)})^T (\mathbf{\Sigma}^{(t+1)})^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega}^{(t+1)} ) 
\end{split}\end{equation}$$
where $\mathbf{\Sigma} = \sigma_\mathbf{\beta}^2 \mathbf{X}\mathbf{X}^T + \sigma_e^2 \mathbf{I}_n$. When $|\ell_c(\mathbf{\Theta}^{(t+1)}) - \ell_c(\mathbf{\Theta}^{(t)})| < \varepsilon$, where $\varepsilon$ is a small number, the algorithm is considered to be converged.


### The Pseudocode of EM Algorithm
The EM algorithm is an iterative algorithm that alternates between the E-step and the M-step until convergence. The algorithm is summarized as follows:
1. Initialize $\mathbf{\Theta}^{(0)}$ and $\mathbf{\beta}$ randomly.
2. For $t = 0, 1, \dots$, MAX_ITERATION:
   1. E-step: Estimate $\hat{\mathbf{\beta}}^{(t)}$.
   2. M-step: Estimate $\hat{\mathbf{\Theta}}^{(t+1)}=\{\hat{\mathbf{\omega}}^{(t+1)}, \hat{\sigma}_\beta^{2(t+1)}, \hat{\sigma}_e^{2(t+1)}\}$.
   3. Check $|\Delta \ell_c|$ for convergence. If converged, stop. Otherwise, continue.
3. Return results.


## Codes and Results
The following results are obtained by running the EM algorithm on the given dataset.
[Link to code](https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/em_result)

Calculation details:
- $\log |\mathbf{\Sigma}|$: Directly calculate the log determinant of $\mathbf{\Sigma}$ would cause numerical overflow:
```python
RuntimeWarning: overflow encountered in reduce return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
```
That's because "If an array has a very small or very large determinant, then a call to `det` may overflow or underflow". When the dimension of $\mathbf{\Sigma}$ is large, $|\mathbf{\Sigma}|$ would be extremely close to zero so that it may be considered as zero by numpy. Therefore, we use $\log |\mathbf{\Sigma}| = \sum_{i=1}^n \log \lambda_i$, where $\lambda_i$ is the $i$-th eigenvalue of $\mathbf{\Sigma}$.

Alternative methods: np.linalg.slogdet apply LU factorization to calculate the log determinant of a matrix [doc](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html).

## References
1. An EM Algorithm for Linear Mixed Effects Models. [MAP566](https://jchiquet.github.io/MAP566/docs/mixed-models/map566-lecture-EM-linear-mixed-model.html)
2. Lindstrom M J, Bates D M. Newton—Raphson and EM algorithms for linear mixed-effects models for repeated-measures data[J]. Journal of the American Statistical Association, 1988, 83(404): 1014-1022. https://www.jstor.org/stable/2290128

3. Laird N M, Ware J H. Random-effects models for longitudinal data[J]. Biometrics, 1982: 963-974. https://www.jstor.org/stable/2529876