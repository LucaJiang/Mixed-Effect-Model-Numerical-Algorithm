# Expectation-Maximization (EM) algorithm for Linear Mixed-Effect Model (LMM)
- [Expectation-Maximization (EM) algorithm for Linear Mixed-Effect Model (LMM)](#expectation-maximization-em-algorithm-for-linear-mixed-effect-model-lmm)
  - [Model Description](#model-description)
  - [Statistical Inference in LMM](#statistical-inference-in-lmm)
    - [Complete Data Log-Likelihood](#complete-data-log-likelihood)
    - [Statistical Inference in E-Step](#statistical-inference-in-e-step)
    - [Statistical Inference in M-Step](#statistical-inference-in-m-step)
  - [The EM Algorithm](#the-em-algorithm)
    - [The Pseudocode of EM Algorithm](#the-pseudocode-of-em-algorithm)
  - [Codes and Results](#codes-and-results)
  - [TODO: Acceleration:](#todo-acceleration)
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

<p width="80%">
<figure align="center"  class="image">
  <img width="40%" src="https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/dag.bmp" alt="Directed acyclic graph">
  <figcaption style="text-align: left;">Figure 1. The directed acyclic graph for the linear mixed-effect model. The shaded nodes are observed variables and the unshaded nodes are latent variables. The arrows indicate the conditional dependencies. 
  The points indicate the parameters.
  The plate indicates that the variables inside the plate are replicated n times.
  </figcaption>
</figure>
</p>

The EM algorithm is an iterative algorithm that alternates between the E-step and the M-step until convergence. In the following sections, we will first find the posterior distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\hat{\mathbf{\Theta}}$ in the E-step and calculate the $Q$ function. Then we will  find the ML estimator of $\mathbf{\Theta}$ which is used in M-step.  Finally, we will calculate and track the complete data log-likelihood $\ell(\mathbf{\Theta})$ to check the convergence of the algorithm.


## Statistical Inference in LMM
<!-- Question: Can we use $\mathbf{y}-\mathbf{Z}\mathbf{\omega}| \mathbf{\Theta}, \mathbf{\beta} \sim \mathcal{N}(\mathbf{X}\mathbf{\beta}, \sigma_\beta^2\mathbf{X}\mathbf{X}^T + \sigma_e^2\mathbf{I}_n)$?[^1] The following derivation are based on Ref 1. -->

<!-- [^1]: If so, $\omega = (\mathbf{Z}^T\Sigma^{-1} \mathbf{Z})^{-1} \mathbf{Z}^T\Sigma^{-1} \mathbf{y}$, where $\Sigma = \sigma_\beta^2\mathbf{X}\mathbf{X}^T + \sigma_e^2\mathbf{I}_n$. And $\mathbf{\beta}= \sigma_\beta^2\mathbf{X}^T\Sigma^{-1} (\mathbf{y}-\mathbf{Z}\mathbf{\omega})$, $\mathbb{V}(\mathbf{\omega})= (\mathbf{Z}\Sigma^{-1}\mathbf{X})^{-1}$, $\mathbb{V}(\mathbf{\beta})=\sigma_\beta^4 \mathbf{X}^T {\Sigma^{-1}-\Sigma^{-1}\mathbf{Z}(\mathbf{Z}\Sigma^{-1}\mathbf{Z})^{-1}\mathbf{Z}^T\Sigma^{-1}}\mathbf{X}$. More details in Ref 2 and 3. -->

### Complete Data Log-Likelihood
The complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell(\mathbf{\Theta}, \mathbf{\beta}) &= \log p(\mathbf{y}, \mathbf{\beta}| \mathbf{\Theta})\\
&= \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \mathbf{\Theta})\\
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2\\
&\quad -\frac{p}{2} \log (2\pi) -\frac{p}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}
\end{split}\end{equation}$$


### Statistical Inference in E-Step

The E-step is to compute the expectation of the complete data log-likelihood with respect to the conditional distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\Theta$:
$$\begin{equation}
\begin{split}
Q(\Theta|\Theta^{\text{old}}) &= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \Theta^{\text{old}}} \left[ \log p(\mathbf{y}, \mathbf{\beta}|\Theta) \right]
\end{split}\end{equation}.$$

Therefore, we need to find the posterior distribution of $\mathbf{\beta}$ given $\mathbf{y}$ and $\mathbf{\Theta}$. However the posterior distribution of $\mathbf{\beta}$ can not be obtained directly. Instead, by applying Bayes' rule, we have
$$\begin{equation}
\begin{split}
p(\mathbf{\beta}| \mathbf{y}, \mathbf{\Theta}) 
&= \frac{p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) p(\mathbf{\beta}| \mathbf{\Theta})}{p(\mathbf{y}| \mathbf{\Theta})}\\
&\propto p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) p(\mathbf{\beta}| \mathbf{\Theta})\\
&\propto \exp \left\{-\frac{1}{2}(\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T (\sigma_e^2\mathbf{I}_n)^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}) \right\} \\
&\quad \times \exp \left\{-\frac{1}{2} \mathbf{\beta}^T (\sigma_\mathbf{\beta}^2 \mathbf{I}_p)^{-1} \mathbf{\beta} \right\} \\
&\propto \exp \left\{-\frac{1}{2}\left( \mathbf{\beta} - \mathbf{\mu}\right)^T \mathbf{\Gamma}^{-1} \left( \mathbf{\beta} - \mathbf{\mu}\right) \right\}
\end{split}\end{equation}$$

where $\mathbf{\Gamma} = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$ and $\mathbf{\mu} = \mathbf{\Gamma} \mathbf{X}^T (\mathbf{y}-\mathbf{Z}\mathbf{\omega})/\sigma_e^2$. And, the posterior distribution of $\mathbf{\beta}$ is $\mathbf{\beta}|\mathbf{y}, \mathbf{\Theta} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Gamma})$. [^Notes]

[^Notes]: Notes on the derivation of the posterior distribution of $\mathbf{\beta}$: Let $g(\mathbf{\beta})= -\frac{1}{2}(\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T (\sigma_e^2\mathbf{I}_n)^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}) -\frac{1}{2} \mathbf{\beta}^T (\sigma_\mathbf{\beta}^2 \mathbf{I}_p)^{-1} \mathbf{\beta}$. Then we have $\nabla g(\mathbf{\beta}) = \mathbf{X}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})/\sigma_e^2 - \mathbf{\beta}/\sigma_\mathbf{\beta}^2$. Let $\nabla g(\mathbf{\beta}) = 0$, we have $\mathbf{\beta} = \mathbf{\Gamma} \mathbf{X}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega})/\sigma_e^2$, where $g(\mathbf{\beta})$ is maximized. Thus, we find the $\mathbf{\mu}$. To obtain the variance of $\mathbf{\beta}$, we need to calculate the Hessian matrix of $g(\mathbf{\beta})$ and evaluate it at $\mathbf{\beta} = \mathbf{\mu}$. The Hessian matrix is given by $\nabla^2 g(\mathbf{\beta}) = \mathbf{X}^T \mathbf{X}/\sigma_e^2 + \mathbf{I}_p/\sigma_\mathbf{\beta}^2$. Therefore, the variance of $\mathbf{\beta}$ is $\mathbf{\Gamma} = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$.

Thus, we have
$$\begin{equation}
\begin{split}
Q(\Theta|\Theta^{\text{old}}) &= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \Theta^{\text{old}}} \left[ \log p(\mathbf{y}, \mathbf{\beta}|\Theta) \right]\\
&= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \Theta^{\text{old}}} \left[ \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \Theta) \right]\\
&= \mathbb{E}_{\mathbf{\beta}|\mathbf{y}, \Theta^{\text{old}}} \left[ -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2 \right.\\
&\quad \left. -\frac{p}{2} \log (2\pi) -\frac{p}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta} \right]\\
&= -\frac{n+p}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^{2}\\
&\quad - \frac{1}{2\sigma_e^{2}} \left(\|\mathbf{y} - \mathbf{Z}\mathbf{\omega}\|^2 + \text{tr}(\mathbf{X}\mathbf{\Gamma} \mathbf{X}^T) + \mathbf{\mu}^T\mathbf{X}^T \mathbf{X}\mathbf{\mu}- 2(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu} \right)\\
&\quad -\frac{p}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \text{tr}(\mathbf{\Gamma}) - \frac{1}{2\sigma_\beta^2} \mathbf{\mu}^{T} \mathbf{\mu}
\end{split}\end{equation}$$

Note that $\mathbb{E}[\mathbf{\beta}^T \mathbf{\beta}] = \text{tr}(\mathbb{V}[\mathbf{\beta}]) + \mathbb{E}[\mathbf{\beta}]^T \mathbb{E}[\mathbf{\beta}] = \text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}$ and $\mathbb{E}[(\mathbf{X}\mathbf{\beta})^T (\mathbf{X}\mathbf{\beta})] = \text{tr}(\mathbf{X}\mathbf{\Gamma} \mathbf{X}^T) + (\mathbf{X}\mathbf{\mu})^T (\mathbf{X}\mathbf{\mu}).$

### Statistical Inference in M-Step
M-step is to maximize the $Q$ function with respect to $\mathbf{\Theta}$, which  is equivalent to set the partial derivatives of $Q$ with respect to $\mathbf{\Theta}$ to zero. Thus, we have
$$\begin{equation}
\begin{split}
\frac{\partial Q}{\partial \mathbf{\omega}} &= -\frac{1}{\sigma_e^2} \mathbf{Z}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}) =: 0\\
\Rightarrow \hat{\mathbf{\omega}} &= (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T (\mathbf{y} - \mathbf{X}\mathbf{\mu})\\
\frac{\partial Q}{\partial \sigma_\beta^2} &= -\frac{p}{2\sigma_\beta^2} + \frac{1}{2\sigma_\beta^4} \left(\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu} \right)
=: 0\\
\Rightarrow \hat{\sigma}_\beta^2 &= \frac{\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}}{p}\\
\frac{\partial Q}{\partial \sigma_e^2} &= -\frac{n}{2\sigma_e^2} + \frac{1}{2\sigma_e^4}  \left(\|\mathbf{y} - \mathbf{Z}\mathbf{\omega}\|^2 + \text{tr}(\mathbf{X}\mathbf{\Gamma} \mathbf{X}^T) + \mathbf{\mu}^T\mathbf{X}^T \mathbf{X}\mathbf{\mu}- 2(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu} \right) =: 0\\
\Rightarrow \hat{\sigma}_e^2 &=  \frac{1}{n} \left( \|\mathbf{y}-\mathbf{Z}\mathbf{\omega}\|^2 + \text{tr}\left(\mathbf{\Gamma} \mathbf{X}^T\mathbf{X}\right)+ \mathbf{\mu}^T\mathbf{X}^T\mathbf{X}\mathbf{\mu} - 2(\mathbf{y}-\mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu}\right)
\end{split}\end{equation}$$


## The EM Algorithm
The E-step is to compute the following expectations:

$$\begin{equation}
\hat{\mathbf{\beta}}= \mathbf{\mu}^{\text{old}}
\end{equation}$$

where $\mathbf{\Gamma} = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$ and $\mathbf{\mu} = \mathbf{\Gamma} \mathbf{X}^T (\mathbf{y}-\mathbf{Z}\mathbf{\omega})/\sigma_e^2$.

The M-step is to maximize the expectation of the complete data log-likelihood with respect to $\mathbf{\Theta}$:
$$\begin{equation}
\mathbf{\Theta} = \arg\max_\mathbf{\Theta} Q(\mathbf{\Theta}|\mathbf{\Theta}^{\text{old}})
\end{equation}$$
We have
$$\begin{equation}
\begin{split}
\hat{\mathbf{\omega}} &= (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T (\mathbf{y} - \mathbf{X}\mathbf{\beta})\\
\hat{\mathbf{\sigma}}_\beta^{2} &= \frac{1}{p} \left(\text{tr}(\mathbf{\Gamma}) + \| \mathbf{\beta}\|^2 \right)\\
\hat{\mathbf{\sigma}}_e^{2} &= \frac{1}{n}\left( \|\mathbf{y}-\mathbf{Z}\hat{\mathbf{\omega}}\|^2 + \text{tr}\left(\mathbf{\Gamma}\mathbf{X}^T\mathbf{X}\right)+ \|\mathbf{X}\mathbf{\beta} \|^2 - 2(\mathbf{y}-\mathbf{Z}\hat{\mathbf{\omega}})^T \mathbf{X}\mathbf{\beta}\right)
\end{split}\end{equation}$$
Then the complete data log-likelihood is given by
$$\begin{equation}
\begin{split}
\ell(\mathbf{\Theta}, \mathbf{\beta}) &= \log p(\mathbf{y}, \mathbf{\beta}| \mathbf{\Theta})\\
&= \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \mathbf{\Theta})\\
&= -\frac{n+p}{2} \log (2\pi) -\frac{n}{2} \log \hat{\sigma}_e^2\\
&\quad  - \frac{1}{2\hat{\sigma}_e^2} \|\mathbf{y} - \mathbf{Z}\hat{\mathbf{\omega}} - \mathbf{X}\mathbf{\beta}\|^2-\frac{p}{2} \log \hat{\sigma}_\beta^2 - \frac{1}{2\hat{\sigma}_\beta^2} \mathbf{\beta}^T \mathbf{\beta}
\end{split}\end{equation}$$


When $|\Delta \ell| = |\ell(\mathbf{\Theta}, \mathbf{\beta}) - \ell(\mathbf{\Theta}^{\text{old}}, \mathbf{\beta})| < \varepsilon$, where $\varepsilon$ is a small number, the algorithm is considered to be converged.


### The Pseudocode of EM Algorithm
The EM algorithm is an iterative algorithm that alternates between the E-step and the M-step until convergence. The algorithm is summarized as follows:
1. Initialize $\mathbf{\Theta}^{(0)}$ and $\mathbf{\beta}$ randomly.
2. For $t = 0, 1, \dots$, MAX_ITERATION:
   1. E-step: Estimate $\mathbf{\beta}$.
   2. M-step: Estimate $\hat{\mathbf{\Theta}}=\{\hat{\mathbf{\omega}}, \hat{\sigma}_\beta^{2}, \hat{\sigma}_e^{2}\}$.
   3. Check $|\Delta \ell|$ for convergence. If converged, stop. Otherwise, continue.
3. Return results.


## Codes and Results

The results below are obtained by running the EM algorithm on a generated dataset.
![Result on generated dataset](https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/lmm_emfake_data.png)

The results in [code](https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/em_result) are obtained by running the EM algorithm on a given dataset.


Calculation details:
Since $\mathbf{\Gamma} = \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$, we need to calculate this inverse in each iteration. When $p$ is large and the elements of $\mathbf{X}$ are small, the inverse of $\mathbf{X}^T \mathbf{X}$ may be ill-conditioned. Therefore, we use the eigenvalue decomposition of $\mathbf{X}^T \mathbf{X}$ to accelerate the calculation of $\mathbf{\Gamma}$.

Let $\mathbf{X}^T \mathbf{X} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$, where $\mathbf{Q}$ is an orthogonal matrix and $\mathbf{\Lambda}$ is a diagonal matrix with the eigenvalues of $\mathbf{X}^T \mathbf{X}$ on the diagonal. Then we have
$$\begin{equation}
\begin{split}
\mathbf{\Gamma} &= \left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}\\
&= \left(\frac{\mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}\\
&= \mathbf{Q} \left(\frac{\mathbf{\Lambda}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1} \mathbf{Q}^T
\end{split}\end{equation}$$

where $\left(\frac{\mathbf{\Lambda}}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\mathbf{\beta}^2} \right)^{-1}$ is a diagonal matrix which is easy to calculate the inverse.

When calculating the eigenvalue decomposition of $\mathbf{\Gamma}$, it's better to use '[numpy.linalg.eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)' instead of '[numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)' since $\mathbf{\Gamma}$ is a real symmetric matrix. The former is faster and more accurate than the latter.


## TODO: Acceleration:
1. Aitken acceleration
2. numba

## References
1. An EM Algorithm for Linear Mixed Effects Models. [MAP566](https://jchiquet.github.io/MAP566/docs/mixed-models/map566-lecture-EM-linear-mixed-model.html)
2. Lindstrom M J, Bates D M. Newton—Raphson and EM algorithms for linear mixed-effects models for repeated-measures data[J]. Journal of the American Statistical Association, 1988, 83(404): 1014-1022. https://www.jstor.org/stable/2290128

3. Laird N M, Ware J H. Random-effects models for longitudinal data[J]. Biometrics, 1982: 963-974. https://www.jstor.org/stable/2529876