# The Introduction of Linear Mixed-Effect Model
- [The Introduction of Linear Mixed-Effect Model](#the-introduction-of-linear-mixed-effect-model)
  - [From OLM to LMM](#from-olm-to-lmm)
  - [Model Description](#model-description)
  - [Formal Description of Bayesian Inference and Variational Inference](#formal-description-of-bayesian-inference-and-variational-inference)
    - [Definitions](#definitions)
    - [Bayesian Inference](#bayesian-inference)
    - [Bayesian Prediction](#bayesian-prediction)
    - [Variational Inference](#variational-inference)

## From OLM to LMM
The OLM (Ordinary Linear Model) assumes that the observations are independent and identically distributed (i.i.d.). It can be written as
$$\begin{equation}
\mathbf{y} = \mathbf{X}\mathbf{\gamma} + \mathbf{e},
\end{equation}$$
where $\mathbf{y} \in \mathbb{R}^n$ is the vector of response variable, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the matrix of $p$ independent variables, $\mathbf{\gamma} \in \mathbb{R}^p$ is the vector of coefficients, and $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I}_n)$ is the independent noise term. To simplify the notation, we assume that $\mathbf{X}$ and $\mathbf{y}$ are centered, i.e. the mean of $\mathbf{y}$ is $\mathbf{0}$, and the mean of $\mathbf{X}$ is $\mathbf{0}$ as well. Thus, we do not need to estimate the intercept term.

In the OLM, if we want to obtain the coefficients $\mathbf{\gamma}$, we can use the least square method to minimize the loss function. However, when $n<p$, the matrix $\mathbf{X}$ is not full rank, and the least square method could lead to overfitting. Because we have $p+1$ (including $\sigma_e^2$) parameters to estimate, but only $n$ observations. 

Let's assume the design matrix $\mathbf{X}$ can be decomposed into two parts, i.e., $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]$, where $\mathbf{X}_1 \in \mathbb{R}^{n \times p_1}, p_1 < n\ll p$ and $\mathbf{X}_2 \in \mathbb{R}^{n \times p_2}$. And, $\mathbf{\gamma}$ can be break down into a fixed effect part $\mathbf{\omega}\in \mathbb{R}^{p_1}$ and a random effect part $\mathbf{\beta}\in \mathbb{R}^{p_2}$ the distribution of which is $\mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_{p_2})$, i.e., $\mathbf{\gamma}^T = [\mathbf{\omega}^T, \mathbf{\beta}^T]$. Then, we have
$$\begin{equation}
\mathbf{y} = \mathbf{X}_1\mathbf{\omega} + \mathbf{X}_2\mathbf{\beta} + \mathbf{e}
\end{equation}$$
Thus, we obtain the LMM (Linear Mixed-Effect Model). In this case, the number of parameters to estimate is $p_1 + 1 + 1$ (including $\sigma_\beta^2$ and $\sigma_e^2$), which is much smaller than $p+1$ in the OLM.


## Model Description
LMM is a statistical model that accounts for both fixed effects and random effects in a linear regression model. It is used for modeling data where observations are not independent or identically distributed.

Consider a dataset $\{\mathbf{y}, \mathbf{X},\mathbf{Z}\}$ with $n$ samples, where $\mathbf{y} \in \mathbb{R}^n$ is the vector of response variable, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the matrix of $p$ independent variables, and $\mathbf{Z} \in \mathbb{R}^{n \times c}$ is another matrix of $c$ variables. The linear mixed model builds upon a linear relationship from $\mathbf{y}$ to $\mathbf{X}$ and $\mathbf{Z}$ by
$$\begin{equation}
\mathbf{y} = \underbrace{\mathbf{Z}\mathbf{\omega}}_{\text {fixed}} + \underbrace{\mathbf{X}\mathbf{\beta}}_{\text {random}} + \underbrace{\mathbf{e}}_{\text {error}},
\end{equation}$$
where $\mathbf{\omega} \in \mathbb{R}^c$ is the vector of fixed effects, $\mathbf{\beta} \in \mathbb{R}^p$ is the vector of random effects with $\mathbf{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_p)$, and $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I}_n)$ is the independent noise term. 

The LMM can be solved by various methods, such as the restricted maximum likelihood (REML) and the maximum likelihood (ML). The REML is a method of estimation that does not base estimates on a maximum likelihood fit of all the information, but instead uses a likelihood function derived from a transformed set of data, so that nuisance parameters have no effect. The ML is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations given the parameters. The ML is a special case of the maximum a posteriori estimation (MAP) that assumes that the prior over the parameters is uniform or non-informative.

## Formal Description of Bayesian Inference and Variational Inference
The following contents are basically from [Wikipedia-BI](https://en.wikipedia.org/wiki/Bayesian_inference) and [Wikipedia-VBM](https://en.wikipedia.org/wiki/Variational_Bayesian_methods). The purpose of this section is to provide a formal description of Bayesian inference and variational inference, which will be used in the following sections.

### Definitions
- $\mathbf{x}$: a data point in general.
- $\mathbf{X}$: sample, a set of $n$ observed data points, i.e., $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_n\}$.
- $\tilde{\mathbf{x}}$: a new data point to be predicted.
- $\mathbf{\theta}$: the parameters of the data point's distribution, i.e., $\mathbf{x} \sim p(\mathbf{x}|\mathbf{\theta})$.
- $\mathbf{\alpha}$: the hyperparameters of the distribution of $\mathbf{\theta}$, i.e., $\mathbf{\theta} \sim p(\mathbf{\theta}|\mathbf{\alpha})$.
- $\mathbf{Z}$: the latent variables of the data point's distribution, i.e., $\mathbf{x} \sim p(\mathbf{x}|\mathbf{Z})$.
- $q(\mathbf{Z})$: variational distribution, a distribution over the latent variables $\mathbf{Z}$, i.e., $P(\mathbf{Z}|\mathbf{X}) \approx q(\mathbf{Z})$.

### Bayesian Inference
- **Prior**: $p(\mathbf{\theta}|\mathbf{\alpha})$, the distribution of the parameters $\mathbf{\theta}$ before observing the data $\mathbf{X}$.
- **Sampling Distribution** and **Likelihood**: $p(\mathbf{X}|\mathbf{\theta})=L(\mathbf{\theta}|\mathbf{X})$, the distribution of the data $\mathbf{X}$ given the parameters $\mathbf{\theta}$, or the likelihood of the parameters $\mathbf{\theta}$ given the data $\mathbf{X}$.
- **Marginal Likelihood (Evidence)**: $p(\mathbf{X}|\mathbf{\alpha})$, the distribution of the observed data $\mathbf{X}$ marginalized over the prior distribution of the parameters $\mathbf{\theta}$, i.e., $p(\mathbf{X}|\mathbf{\alpha}) = \int p(\mathbf{X}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\alpha})d\mathbf{\theta}$.
- **Posterior**: $p(\mathbf{\theta}|\mathbf{X}, \mathbf{\alpha})$, the distribution of the parameters $\mathbf{\theta}$ after observing the data $\mathbf{X}$. It is proportional to the product of the likelihood and the prior, i.e., $p(\mathbf{\theta}|\mathbf{X}, \mathbf{\alpha}) \propto p(\mathbf{X}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\alpha})$.
- **Bayes' Rule**: $p(\mathbf{\theta}|\mathbf{X}, \mathbf{\alpha}) = \frac{p(\mathbf{X}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\alpha})}{p(\mathbf{X}|\mathbf{\alpha})}\propto p(\mathbf{X}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\alpha})$.

### Bayesian Prediction
- **Posterior Predictive Distribution**: $p(\tilde{\mathbf{x}}|\mathbf{X}, \mathbf{\alpha}) = \int p(\tilde{\mathbf{x}}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{X}, \mathbf{\alpha})d\mathbf{\theta}$, where $p(\tilde{\mathbf{x}}|\mathbf{\theta})$ is the distribution of the new data point $\tilde{\mathbf{x}}$ given the parameters $\mathbf{\theta}$, and $p(\mathbf{\theta}|\mathbf{X}, \mathbf{\alpha})$ is the posterior distribution of the parameters $\mathbf{\theta}$ given the observed data $\mathbf{X}$.
- **Prior Predictive Distribution**: $p(\tilde{\mathbf{x}}|\mathbf{\alpha}) = \int p(\tilde{\mathbf{x}}|\mathbf{\theta})p(\mathbf{\theta}|\mathbf{\alpha})d\mathbf{\theta}$, where $p(\tilde{\mathbf{x}}|\mathbf{\theta})$ is the distribution of the new data point $\tilde{\mathbf{x}}$ given the parameters $\mathbf{\theta}$, and $p(\mathbf{\theta}|\mathbf{\alpha})$ is the prior distribution of the parameters $\mathbf{\theta}$.

### Variational Inference
- **KL Divergence**: $\text{KL}(q||p) = \int q(\mathbf{Z})\log\frac{q(\mathbf{Z})}{p(\mathbf{Z})}d\mathbf{Z}$, where $q(\mathbf{Z})$ and $p(\mathbf{Z})$ are two distributions over the latent variables $\mathbf{Z}$.
- **Evidence Lower Bound (ELBO)**: $\mathcal{L}(q)=\text{ELBO}(q) = \int q(\mathbf{Z})\log\frac{p(\mathbf{X}, \mathbf{Z})}{q(\mathbf{Z})}d\mathbf{Z} = \int q(\mathbf{Z})\log p(\mathbf{X}|\mathbf{Z})d\mathbf{Z} - \text{KL}(q(\mathbf{Z})||p(\mathbf{Z}))$, where $p(\mathbf{X}, \mathbf{Z}) = p(\mathbf{X}|\mathbf{Z})p(\mathbf{Z})$ is the joint distribution of the observed data $\mathbf{X}$ and the latent variables $\mathbf{Z}$, and $p(\mathbf{Z})$ is the prior distribution of the latent variables $\mathbf{Z}$.
- **Log-Evidence**: $\log p(\mathbf{X}) = \text{ELBO}(q) + \text{KL}(q(\mathbf{Z})||p(\mathbf{Z}|\mathbf{X}))$.
- **Variational Inference**: $q^*(\mathbf{Z}) = \arg\min_{q(\mathbf{Z})} \text{KL}(q(\mathbf{Z})||p(\mathbf{Z}|\mathbf{X}))= \arg\max_{q(\mathbf{Z})} \text{ELBO}(q)$.
- **Mean Field Approximation**: $q(\mathbf{Z}) = \prod_{i=1}^n q_i(\mathbf{Z}_i)$, where $\mathbf{Z}_i$ is the $i$-th latent variable.
- **Coordinate Ascent Variational Inference**: $q_i^*(\mathbf{Z}_i) = \arg\max_{q_i(\mathbf{Z}_i)} \text{ELBO}(q)$, where $q_{-i}(\mathbf{Z}_{-i}) = \prod_{j\neq i}q_j(\mathbf{Z}_j)$ is fixed.
- **Expectation Maximization with Mean Field Approximation**: $q_i^*(\mathbf{Z}_i) = \arg\max_{q_i(\mathbf{Z}_i)} \mathbb{E}_{q_{-i}(\mathbf{Z}_{-i})}[\text{ELBO}(q)]$, where $q_{-i}(\mathbf{Z}_{-i}) = \prod_{j\neq i}q_j(\mathbf{Z}_j)$ is updated by $q_{-i}^*(\mathbf{Z}_{-i}) = \arg\max_{q_{-i}(\mathbf{Z}_{-i})} \mathbb{E}_{q_i(\mathbf{Z}_i)}[\text{ELBO}(q)]$. In practice, we usually work in the log space, i.e., $\log q_i^*(\mathbf{Z}_i|\mathbf{X}) = \mathbb{E}_{q_{-i}(\mathbf{Z}_{-i})}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$.

