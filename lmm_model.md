# The Introduction of Linear Mixed-Effect Model
- [The Introduction of Linear Mixed-Effect Model](#the-introduction-of-linear-mixed-effect-model)
  - [From OLM to LMM](#from-olm-to-lmm)
  - [Model Description](#model-description)

## From OLM to LMM
The OLM (Ordinary Linear Model) assumes that the observations are independent and identically distributed (i.i.d.). It can be written as
$$\begin{equation}
\mathbf{y} = \mathbf{X}\mathbf{\gamma} + \mathbf{e},
\end{equation}$$
where $\mathbf{y} \in \mathbb{R}^n$ is the vector of response variable, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the matrix of $p$ independent variables, $\mathbf{\gamma} \in \mathbb{R}^p$ is the vector of coefficients, and $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I}_n)$ is the independent noise term.

In the OLM, if we want to obtain the coefficients $\mathbf{\gamma}$, we can use the least square method to minimize the loss function. However, when $n<p$, the matrix $\mathbf{X}$ is not full rank, and the least square method could lead to overfitting. Because we have $p+1$ (including $\sigma_e^2$) parameters to estimate, but only $n$ observations. 

Let's assume the design matrix $\mathbf{X}$ can be decomposed into two parts, i.e., $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2]$, where $\mathbf{X}_1 \in \mathbb{R}^{n \times p_1}, p_1 < n\ll p$ and $\mathbf{X}_2 \in \mathbb{R}^{n \times p_2}$. And, $\mathbf{\gamma}$ can be break down into a fixed effect part $\mathbf{\omega}\in \mathbb{R}^{p_1}$ and a random effect part $\mathbf{\beta}\in \mathbb{R}^{p_1}$ the distribution of which is $\mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_{p_2})$, i.e., $\mathbf{\gamma}^T = [\mathbf{\omega}^T, \mathbf{\beta}^T]$. Then, we have
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