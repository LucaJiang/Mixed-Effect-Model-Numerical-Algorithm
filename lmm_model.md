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

Let's assume $\mathbf{\gamma}$ can be break down into a fixed effect part $\mathbf{\omega}$ and a random effect part $\mathbf{\beta}\sim \mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_p)$, i.e., $\mathbf{\gamma} = \mathbf{\omega} + \mathbf{\beta}$. Then, for an individual $i$, we have
$$\begin{equation}
y_i = \mathbf{x}_i^T\mathbf{\omega} + \mathbf{x}_i^T\mathbf{\beta}_i + e_i
\end{equation}$$
Thus, we obtain the LMM (Linear Mixed-Effect Model), which relaxes the assumption of i.i.d.

## Model Description
LMM is a statistical model that accounts for both fixed effects and random effects in a linear regression model. It is used for modeling data where observations are not independent or identically distributed.

Consider a dataset $\{\mathbf{y}, \mathbf{X},\mathbf{Z}\}$ with $n$ samples, where $\mathbf{y} \in \mathbb{R}^n$ is the vector of response variable, $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the matrix of $p$ independent variables, and $\mathbf{Z} \in \mathbb{R}^{n \times c}$ is another matrix of $c$ variables. The linear mixed model builds upon a linear relationship from $\mathbf{y}$ to $\mathbf{X}$ and $\mathbf{Z}$ by
$$\begin{equation}
\mathbf{y} = \underbrace{\mathbf{Z}\mathbf{\omega}}_{\text {fixed}} + \underbrace{\mathbf{X}\mathbf{\beta}}_{\text {random}} + \underbrace{\mathbf{e}}_{\text {error}},
\end{equation}$$
where $\mathbf{\omega} \in \mathbb{R}^c$ is the vector of fixed effects, $\mathbf{\beta} \in \mathbb{R}^p$ is the vector of random effects with $\mathbf{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma^2_\mathbf{\beta} \mathbf{I}_p)$, and $\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I}_n)$ is the independent noise term. 