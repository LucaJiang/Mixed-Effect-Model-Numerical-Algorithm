# Mean-Field Variational Inference Algorithm for Mixed Effect Model

- [Mean-Field Variational Inference Algorithm for Mixed Effect Model](#mean-field-variational-inference-algorithm-for-mixed-effect-model)
  - [Variational Inference](#variational-inference)
  - [Mean-Field Variational Inference](#mean-field-variational-inference)
    - [E-step](#e-step)
    - [Q-function and ELBO](#q-function-and-elbo)
    - [M-step](#m-step)
  - [EM Algorithm with MFVI](#em-algorithm-with-mfvi)
  - [Code and Results](#code-and-results)
  - [References](#references)

In the previous section, we have applied the EM algorithm to the mixed effect model. In this section, we will use the MFVI algorithm to estimate the distributions of latent variables instead of using the MLE in E-step. And then, we will use the EM algorithm to estimate the parameters in M-step.

## Variational Inference

Assume $\Theta=\{\mathbf{\omega}, \sigma_\beta^2, \sigma_e^2\}$ has been estimated in the M-step. In order to use Mean-Field Variational Inference (MFVI) to find a $q(\mathbf{\beta})$ which approximate the true posterior $p(\mathbf{\beta}|\mathbf{y})$. We need to find the optimal $q(\mathbf{\beta})$ that minimizes the KL divergence between $q(\mathbf{\beta})$ and $p(\mathbf{\beta}|\mathbf{y})$, which is equivalent to maximizing the ELBO (Evidence Lower BOund) function:
$$\begin{equation}\begin{split}
\text{ELBO}(\mathbf{q}) &=: \mathbb{E}_{q(\mathbf{\beta})}[\log \frac{p(\mathbf{y}, q(\mathbf{\beta}))}{q(\mathbf{\beta})}] \\
&=\mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y}), \mathbf{\beta}] - \text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta}|\mathbf{y}))\\
\text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta})) &= -\mathbb{E}_{q(\mathbf{\beta})}[\log \frac{p(\mathbf{\beta}|\mathbf{y})}{q(\mathbf{\beta})}] \\
&= -\mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{\beta}|\mathbf{y})] + \mathbb{E}_{q(\mathbf{\beta})}[\log q(\mathbf{\beta})]
\end{split}\end{equation}$$

where $\text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta}))$ is the KL divergence between $q(\mathbf{\beta})$ and $p(\mathbf{\beta}|\mathbf{y})$. And $\mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y})]$ is a constant with respect to $q(\mathbf{\beta})$.

## Mean-Field Variational Inference

### E-step

In the MFVI algorithm, we will make the following mean field assumption: $$\begin{equation}q(\mathbf{\beta}) = \prod_{j=1}^{p}q_j(\beta_j)\end{equation}$$
where $q_j(\beta_j)$ is the posterior distribution of $\beta_j$.

Since the complete data log-likelihood is given by
$$\begin{equation}\begin{split}
\ell(\mathbf{\Theta}, \mathbf{\beta}) &= \log p(\mathbf{y}, \mathbf{\beta}| \mathbf{\Theta})\\
&= \log p(\mathbf{y}| \mathbf{\beta}, \mathbf{\Theta}) + \log p(\mathbf{\beta}| \mathbf{\Theta})\\
&= -\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2\\
&\quad -\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}
\end{split}\end{equation}$$
we can calculate the expectation of the complete data log-likelihood with respect to $q(\mathbf{\beta})$:
$$\begin{equation}\begin{split}
\mathbb{E}_{q(\mathbf{\beta})}[\ell(\mathbf{\Theta}, \mathbf{\beta})] &= \mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y}, \mathbf{\beta}| \mathbf{\Theta})] \\
&\propto \mathbb{E}_{q(\mathbf{\beta})}[-\frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2 - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}] \\
&\propto \sum_{j=1}^{p} \mathbb{E}_{q(\beta_j)}\left[-\frac{1}{2\sigma_e^2} \left( \mathbf{X}_{.j}^T \mathbf{X}_{.j} \beta_j^2 +2\mathbf{X}_{.j}^T\mathbf{X}_{.-j}\mathbf{\beta}_{-j}\beta_j\right.\right.\\
&\quad\left.\left. - 2\left(\left(\mathbf{y}-\mathbf{Z}\mathbf{\omega}\right)^T \mathbf{X}\right)_{.j}\beta_j\right)- \frac{1}{2\sigma_\beta^2} \beta_j^2\right] \\
&\propto \sum_{j=1}^{p} \mathbb{E}_{q(\beta_j)}\left[-\frac{\left(\beta_j-\mu_j\right)^2}{2\sigma_j^2}\right]
\end{split}\end{equation}$$
where $\mathbf{X}_{.j}$ is the $j$ th column of $\mathbf{X}$, $\mathbf{X}_{j.}$ is the $j$ th row of $\mathbf{X}$, $\sigma_j^2=1/\left(\frac{\mathbf{X}_{.j}^T \mathbf{X}_{.j}}{\sigma_e^2} + \frac{1}{\sigma_\beta^2}\right)$ and $\mu_j=\sigma_j^2/\sigma_e^2 \left(\mathbf{y}-\mathbf{Z}\mathbf{\omega} -\mathbf{X}_{.-j}\mathbf{\mu}_{-j}\right)^T\mathbf{X}_{.j}$. And $\mathbf{X}_{.-j}$ is the matrix $\mathbf{X}$ where the $j$ th column is 0, and $\mathbf{\mu}_{-j}$ is the vector $\mathbf{\mu}$ where the $j$ th element is 0.

Therefore, the posterior distribution of $\beta_j$ is given by $\mathcal{N}(\mu_j, \sigma_j^2) $. And the posterior distribution of $\mathbf{\beta}$ is given by $\mathcal{N}(\mathbf{\mu}, \mathbf{\Gamma}) $, where $\mathbf{\mu}=[\mu_1, \dots, \mu_p]^T$ and $\mathbf{\Gamma}=\text{diag}(\sigma_1^2, \dots, \sigma_p^2)=\left(\frac{\text{diag}(\mathbf{X}^T \mathbf{X})}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\beta^2}\right)^{-1}$.

### Q-function and ELBO
To calculate the Q-function, we need to calculate the expectation of the complete data log-likelihood with respect to the posterior distribution of $\mathbf{\beta}$. Therefore, we need to calculate the following expectation:
$$\begin{equation}
\begin{split}
\mathbb{E}_{q(\mathbf{\beta})}[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2] &= \mathbb{E}_{q(\mathbf{\beta})}[\|(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega})-2(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\beta} + \mathbf{\beta}^T \mathbf{X}^T \mathbf{X}\mathbf{\beta}\|] \\
&= (\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega}) - 2(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu} \\&\quad + \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})+ \mathbf{\mu}^T \mathbf{X}^T \mathbf{X}\mathbf{\mu}\\
\mathbb{E}_{q(\mathbf{\beta})}[\mathbf{\beta}^T \mathbf{\beta}] &= \text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}
\end{split}\end{equation}$$
<!-- ???
$\mu_j=\sigma_j^2/\sigma_e^2 \mathbf{X}_{.j}^T \left(\mathbf{y}-\mathbf{Z}\mathbf{\omega} +\mathbf{X}_{.-j}\mathbf{\beta}_{-j}/2\right)$, $\mathbf{\mu}= \mathbf{\Gamma} \mathbf{X}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega}+\mathbf{X}\mathbf{\mu}(1-\mathbf{1}_p)/2)/\sigma_e^2$.
where $1-\mathbf{1}_p=
\begin{pmatrix}
0 & 1 & \cdots & 1 \\
1 & 0 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 0
\end{pmatrix}_{p*p}$
??? -->

Note that $\mathbb{E}[\mathbf{\beta}^T \mathbf{\beta}] = \text{tr}(\mathbb{V}[\mathbf{\beta}]) + \mathbb{E}[\mathbf{\beta}]^T \mathbb{E}[\mathbf{\beta}] = \text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}$ and $\mathbb{E}[(\mathbf{X}\mathbf{\beta})^T (\mathbf{X}\mathbf{\beta})] = \text{tr}(\mathbf{X}\mathbf{\Gamma} \mathbf{X}^T) + (\mathbf{X}\mathbf{\mu})^T (\mathbf{X}\mathbf{\mu}).$

Then, we can calculate the Q-function:
$$\begin{equation}\begin{split}
\mathcal{Q}(\mathbf{q}) &= \mathbb{E}_{q(\mathbf{\beta})}[\ell(\mathbf{\Theta}, \mathbf{\beta}) ] \\
&= \mathbb{E}_{q(\mathbf{\beta})}[-\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2\\
&\quad -\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}] \\
&= -\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \mathbb{E}_{q(\mathbf{\beta})}[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2]\\
&\quad -\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \mathbb{E}_{q(\mathbf{\beta})}[\mathbf{\beta}^T \mathbf{\beta}] \\
&= -\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \left[(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega}) - 2(\mathbf{y} - \mathbf{Z}\mathbf{\omega})^T \mathbf{X}\mathbf{\mu} \right.\\
&\quad \left.+ \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})+ (\mathbf{\mu}^T \mathbf{X}^T \mathbf{X}\mathbf{\mu})\right]-\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \left[\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}\right]\\
&= -\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})\right]\\
&\quad-\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \left[\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}\right]\\
\end{split}\end{equation}$$
and the ELBO:
$$\begin{equation}\begin{split}
\text{ELBO} &= \mathcal{Q}(\mathbf{q}) - \mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{\beta}| \mathbf{\Theta})] \\
&= \mathcal{Q}(\mathbf{q}) - \mathbb{E}_{q(\mathbf{\beta})}[-\frac{p}{2} \log (2\pi \sigma_\beta^2 ) - \frac{1}{2\sigma_\beta^2} \mathbf{\beta}^T \mathbf{\beta}] \\
&= \mathcal{Q}(\mathbf{q}) +\frac{p}{2} \log (2\pi \sigma_\beta^2 ) + \frac{1}{2\sigma_\beta^2} \left[\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}\right]\\
&= -\frac{n}{2} \log (2\pi \sigma_e^2) - \frac{1}{2\sigma_e^2} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})\right]
\end{split}\end{equation}$$

$$\begin{equation}\begin{split}
\text{ELBO} &= \mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y}, \mathbf{\beta}| \mathbf{\Theta})] - \text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta})) ???\\
&= \mathcal{Q}(\mathbf{q}) + \frac{1}{2}\log |2\pi\mathbf{\Gamma}|
\end{split}\end{equation}$$

Note: The KL divergence between the posterior $q(\mathbf{\beta})$ and the prior $p(\mathbf{\beta})$ is given by:[^1]
[^1]: [KL-divergence between 2 gaussian distributions](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)
$$\begin{equation}\begin{split}
\text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta}))
&=\frac{1}{2}\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - p + (\boldsymbol{\mu_p}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\boldsymbol{\mu_p}-\boldsymbol{\mu_q}) + tr\left\{\Sigma_q^{-1}\Sigma_p\right\}\right]
\end{split}\end{equation}$$

$$\begin{equation}\begin{split}
\text{KL}(q(\mathbf{\beta})||p(\mathbf{\beta}))
% &=\frac{1}{2}\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - p + (\boldsymbol{\mu_p}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\boldsymbol{\mu_p}-\boldsymbol{\mu_q}) + tr\left\{\Sigma_q^{-1}\Sigma_p\right\}\right]\\
&= \frac{1}{2}\left[\log\frac{|\mathbf{\Gamma}|}{\sigma_\beta^{2p}}-p+\mathbf{\mu}^T\mathbf{\Gamma}^{-1}\mathbf{\mu}+\sigma_\beta^{2}\text{tr}\left(\mathbf{\Gamma}^{-1}\right)\right]\\
&= \frac{1}{2}\left[\sum_{j=1}^{p}\log\sigma_j^2 -p\log\sigma_\beta^2+\sum_{j=1}^{p}\frac{\mu_j^2}{\sigma_j^2}+\sum_{j=1}^{p}\frac{\sigma_\beta^{2}}{\sigma_j^2}\right]
\end{split}\end{equation}$$

### M-step
In the M-step, we need to find the optimal $\Theta$ that maximizes the Q-function.
<!-- $\mathbf{\mu}=[\mu_1, \dots, \mu_p]^T= \mathbf{\Gamma} \mathbf{X}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega}+\mathbf{X}\mathbf{\mu}/2)/\sigma_e^2$ and $\mathbf{\Gamma}=\text{diag}(\sigma_1^2, \dots, \sigma_p^2)=\left(\frac{\text{diag}(\mathbf{X}^T \mathbf{X})}{\sigma_e^2} + \frac{\mathbf{I}_p}{\sigma_\beta^2}\right)^{-1}$. -->

$$\begin{equation}\begin{split}
\frac{\partial \mathcal{Q}}{\partial \sigma_\beta^2} &= -\frac{p}{2\sigma_\beta^2} + \frac{1}{2\sigma_\beta^4} \left[\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}\right]=:0 \\
\Rightarrow \hat{\sigma}_\beta^2 &= \frac{1}{p} \left[\text{tr}(\mathbf{\Gamma}) + \mathbf{\mu}^T \mathbf{\mu}\right]=\frac{1}{p} \left[\sum_{j=1}^{p} \sigma_j^2 + \mathbf{\mu}^T \mathbf{\mu}\right]\\
\frac{\partial \mathcal{Q}}{\partial \sigma_e^2} &= -\frac{n}{2\sigma_e^2} + \frac{1}{2\sigma_e^4} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})\right]=:0 \\
\Rightarrow \hat{\sigma}_e^2 &= \frac{1}{n} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \text{tr}(\mathbf{X}^T \mathbf{\Gamma}\mathbf{X})\right]\\
&=\frac{1}{n} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \text{tr}(\mathbf{X}^T\mathbf{X} \mathbf{\Gamma})\right]\\
&= \frac{1}{n} \left[\|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 + \sum_{j=1}^{p} \text{diag}(\mathbf{X}^T\mathbf{X})_j \sigma_j^2\right]\\
\frac{\partial \mathcal{Q}}{\partial \mathbf{\omega}} &= \frac{1}{\sigma_e^2} \mathbf{Z}^T (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}) =: 0 \\
\Rightarrow \hat{\mathbf{\omega}} &= \left(\mathbf{Z}^T \mathbf{Z}\right)^{-1} \mathbf{Z}^T (\mathbf{y} - \mathbf{X}\mathbf{\mu})\\
\end{split}\end{equation}$$

## EM Algorithm with MFVI
Using the following algorithm to estimate $\mathbb{E}(\mathbf{\beta})$ in the

1. Initialize $\Theta^{(0)}$ and $\mathbf{\mu}^{(0)}$ randomly.
2. For $t = 0, 1, \dots$, MAX_ITERATION:
   1. E-step: Estimate $\hat{\mathbf{\beta}}^{(t)}$.
   2. Compute $\text{ELBO}$
   3. M-step: Estimate $\hat{\Theta}^{(t+1)}=\{\hat{\mathbf{\omega}}^{(t+1)}, \hat{\sigma}_\beta^{2(t+1)}, \hat{\sigma}_e^{2(t+1)}\}$.
   4. Compute $\Delta \ell_c$. Check $|\Delta \ell_c|$ for convergence. If converged, stop. Otherwise, continue.
3. Return results.

## Code and Results
The following results are obtained by running the EM + MFVI algorithm on the given dataset.
[Link to code](https://lucajiang.github.io/Mixed-Effect-Model-Numerical-Algorithm/mfvi_result)

Calculation details:
ToDO

## References
1.  Blei D M, Kucukelbir A, McAuliffe J D. [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) [J]. Journal of the American statistical Association, 2017, 112(518): 859-877.

2. https://jchiquet.github.io/MAP566/docs/mixed-models/map566-lecture-linear-mixed-model.html
