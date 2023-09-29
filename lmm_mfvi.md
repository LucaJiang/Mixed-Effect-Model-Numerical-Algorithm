# Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model
- [Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model](#mean-field-variational-inference-mfvi-algorithm-for-mixed-effect-model)
  - [Variational Inference](#variational-inference)
  - [Mean-Field Variational Inference (MFVI)](#mean-field-variational-inference-mfvi)
  - [Calculation of ELBO](#calculation-of-elbo)
  - [EM Algorithm with MFVI](#em-algorithm-with-mfvi)
  - [References](#references)

In the previous section, we have applied the EM algorithm to mixed effect model. In this section, we will use the MFVI algorithm to estimate the distributions of latent variables instead of using the MLE in E-step.

## Variational Inference
Assume $\Theta=\{\mathbf{\omega}, \sigma_\beta^2, \sigma_e^2\}$ has been estimated in the M-step. In order to use MFVI to find a $\mathbf{q(\mathbf{\beta})}$ which approximate the true posterior $p(\mathbf{\beta}|\mathbf{y})$. We need to find the optimal $\mathbf{q(\mathbf{\beta})}$ that minimizes the KL divergence between $\mathbf{q(\mathbf{\beta})}$ and $p(\mathbf{\beta}|\mathbf{y})$, which is equivalent to maximizing the ELBO (Evidence Lower BOund) function:
$$
\begin{equation}\begin{split}
\mathcal{L}(\mathbf{q}) &=: \mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log \frac{p(\mathbf{y}, \mathbf{q(\mathbf{\beta})})}{\mathbf{q(\mathbf{\beta})}}] \\
&=\mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log p(\mathbf{y})] - \text{KL}(\mathbf{q(\mathbf{\beta})}||p(\mathbf{\beta}))
\end{split}\end{equation}$$


## Mean-Field Variational Inference (MFVI)
In the MFVI algorithm, let $\mathbf{q(\mathbf{\beta})} = \prod_{i=1}^{n}q(\beta_i)$ and $q(\beta_i) = \mathcal{N}(\mu_i, \sigma_\beta^2)$. Then with the coordinate ascent variational inference, we can update $q(\beta_i)$ by fixing all other $q(\mathbf{\beta}_j)$ for $j\neq i$. Thus, we have:
$$
\begin{equation} \begin{split}
\mathcal{L}(\mathbf{q}) &= \mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log p(\mathbf{y})+\log p(\mathbf{\beta}|\mathbf{y}) + \log \mathbf{q(\mathbf{\beta})}] \\
&= \log p(\mathbf{y}) + \sum_{i=1}^{n}\mathbb{E}_{q(\beta_i)}[\log p(\beta_i|\mathbf{\beta}_{-i}, \mathbf{y})] - \sum_{i=1}^{n}\mathbb{E}_{q(\beta_i)}[\log q(\beta_i)]
\end{split}\end{equation}$$
where $\mathbf{\beta}_{-i} = \{\mathbf{\beta}_j\}_{j\neq i}$.

Find the derivative of $\mathcal{L}(\mathbf{q})$ with respect to $\beta_i$ and set it to zero, we have:
$$
\begin{equation}\begin{split}
\frac{\partial \mathcal{L}(\mathbf{q})}{\partial \beta_i} &= \mathbb{E}_{-i} [\log p(\beta_i|\mathbf{\beta}_{-i}, \mathbf{y})] - \log q(\beta_i) -1 =: 0 \\
\Rightarrow q^*(\beta_i) &\propto \exp \{\mathbb{E}_{-i} [\log p(\mathbf{\beta}, \mathbf{y})]\} \\
&= \exp \{\mathbb{E}_{-i} [\log p(\mathbf{y}|\mathbf{\beta}) + \log p(\mathbf{\beta})]\}
\end{split}\end{equation}$$

Since
$$\begin{equation}
\begin{split}
\log p(\mathbf{y}| \mathbf{\beta})
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2
\end{split}\end{equation}$$
and
$$\begin{equation}
\begin{split}
\log p(\mathbf{\beta}_{-i}| \mathbf{\mu}_{-i}) &= -\frac{p-1}{2} \log (2\pi) -\frac{p-1}{2} \log \sigma_\beta^2 \\&\quad - \frac{1}{2\sigma_\beta^2} \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)^T \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)
\end{split}\end{equation}$$

So the update equations for $\mu_i$ is:
$$
\begin{equation}\begin{split}
\mu_i =\left(\frac{\mathbf{X}^T_{-i} \mathbf{X}_{-i}}{\sigma_e^2} + \frac{\mathbf{I}_{p-1}}{\sigma_\mathbf{\beta}^2} \right)^{-1} \left(\frac{\mathbf{X}^T_{-i}\left(\mathbf{y}_{-i}- \mathbf{Z}_{-i}\mathbf{\omega}_{-i}\right)}{\sigma_e^2} + \frac{\mathbf{\mu}_{-i}}{\sigma_\mathbf{\beta}^2} \right)\mathbf{1}_{p-1}/(p-1)
\end{split}\end{equation}$$

## Calculation of ELBO
To calculate the ELBO, we need to compute the expected log joint probability of the data and the model parameters under the variational distribution, i.e.,

$$ \mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y}, \mathbf{\beta}|\mathbf{\Theta})] = \int \log p(\mathbf{y}, \mathbf{\beta}|\mathbf{\Theta}) q(\mathbf{\beta}) d\mathbf{\beta} $$

This integral is generally intractable, so we use Monte Carlo integration to approximate it. Specifically, we draw $S$ samples $\mathbf{\beta}^{(s)}$ from the variational distribution, and estimate the integral as

$$ \frac{1}{S} \sum_{s=1}^S \log p(\mathbf{y}, \mathbf{\beta}^{(s)}|\mathbf{\Theta}) $$

To compute this estimate, we need to evaluate the likelihood function $p(\mathbf{y}|\mathbf{X}, \mathbf{\beta}^{(s)}, \mathbf{\Theta})$ and the prior distribution $p(\mathbf{\beta}^{(s)}|\mathbf{\Theta})$ for each sample $\mathbf{\beta}^{(s)}$.

Finally, we need to compute the entropy of the variational distribution, i.e.,

$$ \mathbb{H}[q(\mathbf{\beta})] = -\int q(\mathbf{\beta}) \log q(\mathbf{\beta}) d\mathbf{\beta} = \frac{p}{2} (1 + \log(2\pi)) + \frac{\sigma_\beta^2}{2}$$

The ELBO is then given by the sum of the expected log joint probability and the negative entropy of the variational distribution, i.e.,

$$ \mathcal{L}(\mathbf{\mu}) = \mathbb{E}_{q(\mathbf{\beta})}[\log p(\mathbf{y}, \mathbf{\beta}|\mathbf{\theta})] - \mathbb{H}[q(\mathbf{\beta})] $$

## EM Algorithm with MFVI
Using the following algorithm to estimate $\mathbb{E}(\mathbf{\beta})$ in the 

1. Initialize $\Theta^{(0)}$ and $\mathbf{\mu}^{(0)}$ randomly.
2. For $t = 0, 1, \dots$, MAX_ITERATION:
   1. E-step: Estimate $\hat{\mathbf{\beta}}^{(t)}$:For $t=1,2,\dots$:
      $\qquad$ Update $\mathbf{\mu}^{(t)}$.
    2. M-step: Estimate $\hat{\Theta}^{(t+1)}=\{\hat{\mathbf{\omega}}^{(t+1)}, \hat{\sigma}_\beta^{2(t+1)}, \hat{\sigma}_e^{2(t+1)}\}$.
3. Compute $\text{ELBO}(\mathbf{\mu}) = \mathbb{E}\left[\log p(\mathbf{y}, \mathbf{\beta})\right] - \mathbb{E}\left[\log q(\mathbf{\beta})\right]$. 
4. Check $|\Delta \ell_c|$ for convergence. If converged, stop. Otherwise, continue.

## References
1.  Blei D M, Kucukelbir A, McAuliffe J D. [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) [J]. Journal of the American statistical Association, 2017, 112(518): 859-877.



