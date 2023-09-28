# Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model
- [Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model](#mean-field-variational-inference-mfvi-algorithm-for-mixed-effect-model)
  - [Variational Inference](#variational-inference)
  - [Mean-Field Variational Inference (MFVI)](#mean-field-variational-inference-mfvi)
  - [Algorithm for Estimating the Posterior Mean of Random Effects](#algorithm-for-estimating-the-posterior-mean-of-random-effects)

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
\begin{equation}\begin{split}
\mathcal{L}(\mathbf{q}) &= \mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log p(\mathbf{y})+\log p(\mathbf{q(\mathbf{\beta})}|\mathbf{y} ) + \log \mathbf{q(\mathbf{\beta})}] \\
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
\log p(\mathbf{y}| \mathbf{\beta}) &= -\frac{n}{2} \log (2\pi) -\frac{1}{2} \log |\sigma_e^2\mathbf{I}_n| - \frac{1}{2} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})^T (\sigma_e^2\mathbf{I}_n)^{-1} (\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta})\\
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2
\end{split}\end{equation}$$
and
$$\begin{equation}
\begin{split}
\log p(\mathbf{\beta}_{-i}| \mathbf{\mu}_{-i}) &= -\frac{p-1}{2} \log (2\pi) -\frac{p-1}{2} \log \sigma_\beta^2 \\&\quad - \frac{1}{2\sigma_\beta^2} \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)^T \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)
\end{split}\end{equation}$$

And the update equations for $\mu_i$ is:
$$
\begin{equation}\begin{split}
\mu_i =\left(\frac{\mathbf{X}^T_{-i} \mathbf{X}_{-i}}{\sigma_e^2} + \frac{\mathbf{I}_{p-1}}{\sigma_\mathbf{\beta}^2} \right)^{-1} \left(\frac{\mathbf{X}^T_{-i}\left(\mathbf{y}_{-i}- \mathbf{Z}_{-i}\mathbf{\omega}_{-i}\right)}{\sigma_e^2} + \frac{\mathbf{\mu}_{-i}}{\sigma_\mathbf{\beta}^2} \right)\mathbf{1}_{p-1}/(p-1)
\end{split}\end{equation}$$

## Algorithm for Estimating the Posterior Mean of Random Effects
Using the following algorithm to estimate $\mathcal{E}(\mathbf{\beta})$ in the E-step of EM algorithm:
1. Initialize $\mathbf{\mu}^{(0)}$ and $\mathbf{\sigma}^{(0)}$.
2. For $t=1,2,\dots$:
   1. Update $\mathbf{\mu}^{(t)}$ and $\mathbf{\sigma}^{(t)}$ using the update equations above.
   2. If $\|\mathbf{\mu}^{(t)} - \mathbf{\mu}^{(t-1)}\| < \varepsilon$, stop. Otherwise, continue.



