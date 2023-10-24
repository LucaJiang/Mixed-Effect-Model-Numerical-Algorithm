# Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model
- [Mean-Field Variational Inference (MFVI) Algorithm for Mixed Effect Model](#mean-field-variational-inference-mfvi-algorithm-for-mixed-effect-model)
  - [Variational Inference](#variational-inference)
  - [Mean-Field Variational Inference (MFVI)](#mean-field-variational-inference-mfvi)
  - [Calculation of ELBO](#calculation-of-elbo)
  - [EM Algorithm with MFVI](#em-algorithm-with-mfvi)
  - [Code and Results](#code-and-results)
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
In the MFVI algorithm, let $\mathbf{q(\mathbf{\beta})} = \prod_{i=1}^{p}q(\beta_i)$ and $q(\beta_i) = \mathcal{N}(\mu_i, \sigma_\beta^2)$. Then with the coordinate ascent variational inference, we can update $q(\beta_i)$ by fixing all other $q(\mathbf{\beta}_j)$ for $j\neq i$. Thus, we have:
$$
\begin{equation} \begin{split}
\mathcal{L}(\mathbf{q}) &= \mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log p(\mathbf{y})+\log p(\mathbf{\beta}|\mathbf{y}) + \log \mathbf{q(\mathbf{\beta})}] \\
&= \log p(\mathbf{y}) + \sum_{i=1}^{p}\mathbb{E}_{q(\beta_i)}[\log p(\beta_i|\mathbf{\beta}, \mathbf{y})] - \sum_{i=1}^{p}\mathbb{E}_{q(\beta_i)}[\log q(\beta_i)]
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

So the update equations for $\mathbf{\mu}$ is:
$$\begin{equation}
\mathbf{\mu}^{(t+1)} =\left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_{p}}{\sigma_\mathbf{\beta}^2} \right)^{-1} \left(\frac{\mathbf{X}^T\left(\mathbf{y}- \mathbf{Z}\mathbf{\omega}\right)}{\sigma_e^2} + \frac{\mathbf{\mu}^{(t)}}{\sigma_\mathbf{\beta}^2} \right)
\end{equation}$$

Previous equation can be rewritten as:
$$\begin{equation}
\mathbf{\mu} =\left(\frac{\mathbf{X}^T \mathbf{X}}{\sigma_e^2} + \frac{\mathbf{I}_{p}}{\sigma_\mathbf{\beta}^2} \right)^{-1} \left(\frac{\mathbf{X}^T\mathbf{X}}{\sigma_e^2} \left(\mathbf{X}^T \mathbf{X}\right)^{-1}\mathbf{X}^T \left(\mathbf{y}- \mathbf{Z}\mathbf{\omega}\right) + \frac{\mathbf{\beta}}{\sigma_\mathbf{\beta}^2} \right)
\end{equation}$$
which is a weighted average of the MLE solve of $\mathbf{y}-\mathbf{Z}\mathbf{\omega} = \mathbf{X}\mathbf{\beta}$ and the prior $\mathbf{\beta} \sim \mathcal{N}(\mathbf{0}, \sigma_\mathbf{\beta}^2 \mathbf{I}_{p})$.  

## Calculation of ELBO
To calculate the ELBO after each E-step, we need to calculate the following terms:
$$\begin{equation}
\begin{split}
\mathbb{E}_{q(\beta_i)}[\log p(\beta_i|\mathbf{\beta}_{-i}, \mathbf{y})] &= \mathbb{E}_{q(\beta_i)}[\log p(\beta_i|\mathbf{y})] \\
&= \mathbb{E}_{q(\beta_i)}[\log p(\mathbf{y}|\beta_i) + \log p(\beta_i)] \\
&= \mathbb{E}_{q(\beta_i)}[-\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\beta}\|^2] \\
&\quad + \mathbb{E}_{q(\beta_i)}[-\frac{p-1}{2} \log (2\pi) -\frac{p-1}{2} \log \sigma_\beta^2 \\&\quad - \frac{1}{2\sigma_\beta^2} \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)^T \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)] \\
&= -\frac{n}{2} \log (2\pi) -\frac{n}{2} \log \sigma_e^2 - \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 \\
&\quad -\frac{p-1}{2} \log (2\pi) -\frac{p-1}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)^T \left(\mathbf{\beta}_{-i}-\mathbf{\mu}_{-i}\right)
\end{split}\end{equation}$$

$$\begin{equation}
\begin{split}
\mathbb{E}_{q(\beta_i)}[\log q(\beta_i)] &= \mathbb{E}_{q(\beta_i)}[-\frac{1}{2} \log (2\pi) -\frac{1}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \left(\beta_i-\mu_i\right)^2] \\
&= -\frac{1}{2} \log (2\pi) -\frac{1}{2} \log \sigma_\beta^2 - \frac{1}{2\sigma_\beta^2} \mathbb{E}_{q(\beta_i)}\left[\left(\beta_i-\mu_i\right)^2\right] \\
&= -\frac{1}{2} \log (2\pi) -\frac{1}{2} \log \sigma_\beta^2 - \frac{\mu_i^2}{2\sigma_\beta^2} - \frac{1}{2}
\end{split}\end{equation}$$

Since $\mathbf{y}\sim \mathcal{N}(\mathbf{Z}\mathbf{\omega} + \mathbf{X}\mathbf{\beta}, \sigma_e^2 \mathbf{I}_n)$, the ELBO function is:
$$
\begin{equation} \begin{split}
\mathcal{L}(\mathbf{q}) &= \mathbb{E}_{\mathbf{q(\mathbf{\beta})}}[\log p(\mathbf{y})+\log p(\mathbf{\beta}|\mathbf{y}) + \log \mathbf{q(\mathbf{\beta})}] \\
&= \log p(\mathbf{y}) + \sum_{i=1}^{p}\mathbb{E}_{q(\beta_i)}[\log p(\beta_i|\mathbf{\beta}, \mathbf{y})] - \sum_{i=1}^{p}\mathbb{E}_{q(\beta_i)}[\log q(\beta_i)] \\
&= \frac{1}{2} \sum_{i=1}^{n} \log \sigma_\beta^2 + \frac{1}{2} \sum_{i=1}^{n} \frac{\mu_i^2}{\sigma_\beta^2} + \frac{1}{2\sigma_e^2} \|\mathbf{y} - \mathbf{Z}\mathbf{\omega} - \mathbf{X}\mathbf{\mu}\|^2 \\
&\quad +\frac{p-1}{2} \log (2\pi) +\frac{p-1}{2} \log \sigma_\beta^2 + \frac{1}{2\sigma_\beta^2} \left(\mathbf{\beta}-\mathbf{\mu}\right)^T \left(\mathbf{\beta}-\mathbf{\mu}\right) \\
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


