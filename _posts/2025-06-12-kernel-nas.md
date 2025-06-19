---
author: Jasraj Singh
bibliography: refs.bib
csl: custom.csl
date: 12 June, 2025
link-citations: true
title: Kernel Methods for Neural Architecture Search
permalink: /blog-posts/kernel-nas/
tags:
- failed ideas
---

# Summary

The aim of this project was to propose a principled (training-free)
metric for scoring models on a given dataset, thereby introducing a new
strategy for Neural Architecture Search (NAS). The score is defined as
the (kernel) canonical correlation ([Akaho, 2007](#ref-akaho2007kcca);
[Melzer et al., 2001](#ref-thomas2001kcca)) between the inputs,
\\(\mathbf{X}\\), and the outputs, \\(\mathbf{Y}\\), with respect to the
Reproducing Kernel Hilbert Spaces (RKHS) corresponding to the Neural
Tangent Kernel (NTK) and the linear kernel, respectively. We provide a
theoretical motivation for this metric, and aim to validate its efficacy
on NAS benchmarks: NAS-Bench-201 ([Dong et al.,
2020](#ref-dong2020nasbench201)) and DARTS ([Liu et al.,
2019](#ref-liu2018darts)).

# Introduction

Selecting Neural Network (NN) architectures for a task has traditionally
been a manual, time-intensive process requiring domain expertise and
extensive trial-and-error. For example, cross-validation is a popular
choice for model selection, involving training of a number of randomly
initialized models. NAS addresses this challenge by automating the
discovery of high-performing architectures within a predefined search
space ([Poyser et al., 2024](#ref-poyser_2024_nas-review)). Most methods
for NAS incur a high search cost in the form of (partially) training the
architectures to score them, and/or training a \*search model\* that can
make the architecture search efficient. Recently, there has been a
increased focus on cheapening the search process, while retaining or
improving the quality of the selected architectures. Some of these are
NTK-based methods like TE-NAS ([Chen et al., 2021](#ref-chen2020tenas)),
which uses the condition number of the NTK Gram Matrix, and KNAS ([Xu et
al., 2021](#ref-pmlr-v139-xu21m)), which uses the mean of the Matrix's
entries. These studies report competitive performances on real-world
computer vision tasks in NAS benchmarks ([Dong & Yang,
2020](#ref-dong2020nasbench201); [Liu et al., 2019](#ref-liu2018darts);
[Ying et al., 2019](#ref-ying19nasbench101)).

Citing the success of NTK-based NAS methods that use correlation between
architecture scores and the corresponding test accuracy as a measure of
their strategy's efficacy, we aim to propose a scoring method that is
closely correlated to the training loss. As with other NTK-based
methods, it will utilize the Gram matrix associated with the NTK, but
for the purpose of computing the kernel canonical correlation (KCC)
([Akaho, 2007](#ref-akaho2007kcca); [Melzer et al.,
2001](#ref-thomas2001kcca)) between the inputs and the outputs.

# Theory

For simplicity, we assume that we have a univariate regression task at
hand. Consider the set of neural network functions, \\(\mathcal{A}\\),
parameterized by some architecture, \\(\mathbf{A}\\). Most neural networks
designed for regression have a linear output layer, and we assume the
same for \\(\mathbf{A}\\). In that case, minimizing the Mean Squared Error
(MSE) between the network output and the regression labels, is
equivalent to maximizing their correlation, since the parameters of the
output layer can be adjusted post-training to get the minimizer of the
MSE ([Englisch et al., 1994](#ref-englisch_1994_corr-mse)). Accordingly,
we define the score for architecture \\(\mathbf{A}\\) as

$$\begin{aligned}
  S\left(\mathbf{A}\right)
  = \max_{f\in\mathcal{A}} \text{Corr}\left(f\left(\mathbf{X}\right), \mathbf{Y}\right)
  = \max_{f\in\mathcal{A}, g\in\mathcal{L}} \text{Corr}\left(f\left(\mathbf{X}\right), g\left(\mathbf{Y}\right)\right)
\end{aligned}$$

where \\(\mathcal{L}\\) is the space of linear functions on \\(\mathbf{Y}\\).
This optimization is, in general, hard to perform. Instead, if
we were to optimize over some RKHS, then the optimization is equivalent
to performing kernel canonical correlation analysis (KCCA) with the
associated reproducing kernel on \\(\mathbf{X}\\) and the linear kernel on
\\(\mathbf{Y}\\). Accordingly, we seek an RKHS that can approximate the
space of functions the network can converge to.

## Neural Tangent Kernel

Consider an \\(L\\)-layer fully-connected feed-forward network under the NTK
parameterization:

$$\begin{aligned}
  \mathbf{x}^{l+1} = \frac{1}{\sqrt{n_l}} \mathbf{W}^{l+1}\mathbf{x}^l + \mathbf{b}^{l+1}
\end{aligned}$$

with \\(n\_l\\) being the width of layer \\(l\\), and all parameters initialized
i.i.d. from the standard normal distribution,
\\(\mathcal{N}\left(0, 1\right)\\). We define
\\(\mathbf{\theta}^l &#x2254; \text{vec}\left(\left\\{W^l,b^l\right\\}\right)\\)
as the collection of parameters in layer \\(l\\), and
\\(\mathbf{\theta} &#x2254; \text{vec}\left(\cup\_{l=1}^L \mathbf{\theta}^l \right)\\)
as the collection of all parameters.

The parameter dynamics and the predictive dynamics for this model under
gradient flow can be written as:

$$\begin{aligned}
  \dot{\theta}_t &= -\eta \nabla_{\theta} \mathcal{L}\left(\mathcal{D}; \theta_t\right) = -\eta \nabla_{\theta} f\left(\mathbf{X}; \theta_t\right)^T \nabla_{f} \mathcal{L}(\mathcal{D}; \theta_t) \\
  \dot{f}(\mathbf{X}; \theta_t) &= \nabla_{\theta}f(\mathbf{X};\theta_t) \dot{\theta}_t = -\eta \underbrace{\nabla_{\theta} f(\mathbf{X}; \theta_t) \nabla_{\theta} f(\mathbf{X}; \theta_t)^T}_{\triangleq \hat{\Theta}_t(\mathbf{X}, \mathbf{X})} \nabla_{f} \mathcal{L}(\mathcal{D}; \theta_t)
\end{aligned}$$

where \\(\mathcal{D}\\) is the training data set, \\(\mathcal{L}\\) is the loss
function, \\(\eta\\) is the learning rate, and \\(\hat{\Theta}\_t\\) is the
Empirical Neural Tangent Kernel (NTK) ([Jacot et al.,
2018](#ref-jacot_2018_ntk)).

In the infinite-width limit, the NTK converges in distribution to an
analytical limit, \\(\Theta\\), and the NNs evolve as linear models ([Lee et
al., 2019](#ref-lee_2019_wide-nets-linear)). Under gradient flow, the
predictive distribution of this wide network converges to a normal
distribution ([Lee et al., 2019](#ref-lee_2019_wide-nets-linear)),
\\(f^{\text{lin}}\_{\theta\_{\infty}}\left(x\right) \sim \mathcal{N}\left(\mu\_{\text{NN}}\left(x\right),\Sigma\_{\text{NN}}\left(x,x\right)\right)\\),
where

$$\begin{aligned}
  &\mu_{\text{NN}}\left(x\right) = \Theta\left(x,\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \mathbf{Y} \\
  &\begin{split}
    \Sigma_{\text{NN}}\left(x,x'\right) &= \mathcal{K}\left(x,x'\right) + \Theta\left(x,\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \mathcal{K}\left(\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \Theta\left(\mathbf{X},x'\right) \\
    &\quad - \left(\Theta\left(x,\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \mathcal{K}\left(\mathbf{X},x'\right) +
    \mathcal{K}\left(x',\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \Theta\left(\mathbf{X},x\right)\right)
  \end{split}
\end{aligned}$$

where \\(\mathcal{K}\\) denotes the NN-GP kernel ([G. Matthews et al.,
2018](#ref-matthews2018gaussianprocessbehaviourwide)), defined as
\\(\mathcal{K}\left(x,x'\right) = \mathbb{E} \left[f\_{\theta}\left(x\right) \cdot f\_{\theta}\left(x'\right)\right]\\)
which also converges in the infinite-width limit.

The covariance \\(\Sigma\_{\text{NN}}\\) is inconvenient to deal with,
involving two computationally expensive kernel computations, and a
series of cubic-time matrix operations. To tackle this, we can augment
the forward pass (denoted by \\(\tilde{f}\_{\theta}\\)) by adding a random,
untrainable function, which results in the distribution at convergence
having a GP-posterior-like form, with \\(\Theta\\) as the covariance kernel
([He et al., 2020](#ref-bobby_2020_bayesian-ensembles-ntk)),
\\(\tilde{f}\_{\theta\_{\infty}} \sim \mathcal{N}\left(\mu\_{\text{NTK}},\Sigma\_{\text{NTK}}\right)\\),
where \\(\mu\_{\text{NTK}} = \mu\_{\text{NN}}\\) and:

$$\begin{aligned}
    \Sigma_{\text{NTK}}\left(x,x'\right) = \Theta\left(x,x'\right) - \Theta\left(x,\mathbf{X}\right) \Theta\left(\mathbf{X}\right)^{-1} \Theta\left(\mathbf{X},x'\right)
\end{aligned}$$

Importantly, in my Bachelor's thesis project ([Hemachandra et al.,
2023](#ref-hemachandra23a)), we showed that the ratio between
\\(\Sigma\_{\text{NN}}\left(x,x'\right)\\) and
\\(\Sigma\_{\text{NTK}}\left(x,x'\right)\\) can be tightly upper bounded, and
hence, the NTK-GP posterior,
\\(\mathcal{N}\left(\mu\_{\text{NTK-GP}}\left(x\right),\Sigma\_{\text{NTK-GP}}\left(x,x\right)\right)\\),
may be considered a reasonable approximation for the predictive
distribution,
\\(\mathcal{N}\left(\mu\_{\text{NN}}\left(x\right),\Sigma\_{\text{NN}}\left(x,x\right)\right)\\).

## Kernel Canonical Correlation Analysis

We now consider the more practical architectures which have finite
width, so that the feature mapping,
\\(x \mapsto \nabla\_{\theta}f\_{\theta}\left(x\right)\\), associated with the
empirical NTK, \\(\Theta\\), is finite dimensional. Hence, the samples from
the NTK-GP prior are almost-surely contained in the RKHS,
\\(\mathcal{H}\_{\Theta}\\), associated with \\(\Theta\\) (\<span
style=\"color:red\"\>not sure about this part\</span\>). Since the GP
posterior does not have support where the prior does not, the posterior
samples are also contained in this RKHS. Therefore, we can use the RKHS
associated with the NTK to compute the KCC:

$$\begin{aligned}
    S\left(\mathbf{A}\right)
    \approx \max_{f\in\mathcal{H}_{\Theta}, g\in\mathcal{L}} \text{Corr}\left(f\left(\mathbf{X}\right), g\left(\mathbf{Y}\right)\right)
\end{aligned}$$

This value is trivially equal to \\(\pm 1\\) when the kernel matrices
associated with \\(\mathbf{X}\\) and \\(\mathbf{Y}\\) are full-rank ([Gretton,
Herbrich, et al., 2005](#ref-gretton05a)). A common practice is to add
some regularization to this problem by penalizing rougher witness
functions, \\(f\\) and \\(g\\), which yields the following
generalized-eigenvalue problem:

$$\begin{aligned}
    \begin{bmatrix}
        0 & \tilde{\Theta}\tilde{L} \\
        \tilde{\mathbf{L}}\tilde{\Theta} & 0
    \end{bmatrix} \mathbf{u} = 
    \lambda
    \begin{bmatrix}
        \tilde{\Theta}^2 + m\epsilon\tilde{\Theta} & 0 \\
        0 & \tilde{\mathbf{L}}^2 + m\epsilon\tilde{\mathbf{L}}
    \end{bmatrix} \mathbf{u}
\end{aligned}$$

where
\\(\tilde{\Theta} = \mathbf{H}\Theta\left(\mathbf{X}\right)\mathbf{H}\\) and
\\(\tilde{\mathbf{L}} = \mathbf{H}\mathbf{Y}\mathbf{Y}^T\mathbf{H}\\) are
the centered Gram matrices,
\\(\mathbf{H} = \mathbf{I}\_m - 1/m \cdot \mathbf{1}\_{m\times m}\\) is the
centering matrix, \\(m\\) is the number of data points and \\(\epsilon\\) is the
regularization constant. The regularized canonical correlation is the
maximum eigenvalue of this problem, \\(\gamma = \lambda\_{\text{max}}\\).

# Limitations and Extensions

The proposed scoring function is expected to be over-confident for two
reasons:

1.  Using \\(\Theta\left(\mathbf{X}\right)\\) means that we are optimizing
    the correlation over the NTK-GP prior's support, which is larger
    than the posterior's support.

2.  The witness function, \\(f\\), used for computing the canonical
    correlation is the best NN fitting the training set. This amounts to
    ignoring the probabilistic information in the prior/posterior
    altogether. Therefore, it could represent an over-fitting scenario.

3.  Using a linear kernel might make sense for regression, but needs
    justification for classification tasks. What even is an appropriate
    kernel in the classification case? This is an important issue
    because most NAS benchmarks involve image classification tasks, like
    CIFAR-10/100 and ImageNet.

Another limitation is the lack of interpretability of KCC, which is
crucial in many real-world applications. To address this limitation, we
may explore alternate kernel-based measures, such as those based on
Hilbert-Schmidt Independence Criterion (HSIC) ([Gretton, Bousquet, et
al., 2005](#ref-gretton2005hsic)), and the Kernel Target Alignment (KTA)
([Cortes et al., 2012](#ref-cortes12kta)), as proposed in ([Chang et
al., 2013](#ref-chang13hsic)).


# References

<div id="refs" class="references csl-bib-body hanging-indent"
data-entry-spacing="0" data-line-spacing="2" role="list">
<div id="ref-akaho2007kcca" class="csl-entry" role="listitem">
[1] Akaho, S. (2007). A kernel method for canonical correlation
analysis. Retrieved from <a
href="https://arxiv.org/abs/cs/0609071">https://arxiv.org/abs/cs/0609071</a>
</div>
<div id="ref-chang13hsic" class="csl-entry" role="listitem">
[2] Chang, B., Kruger, U., Kustra, R., &amp; Zhang, J. (2013). Canonical
correlation analysis based on hilbert-schmidt independence criterion and
centered kernel target alignment. In <em>Proceedings of the 30th
international conference on machine learning</em> (Vol. 28, pp.
316–324). Atlanta, Georgia, USA: PMLR. Retrieved from <a
href="https://proceedings.mlr.press/v28/chang13.html">https://proceedings.mlr.press/v28/chang13.html</a>
</div>
<div id="ref-chen2020tenas" class="csl-entry" role="listitem">
[3] Chen, W., Gong, X., &amp; Wang, Z. (2021). Neural architecture
search on ImageNet in four GPU hours: A theoretically inspired
perspective. In <em>International conference on learning
representations</em>.
</div>
<div id="ref-cortes12kta" class="csl-entry" role="listitem">
[4] Cortes, C., Mohri, M., &amp; Rostamizadeh, A. (2012). Algorithms for
learning kernels based on centered alignment. <em>Journal of Machine
Learning Research</em>, <em>13</em>(28), 795–828. Retrieved from <a
href="http://jmlr.org/papers/v13/cortes12a.html">http://jmlr.org/papers/v13/cortes12a.html</a>
</div>
<div id="ref-dong2020nasbench201" class="csl-entry" role="listitem">
[5] Dong, X., &amp; Yang, Y. (2020). NAS-bench-201: Extending the scope
of reproducible neural architecture search. In <em>International
conference on learning representations (ICLR)</em>. Retrieved from <a
href="https://openreview.net/forum?id=HJxyZkBKDr">https://openreview.net/forum?id=HJxyZkBKDr</a>
</div>
<div id="ref-englisch_1994_corr-mse" class="csl-entry" role="listitem">
[6] Englisch, H., &amp; Hiemstra, Y. (1994). The correlation as cost
function in neural networks. In <em>Proceedings of 1994 IEEE
international conference on neural networks (ICNN’94)</em> (Vol. 5, pp.
3170–3172 vol.5). doi:<a
href="https://doi.org/10.1109/ICNN.1994.374741">10.1109/ICNN.1994.374741</a>
</div>
<div id="ref-matthews2018gaussianprocessbehaviourwide" class="csl-entry"
role="listitem">
[7] G. Matthews, A. G. de, Rowland, M., Hron, J., Turner, R. E., &amp;
Ghahramani, Z. (2018). Gaussian process behaviour in wide deep neural
networks. Retrieved from <a
href="https://arxiv.org/abs/1804.11271">https://arxiv.org/abs/1804.11271</a>
</div>
<div id="ref-gretton2005hsic" class="csl-entry" role="listitem">
[8] Gretton, A., Bousquet, O., Smola, A., &amp; Schölkopf, B. (2005).
Measuring statistical dependence with hilbert-schmidt norms. In S. Jain,
H. U. Simon, &amp; E. Tomita (Eds.), <em>Algorithmic learning
theory</em> (pp. 63–77). Berlin, Heidelberg: Springer Berlin Heidelberg.
</div>
<div id="ref-gretton05a" class="csl-entry" role="listitem">
[9] Gretton, A., Herbrich, R., Smola, A., Bousquet, O., &amp; Schölkopf,
B. (2005). Kernel methods for measuring independence. <em>Journal of
Machine Learning Research</em>, <em>6</em>(70), 2075–2129. Retrieved
from <a
href="http://jmlr.org/papers/v6/gretton05a.html">http://jmlr.org/papers/v6/gretton05a.html</a>
</div>
<div id="ref-bobby_2020_bayesian-ensembles-ntk" class="csl-entry"
role="listitem">
[10] He, B., Lakshminarayanan, B., &amp; Teh, Y. W. (2020). Bayesian
deep ensembles via the neural tangent kernel. In <em>Advances in neural
information processing systems</em> (Vol. 33, pp. 1010–1022). Curran
Associates, Inc. Retrieved from <a
href="https://proceedings.neurips.cc/paper_files/paper/2020/file/0b1ec366924b26fc98fa7b71a9c249cf-Paper.pdf">https://proceedings.neurips.cc/paper_files/paper/2020/file/0b1ec366924b26fc98fa7b71a9c249cf-Paper.pdf</a>
</div>
<div id="ref-hemachandra23a" class="csl-entry" role="listitem">
[11] Hemachandra, A., Dai, Z., Singh, J., Ng, S.-K., &amp; Low, B. K. H.
(2023). Training-free neural active learning with
initialization-robustness guarantees. In <em>Proceedings of the 40th
international conference on machine learning</em> (Vol. 202, pp.
12931–12971). PMLR. Retrieved from <a
href="https://proceedings.mlr.press/v202/hemachandra23a.html">https://proceedings.mlr.press/v202/hemachandra23a.html</a>
</div>
<div id="ref-jacot_2018_ntk" class="csl-entry" role="listitem">
[12] Jacot, A., Gabriel, F., &amp; Hongler, C. (2018). Neural tangent
kernel: Convergence and generalization in neural networks. In
<em>Advances in neural information processing systems</em> (Vol. 31).
Curran Associates, Inc. Retrieved from <a
href="https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf">https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf</a>
</div>
<div id="ref-lee_2019_wide-nets-linear" class="csl-entry"
role="listitem">
[13] Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R.,
Sohl-Dickstein, J., &amp; Pennington, J. (2019). Wide neural networks of
any depth evolve as linear models under gradient descent. In
<em>Advances in neural information processing systems</em> (Vol. 32).
Curran Associates, Inc. Retrieved from <a
href="https://proceedings.neurips.cc/paper_files/paper/2019/file/0d1a9651497a38d8b1c3871c84528bd4-Paper.pdf">https://proceedings.neurips.cc/paper_files/paper/2019/file/0d1a9651497a38d8b1c3871c84528bd4-Paper.pdf</a>
</div>
<div id="ref-liu2018darts" class="csl-entry" role="listitem">
[14] Liu, H., Simonyan, K., &amp; Yang, Y. (2019). <span>DARTS</span>:
Differentiable architecture search. In <em>International conference on
learning representations</em>. Retrieved from <a
href="https://openreview.net/forum?id=S1eYHoC5FX">https://openreview.net/forum?id=S1eYHoC5FX</a>
</div>
<div id="ref-thomas2001kcca" class="csl-entry" role="listitem">
[15] Melzer, T., Reiter, M., &amp; Bischof, H. (2001). Nonlinear feature
extraction using generalized canonical correlation analysis. In G.
Dorffner, H. Bischof, &amp; K. Hornik (Eds.), <em>Artificial neural
networks — ICANN 2001</em> (pp. 353–360). Berlin, Heidelberg: Springer
Berlin Heidelberg.
</div>
<div id="ref-poyser_2024_nas-review" class="csl-entry" role="listitem">
[16] Poyser, M., &amp; Breckon, T. P. (2024). Neural architecture
search: A contemporary literature review for computer vision
applications. <em>Pattern Recognition</em>, <em>147</em>, 110052. doi:<a
href="https://doi.org/10.1016/j.patcog.2023.110052">https://doi.org/10.1016/j.patcog.2023.110052</a>
</div>
<div id="ref-pmlr-v139-xu21m" class="csl-entry" role="listitem">
[17] Xu, J., Zhao, L., Lin, J., Gao, R., Sun, X., &amp; Yang, H. (2021).
KNAS: Green neural architecture search. In M. Meila &amp; T. Zhang
(Eds.), <em>Proceedings of the 38th international conference on machine
learning</em> (Vol. 139, pp. 11613–11625). PMLR. Retrieved from <a
href="https://proceedings.mlr.press/v139/xu21m.html">https://proceedings.mlr.press/v139/xu21m.html</a>
</div>
<div id="ref-ying19nasbench101" class="csl-entry" role="listitem">
[18] Ying, C., Klein, A., Christiansen, E., Real, E., Murphy, K., &amp;
Hutter, F. (2019). <span>NAS</span>-bench-101: Towards reproducible
neural architecture search. In <em>Proceedings of the 36th international
conference on machine learning</em> (Vol. 97, pp. 7105–7114). Long
Beach, California, USA: PMLR. Retrieved from <a
href="http://proceedings.mlr.press/v97/ying19a.html">http://proceedings.mlr.press/v97/ying19a.html</a>
</div>
</div>