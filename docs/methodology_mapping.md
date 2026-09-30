# Mathematical Methodology to Code Mapping (Master Reference)

This document provides a line-by-line, formal cross-reference between the theoretical formulation presented in `02_Detailed_Methodology_and_Defense_Document.md` and the implemented Python codebase in `src/`.

---

## 1. Master Formula Mapping (All 27 Equations)

| Equation | Mathematical Object & Formal Formulation | Implementation File | Python Function / Class |
| :--- | :--- | :--- | :--- |
| **Eq. 1 & 2** | **Class-Balanced Loss ($\mathcal{L}_{\mathrm{CB}}$):**<br>$\widetilde{w}_c^{(k)} = \frac{(1 - \beta_{\mathrm{cb}}) / (1 - \beta_{\mathrm{cb}}^{n_c^{(k)}})}{\sum_{j \in \mathcal{Y}_k^{\mathrm{local}}} (1 - \beta_{\mathrm{cb}}) / (1 - \beta_{\mathrm{cb}}^{n_j^{(k)}})}$<br>$\mathcal{L}_{\mathrm{CB}} = - \frac{1}{|\mathcal{B}|} \sum_{(x_i, y_i) \in \mathcal{B}} \widetilde{w}_{y_i}^{(k)} \log p_{k, y_i}(x_i)$ | `src/losses/class_balanced.py` | `class_balanced_focal_loss()` |
| **Eq. 3 & 4** | **VIB Objective ($\mathcal{L}_{\mathrm{VIB}}$):**<br>$q_\theta(z \mid x) = \mathcal{N}(z; \mu(x), \operatorname{diag}(\sigma^2(x)))$<br>$\mathcal{L}_{\mathrm{VIB}} = \frac{1}{2} \sum_{j=1}^{d_z} (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$ | `src/losses/vib_loss.py` | `vib_kl_divergence_loss()` |
| **Eq. 5** | **Shannon Entropy Certainty Gate:**<br>$U_{\mathrm{anc}}(x) = 1 - \frac{H(p_{\mathrm{anc}}^{(\tau_g)}(x))}{\log |C_K|}$ | `src/losses/anchor_loss.py` | `compute_shannon_certainty()` |
| **Eq. 6** | **Sigmoid Dynamic Time-Schedule:**<br>$\lambda_{\mathrm{anc}}(x, t) = \lambda_{\mathrm{anc}}^{\max} \cdot \frac{1}{1 + \exp(-\kappa(t/T - t_0))} \cdot U_{\mathrm{anc}}(x)$ | `src/losses/anchor_loss.py` | `compute_anchor_weight()` |
| **Eq. 7 & 8** | **Dynamic Global Anchor Regularization:**<br>$\mathcal{L}_{\mathrm{anchor}} = \tau_g^2 \cdot D_{\mathrm{KL}}(p_{\mathrm{anc}}^{(\tau_g)}(x) \parallel p_k^{(\tau_g)}(x))$<br>$\frac{\partial [\lambda_{\mathrm{anc}} \mathcal{L}_{\mathrm{anchor}}]}{\partial \ell_{k, c}(x_i)} = \lambda_{\mathrm{anc}} \tau_g (p_{k,c}^{(\tau_g)} - p_{\mathrm{anc},c}^{(\tau_g)}) \neq 0$ | `src/losses/anchor_loss.py` | `DynamicGlobalAnchorLoss` |
| **Eq. 9** | **Composite Local Objective:**<br>$\min_{w_k} \mathcal{L}_{\mathrm{local}} = \mathcal{L}_{\mathrm{CB}} + \lambda_{\mathrm{vib}} \mathcal{L}_{\mathrm{VIB}} + \lambda_{\mathrm{anc}}(x, t) \mathcal{L}_{\mathrm{anchor}}$ | `src/losses/composite_loss.py` | `CompositeLossCalculator` |
| **Eq. 10** | **Cosine Annealing Learning Rate:**<br>$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$ | `src/federated/server.py` | `FederatedServer.compute_cosine_learning_rate()` |
| **Eq. 11** | **FedAvg Global Consensus:**<br>$w^{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}$ | `src/aggregation/fedavg_agg.py` | `aggregate_fedavg_parameters()` |
| **Eq. 12 & 13** | **Phase II Sub-Block Local Moments:**<br>$\boldsymbol{\mu}_c^{(k)} = \frac{1}{n_c^{(k)}} \sum z_{k, i}$<br>$\boldsymbol{\Sigma}_{c, \mathcal{G}_b}^{(k)} = \frac{1}{n_c^{(k)}} \sum (z_{k,i}[\mathcal{G}_b] - \boldsymbol{\mu}_{c, \mathcal{G}_b}^{(k)})(z_{k,i}[\mathcal{G}_b] - \boldsymbol{\mu}_{c, \mathcal{G}_b}^{(k)})^\top$ | `src/federated/client.py` | `FederatedClient.extract_phase_two_moments()` |
| **Eq. 14 & 15** | **Law of Total Variance Pooled Covariance:**<br>$\boldsymbol{\mu}_c = \frac{1}{N_c} \sum_{k \in \mathcal{K}_c} n_c^{(k)} \boldsymbol{\mu}_c^{(k)}$<br>$\boldsymbol{\Sigma}_{c, \mathcal{G}_b} = \frac{1}{N_c} \sum_{k \in \mathcal{K}_c} n_c^{(k)} [\boldsymbol{\Sigma}_{c, \mathcal{G}_b}^{(k)} + (\boldsymbol{\mu}_c^{(k)}[\mathcal{G}_b] - \boldsymbol{\mu}_c[\mathcal{G}_b])(\boldsymbol{\mu}_c^{(k)}[\mathcal{G}_b] - \boldsymbol{\mu}_c[\mathcal{G}_b])^\top] + \epsilon_v \mathbf{I}_{d_b}$ | `src/aggregation/total_variance.py` | `aggregate_total_variance_moments()` |
| **Eq. 16** | **Wire-Speed Block-Diagonal Mahalanobis Distance:**<br>$D_{\mathrm{block}}(z, c) = \sum_{b=1}^B (z[\mathcal{G}_b] - \boldsymbol{\mu}_c[\mathcal{G}_b])^\top (\boldsymbol{\Sigma}_{c, \mathcal{G}_b})^{-1} (z[\mathcal{G}_b] - \boldsymbol{\mu}_c[\mathcal{G}_b])$ | `src/osr/block_mahalanobis.py` | `compute_block_mahalanobis_distances()` |
| **Eq. 17 & 18** | **Candidate Class & Nonconformity Score:**<br>$\widehat{c}(x_{\mathrm{test}}) = \arg\min_{c \in C_K} D_{\mathrm{block}}(z, c)$<br>$S(x_{\mathrm{test}}) = D_{\mathrm{block}}(z, \widehat{c}(x_{\mathrm{test}}))$ | `src/osr/block_mahalanobis.py` | `WireSpeedEdgeDetector.predict_sample()` |
| **Eq. 19 & 20** | **Distributed Mondrian Conformal Quantiles:**<br>$q_k^{(1-\alpha)}(c) = \operatorname{Quantile}(\mathcal{S}_{\mathrm{cal}, k}^c, \frac{\lceil(n+1)(1-\alpha)\rceil}{n})$<br>$\tau_{\alpha, c} = \sum_{k \in \mathcal{K}_c} \frac{n_{\mathrm{cal}, c}^{(k)}}{N_{\mathrm{cal}, c}} q_k^{(1 - \alpha)}(c)$ | `src/osr/conformal.py` | `compute_finite_sample_quantile()`, `aggregate_distributed_quantiles()` |
| **Eq. 21 & 22** | **Finite-Sample Coverage Guarantee (Theorem 2):**<br>$\mathbb{P}(S(X_{\mathrm{test}}) > \tau_{\alpha, c} \mid Y_{\mathrm{test}} = c) \le \alpha$ | `src/metrics/conformal_metrics.py` | `compute_conformal_metrics()` |
| **Eq. 23** | **EVT Zero-Day Power Bound (Theorem 3):**<br>$\mathbb{P}(f_{\mathrm{OSR}}(X) \neq \mathrm{Unknown} \mid X \sim u) \le \exp(- ((\Delta_{\min} - \tau_{\alpha, \widehat{c}}) / \sigma)^{\xi})$ | `src/osr/weibull_evt.py` | `WeibullEVTFitter` |
| **Eq. 24** | **Operational Decision Rule (Algorithm 2 Line 7-10):**<br>$f_{\mathrm{OSR}}(x) = \begin{cases} \mathrm{Unknown} & \text{if } S(x) > \tau_{\alpha, \widehat{c}} \\ \widehat{c} & \text{otherwise} \end{cases}$ | `src/osr/block_mahalanobis.py` | `WireSpeedEdgeDetector` |

---

## 2. Formal Theoretical Theorems Mapping

1. **Theorem 1 (Algebraic Exactness and Zero Feature Leakage of Federated Moment Aggregation):**
   - *Statement:* Without raw feature exchange ($I(D_k; \mathcal{M}_k) = 0$ for $n_c^{(k)} \ge 3$), the pooled covariance matches centralized sample covariance under arbitrary label skew.
   - *Implemented in:* `src/aggregation/total_variance.py`.
   - *Tested in:* `tests/test_total_variance.py` (`test_law_of_total_variance_exactness` passes with machine precision atol=$10^{-5}$).

2. **Theorem 2 (Class-Conditional Mondrian Finite-Sample Coverage Guarantee):**
   - *Statement:* False alarm rate on known traffic satisfies $\mathbb{P}(f_{\mathrm{OSR}}(X) = \mathrm{Unknown} \mid Y = c) \le \alpha$ for all $c \in C_K$.
   - *Implemented in:* `src/osr/conformal.py`.
   - *Tested in:* `tests/test_osr_conformal.py`.

3. **Theorem 3 (Neyman-Pearson Zero-Day Power Bound via EVT):**
   - *Statement:* Escape probability of unseen attacks decays exponentially as the semantic distance to known clusters increases.
   - *Implemented in:* `src/osr/weibull_evt.py`.
