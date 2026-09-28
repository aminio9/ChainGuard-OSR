# Mathematical Methodology to Code Mapping

This document provides a line-by-line, formal cross-reference between the theoretical formulation presented in `02_Detailed_Methodology_and_Defense_Document.md` and the implemented Python codebase in `src/`.

---

## 1. Master Formula Mapping

| Mathematical Object | Mathematical Equation | Implementation File | Python Function / Class |
| :--- | :--- | :--- | :--- |
| **Formula 1: Dirichlet Label Skew** | $\mathbf{p}_k \sim \operatorname{Dir}_C(\beta_D \mathbf{1}_C)$ | `src/data/partitioning.py` | `partition_dirichlet_non_iid()` |
| **Formula 2: Openness Metric** | $\mathcal{O} = 1 - \sqrt{\frac{2 C_{\mathrm{known}}}{2 C_{\mathrm{known}} + U}}$ | `src/data/osr_split.py` | `compute_openness()` |
| **Formula 3: Decoupled Branches** | $z = E_\theta(x), \hat{y} = C_\phi(z), h = H_\psi(z)$ | `src/models/dual_branch.py` | `DecoupledDualBranchNetwork` |
| **Formula 4: Class-Balanced Loss** | $\mathcal{L}_{\mathrm{CB}} = - \frac{1 - \beta_{\mathrm{cb}}}{1 - \beta_{\mathrm{cb}}^{N_y}} (1 - p_y)^\gamma \log p_y$ | `src/losses/class_balanced.py` | `class_balanced_focal_loss()` |
| **Formula 5 & 6: VIB KL Divergence** | $\mathcal{L}_{\mathrm{VIB}} = \frac{1}{2} \sum (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$ | `src/losses/vib_loss.py` | `vib_kl_divergence_loss()` |
| **Formula 7: Orthogonal Anchoring** | $\mathcal{L}_{\mathrm{anchor}} = \frac{1}{C(C-1)} \sum_{i \neq j} \|\cos(\mathbf{w}_i, \mathbf{w}_j)\| + \lambda_r \max(0, \|\mathbf{w}_c\| - r_{\max})$ | `src/losses/orthogonal_loss.py` | `orthogonal_anchor_loss()` |
| **Formula 8: Primal-Dual Penalty** | $\mathcal{L}_{\mathrm{PD}} = \langle \lambda_k, \theta_k - \theta_{\mathrm{global}} \rangle + \frac{\rho}{2} \|\theta_k - \theta_{\mathrm{global}}\|^2$ | `src/losses/primal_dual_loss.py` | `primal_dual_consensus_loss()` |
| **Formula 9: Composite Local Loss** | $\mathcal{L}_{\mathrm{local}} = \mathcal{L}_{\mathrm{CB}} + \lambda_{\mathrm{vib}} \mathcal{L}_{\mathrm{VIB}} + \lambda_{\mathrm{anchor}} \mathcal{L}_{\mathrm{anchor}} + \mathcal{L}_{\mathrm{PD}}$ | `src/losses/composite_loss.py` | `CompositeLossCalculator` |
| **Formula 10: Bayesian Covariance Prototype** | $\bar{\mu}_{\mathrm{global}, c} = (\sum \Sigma_{k,c}^{-1})^{-1} \sum \Sigma_{k,c}^{-1} \bar{\mu}_{k,c}$ | `src/aggregation/covariance_proto.py` | `aggregate_prototypes_covariance()` |
| **Formula 11: Weibull EVT Discounting** | $\omega_c(x) = 1 - \exp(- ((\max(0, d_c - \tau_c)) / \kappa_c)^{\xi_c})$ | `src/osr/weibull_evt.py` | `compute_evt_probabilities()` |
| **Formula 12: Free Helmholtz Energy** | $E(x; \mathbf{w}) = - T \log \sum \exp(f_c(x) / T)$ | `src/osr/energy_ood.py` | `compute_free_energy_score()` |
| **Formulas 13-15: Mondrian Conformal** | $s_i = 1 - \hat{p}_{y_i}(x_i), \hat{q}_c, \mathcal{C}_{1-\alpha}(x)$ | `src/osr/conformal.py` | `MondrianConformalCalibrator` |

---

## 2. Algorithms Mapping

### Algorithm 1: Federated Training & Consensus
* **Lines 1-5 (Initialization & Client Sampling):** `src/federated/server.py` (`FederatedServer.__init__` and `select_active_clients`)
* **Lines 6-11 (Local Epochs, Stochastic VIB pass, Prototype Extraction):** `src/federated/client.py` (`FederatedClient.train_local`)
* **Line 12 (Parameter FedAvg Aggregation):** `src/aggregation/fedavg_agg.py` (`aggregate_fedavg_parameters`)
* **Line 13 (Dual Multiplier Updates):** `src/aggregation/primal_dual_agg.py` (`update_dual_multipliers`)
* **Line 14 (Bayesian Covariance Prototype Synthesis):** `src/aggregation/covariance_proto.py` (`aggregate_prototypes_covariance`)

### Algorithm 2: Real-Time Edge Inference
* **Lines 1-3 (Flow Normalization & Latent Forward Pass $z = \mu(x)$):** `src/osr/edge_inference.py` (`EdgeInferenceEngine.predict_flow`)
* **Line 4 (Logits & Free Energy Score):** `src/osr/energy_ood.py` (`compute_free_energy_score`)
* **Line 5 (EVT Weibull Distance Discounting):** `src/osr/weibull_evt.py` (`compute_evt_probabilities`)
* **Line 6 (Mondrian Conformal Prediction Set Construction):** `src/osr/conformal.py` (`MondrianConformalCalibrator.predict_conformal_sets`)
* **Lines 7-8 (Alert Trigger on $\mathcal{C} = \emptyset$ vs Argmax Output):** `src/osr/edge_inference.py` (decision branching)

---

## 3. Theoretical Guarantees Mapping

* **Theorem 1 (Asymptotic Convergence Bound):** Bounded dissimilarity and quadratic penalty $\rho$ in `src/losses/primal_dual_loss.py`.
* **Theorem 2 (Finite-Sample False Alarm Rate Bound):** $\mathbb{P}(\mathcal{C}_{1-\alpha}(X) = \emptyset \mid \text{Known}) \le \alpha$, empirically calibrated in `src/osr/conformal.py` and validated in `src/metrics/conformal_metrics.py`.
