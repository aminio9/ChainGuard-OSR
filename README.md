# ChainGuard-OSR

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)
[![Colab: Ready](https://img.shields.io/badge/Colab-Ready-orange.svg)](ChainGuard_OSR_Colab_Runner.ipynb)

> **Federated Open-Set Recognition for Blockchain Network Traffic under Non-IID Distributions**

ChainGuard-OSR is a federated intrusion detection framework designed to detect zero-day attacks and novel threats in decentralized peer-to-peer (P2P) blockchain networks subject to extreme label distribution skew ($\beta_D \le 0.1$).

---

## Methodology

ChainGuard-OSR operates across four coordinated stages:

```
[Nodes: Local Flows]
       │
       ▼
1. Representation Learning ──► VIB (Gaussian Prior) + Balanced Logit Adjustment (BLA)
       │
       ▼
2. Federated Class Profiling ──► Law of Total Variance + OAS Regularized Block-Mahalanobis
       │
       ▼
3. Conformal Calibration   ──► Distributed Mondrian Quantiles (Guaranteed False Alarm Bound)
       │
       ▼
4. Three-Way Decision Rule ──► POS (Known Traffic) | BND (Abstain) | NEG (Zero-Day Routed to C+1)
```

1. **Representation Learning (VIB + BLA)**:
   * **Variational Information Bottleneck ($\mathcal{L}_{\mathrm{VIB}}$)** compresses flow features toward a standard Gaussian prior ($\mathcal{N}(\mathbf{0}, \mathbf{I})$), discarding extraneous packet noise while preserving Euclidean distance metrics.
   * **Balanced Logit Adjustment ($\mathcal{L}_{\mathrm{BLA}}$)** dynamically scales decision margins using smoothed local Laplace priors, preventing gradient collapse caused by classes absent on non-IID nodes.

2. **Federated Class Profiling (Law of Total Variance)**:
   * Reconstructs global class centroids and covariance matrices without exchanging raw traffic data by deconstructing total variance into within-client and between-client dispersion:
     $$\mathbf{\Sigma}_c = \sum_{k} w_{k,c} \mathbf{\Sigma}_{k,c} + \sum_{k} w_{k,c} (\boldsymbol{\mu}_{k,c} - \boldsymbol{\mu}_c)(\boldsymbol{\mu}_{k,c} - \boldsymbol{\mu}_c)^\top$$
   * Stabilizes singular covariances using **Oracle Approximating Shrinkage (OAS)** and block-diagonal factorization.

3. **Federated Mondrian Conformal Calibration**:
   * Nodes upload local nonconformity score histograms to the coordinator server, which computes class-specific empirical quantiles $\tau_{\alpha, c}$.
   * Mathematically guarantees that known-class false rejection rates (KFR) are strictly bounded by the significance level $\alpha$ with finite-sample coverage validity.

4. **Line-Rate Three-Way Decision Rule**:
   * Evaluates incoming network flows locally into three decision zones:
     * **Accept (POS)**: High-confidence known traffic category ($D(\mathbf{z}, c) \le \tau_{\mathrm{accept}}$).
     * **Abstain (BND)**: Ambiguous boundary flows routed for secondary inspection.
     * **Reject (NEG)**: Confirmed zero-day anomaly routed to novel class $C+1$ ($D(\mathbf{z}, c) > \tau_{\mathrm{reject}}$).

---

## Quick Start

### Installation

```bash
git clone https://github.com/aminio9/ChainGuard-OSR.git
cd ChainGuard-OSR
pip install -e .
```

### Run

```bash
# 1. Generate synthetic P2P blockchain traffic (10k flows)
python scripts/generate_synthetic_data.py

# 2. Run federated training and open-set evaluation
python experiments/run_federated.py --rounds 20 --config configs/default.yaml

# 3. Run automated test suite (58 tests)
pytest tests/ -v
```

An interactive, GPU-accelerated notebook is available at [ChainGuard_OSR_Colab_Runner.ipynb](ChainGuard_OSR_Colab_Runner.ipynb).

---

## Citation

```bibtex
@article{amini2026chainguard,
  title={Federated Open-Set Recognition for Blockchain Network Traffic under Non-IID Distributions},
  author={Amini, Mohammad Hassan and Entezari-Maleki, Reza},
  journal={Computer Networks},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
