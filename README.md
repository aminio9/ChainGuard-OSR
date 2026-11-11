# Server-Coordinated Federated Open-Set Intrusion Detection for Blockchain Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)

> **Federated Open-Set Recognition for Blockchain Network Traffic under Non-IID Distributions**

An open-set intrusion detection framework designed to detect zero-day attacks and novel threats in decentralized peer-to-peer (P2P) blockchain networks subject to extreme label distribution skew ($\beta_D \le 0.1$).

---

## Methodology

The architecture operates across three coordinated phases:

```
[Nodes: Local Traffic Flows]
       │
       ▼
Phase 1: Representation Learning ──► VIB + Balanced Logit Adjustment (BLA) + Decoupled Aggregation
       │
       ▼
Phase 2: Moment Profiling        ──► One-Shot K-Means + Law of Total Variance + OAS Shrinkage
       │
       ▼
Phase 3: Conformal Calibration   ──► Distributed Mondrian Quantiles + Wire-Speed Edge Inference
```

1. **Representation Learning (VIB + BLA)**:
   * **Variational Information Bottleneck ($\mathcal{L}_{\mathrm{VIB}}$)**: Compresses flow features toward a standard Gaussian prior, discarding packet noise while maintaining geometric separation.
   * **Balanced Logit Adjustment ($\mathcal{L}_{\mathrm{BLA}}$)**: Dynamically scales decision margins based on local class frequencies and strictly masks gradients for absent classes, preventing gradient distortion under severe non-IID skew.
   * **Decoupled Parameter Aggregation**: Aggregates the feature encoder across nodes while maintaining local classifier heads to handle client-specific conditional label skews.

2. **Federated Class Profiling (Law of Total Variance)**:
   * Reconstructs global class centroids and covariance matrices without exchanging raw traffic data by decomposing total variance into within-client and between-client dispersion.
   * Normal traffic is modeled using 4 behavioral reference modes (session handshake, ledger synchronization, transaction propagation, and block announcement), while attack categories are modeled with single centroids.
   * Stabilizes covariance matrices using **Oracle Approximating Shrinkage (OAS)**.

3. **Distributed Mondrian Conformal Calibration & Open-Set Edge Inference**:
   * Nodes upload local nonconformity score histograms to the coordinator server, which computes class-specific empirical quantiles $\hat{q}_{\alpha, c}$.
   * Mathematically guarantees that known-class false rejection rates (KFR) are bounded by the significance level $\alpha$.
   * **Edge Decision Rule**: Constructs conformal prediction set $C(z) = \{c \mid D_c(z) \le \hat{q}_{\alpha, c}\}$. If $C(z) = \emptyset$, the sample is rejected as an unknown / zero-day intrusion ($C+1$). Otherwise, it is classified as $\arg\min_{c \in C(z)} \frac{D_c(z)}{\hat{q}_{\alpha, c}}$.

---

## Quick Start

### Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Run Experiments

```bash
# 1. Run all tests
pytest tests/ -v

# 2. Run federated training and open-set evaluation
python experiments/run_federated.py --rounds 30 --config configs/default.yaml

# 3. Run master benchmark suite
python run_experiments.py
```

## License

This project is licensed under the [MIT License](LICENSE).
