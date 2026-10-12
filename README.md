# FedTROS: Federated Open-Set Recognition for Blockchain Network Traffic under Non-IID Distributions

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)
[![Colab: Ready](https://img.shields.io/badge/Colab-Ready-orange.svg)](FedTROS_Colab_Runner.ipynb)

A modular, reproducible, and mathematically rigorous framework for **Federated Open-Set Recognition (OSR)** in decentralized peer-to-peer (P2P) blockchain networks under extreme label heterogeneity (Dirichlet non-IID skew).

---

## 1. Overview

In decentralized blockchain architectures, validating nodes are subject to both known transactional patterns and novel zero-day attacks. Traditional federated intrusion detection systems face two fundamental failure modes under non-IID distributions:
1. **Gradient Distortion & Client Drift**: Local optimization without absent-class supervision pushes client representations into disjoint subspaces, leading to catastrophic dimensional collapse.
2. **Uncalibrated Open-Set Rejection**: Heuristic thresholding on softmax scores or unregularized distance metrics produces uncontrollable false alarm rates (FAR) on legitimate traffic.

**FedTROS** addresses these challenges via a 4-stage coordinated framework:
* **Variational Information Bottleneck (VIB)**: Enforces latent feature compression toward a standard Gaussian prior ($\mathcal{N}(\mathbf{0}, \mathbf{I})$), filtering high-frequency flow noise while preserving unconstrained Euclidean geometry.
* **Balanced Logit Adjustment (BLA)**: Compensates for local class absence using smoothed Laplace priors, expanding decision margins for underrepresented categories.
* **Federated Class Profiling via Law of Total Variance**: Deconstructs global class covariance into mean-within-client and between-client dispersion, regularized via Oracle Approximating Shrinkage (OAS) and block-diagonal factorization.
* **Federated Mondrian Conformal Calibration**: Computes class-specific rejection thresholds $\tau_{\alpha, c}$ over distributed nonconformity histograms, guaranteeing bounded false rejection rates ($\mathrm{KFR} \le \alpha$) with finite-sample coverage validity.
* **Decision-Theoretic Three-Way Inference**: Partitions incoming traffic flows into Positive (verified known), Boundary (abstained for secondary inspection), and Negative (confirmed zero-day attack routed to class $C+1$).

---

## 2. Repository Structure

```
├── .gitignore                   # Clean ignore rules for data, checkpoints, and logs
├── LICENSE                      # MIT Open Source License
├── pyproject.toml               # Modern packaging specification (Hatchling / UV)
├── README.md                    # Framework documentation and replication guide
├── FedTROS_Colab_Runner.ipynb   # Interactive end-to-end Google Colab runner
├── configs/                     # Declarative YAML experiment configurations
│   ├── default.yaml             # Master benchmark configuration
│   ├── datasets/                # Dataset definitions (B-TAT, B-NAT, Synthetic)
│   ├── models/                  # VIB stochastic encoder & baseline architectures
│   ├── federated/               # Federated aggregation & client schedules
│   ├── osr/                     # Conformal calibration & Mahalanobis settings
│   ├── baselines/               # Baseline comparison configurations
│   └── ablations/               # Systematic ablation study matrices
├── src/                         # Core framework package
│   ├── aggregation/             # Law of Total Variance & prototype aggregators
│   ├── baselines/               # FedProto, Mahalanobis, MSP, Energy, OpenMax
│   ├── config.py                # Pydantic schema validation & YAML loader
│   ├── data/                    # Synthetic generator, preprocessors & Dirichlet partitioner
│   ├── federated/               # Decentralized Client and Server protocol coordinators
│   ├── losses/                  # VIB loss, Balanced Logit-Adjusted loss, Composite loss
│   ├── metrics/                 # AUROC, OS-Macro-F1, FPR@95, Conformal coverage
│   ├── models/                  # Decoupled VIB stochastic encoders and classifiers
│   ├── osr/                     # Block-Mahalanobis, Weibull EVT, Mondrian Conformal
│   ├── privacy/                 # Differential Privacy moments accountant
│   ├── tracking/                # Structured JSONL experiment logger & telemetry
│   ├── utils/                   # Device handlers, checkpointing, and numerical I/O
│   └── visualization/           # High-DPI publication plots and vector export
├── experiments/                 # Reproducible experiment runners
│   ├── run_federated.py         # Main federated training & open-set evaluation
│   ├── run_baselines.py         # Unified comparative baseline benchmarks
│   ├── run_ablations.py         # Architectural ablation suite
│   ├── run_real_benchmarks.py   # Real-world benchmark evaluation runner
│   ├── run_conformal_sweep.py   # Conformal alpha significance sensitivity sweep
│   ├── run_loao.py              # Leave-One-Attack-Out (LOAO) cross-validation
│   ├── run_scalability.py       # Node scaling and Dirichlet sensitivity analysis
│   ├── run_wire_speed_profiling.py # Line-rate inference latency & profiling
│   └── evaluate_checkpoint.py   # Checkpoint inspection and test-time evaluation
├── scripts/                     # Automation pipelines
│   ├── generate_synthetic_data.py # Fast generator for 10k synthetic P2P benchmark flows
│   ├── run_all_experiments.sh   # Bash batch replication script
│   └── run_all_experiments.ps1  # PowerShell batch replication script
└── tests/                       # Complete pytest unit and integration test suite
    ├── test_aggregation.py
    ├── test_anchor_loss.py
    ├── test_baselines.py
    ├── test_block_mahalanobis.py
    ├── test_data.py
    ├── test_federated_flow.py
    ├── test_federated_multi_prototype_integration.py
    ├── test_losses.py
    ├── test_models.py
    ├── test_multi_prototype_osr.py
    ├── test_osr_conformal.py
    ├── test_osr_metrics.py
    ├── test_total_variance.py
    └── test_zero_data_leakage.py
```

---

## 3. Quick Start

### Installation

Clone the repository and install dependencies using `pip` or `uv`:

```bash
# Using standard pip in a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e .

# Or using Astral UV (recommended for high speed):
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### 1-Minute Smoke Test

To verify the installation and run end-to-end training on synthetic blockchain flow data:

```bash
# 1. Generate synthetic P2P blockchain traffic:
python scripts/generate_synthetic_data.py

# 2. Run a 5-round federated training and open-set evaluation:
python experiments/run_federated.py --rounds 5 --config configs/default.yaml
```

---

## 4. Reproducing Paper Experiments

All experimental configurations and seeds are fully deterministic and managed via declarative YAML configs.

### A. Main Proposed Method
Trains the federated representation encoder, computes server-coordinated class profiles via the Law of Total Variance, and evaluates conformal rejection boundaries:

```bash
python experiments/run_federated.py --config configs/default.yaml --rounds 20
```

### B. Comparative Baseline Suite
Evaluates competing decentralized and open-set baselines under identical non-IID client partitions:
* **FedAvg + MSP** (Maximum Softmax Probability)
* **FedAvg + Energy-OOD** (Free Helmholtz Energy)
* **FedProx + OpenMax** (Weibull EVT Calibration)
* **FedProto** (Prototype-based Federated Learning)

```bash
python experiments/run_baselines.py --rounds 20
```

### C. Architectural Ablation Suite
Isolates each foundational mechanism to evaluate individual marginal utility:
* Without VIB Latent Regularization ($\lambda_{\mathrm{vib}} = 0$)
* Without Balanced Logit Adjustment ($\tau_{\mathrm{bla}} = 0$)
* Without Total Variance Law (arithmetic mean covariance)
* Without Conformal Calibration (uncalibrated distance thresholding)

```bash
python experiments/run_ablations.py --rounds 20
```

### D. Conformal Sensitivity Analysis
Validates the finite-sample bounded false alarm property ($\mathrm{KFR} \le \alpha$) across significance levels $\alpha \in [0.03, 0.20]$:

```bash
python experiments/run_conformal_sweep.py --rounds 20
```

### E. Leave-One-Attack-Out (LOAO) Robustness
Evaluates zero-day generalization across rotating unknown attack classes:

```bash
python experiments/run_loao.py --rounds 20
```

### F. Wire-Speed & Latency Profiling
Profiles per-packet feature extraction and decision inference latency:

```bash
python experiments/run_wire_speed_profiling.py
```

---

## 5. Automated Verification & Testing

The test suite covers unit mechanics, loss formulations, numerical stability of covariance shrinkage, and end-to-end federated workflows with zero data leakage:

```bash
pytest tests/ -v
```

All 58 test routines pass with deterministic outputs across platforms.

---

## 6. Citation

If you find this codebase or methodology useful in your research, please cite:

```bibtex
@article{amini2026federated,
  title={Federated Open-Set Recognition for Blockchain Network Traffic under Non-IID Distributions},
  author={Amini, Mohammad Hassan and Entezari-Maleki, Reza},
  journal={Computer Networks},
  year={2026}
}
```

---

## 7. License

This project is licensed under the [MIT License](LICENSE).
