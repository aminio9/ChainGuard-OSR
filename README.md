# Federated Open-Set Recognition and Conformal Defense in Decentralized Networks

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)

A research-grade, reproducible, modular, and scalable Python framework for **Federated Open-Set Intrusion Detection** in decentralized peer-to-peer (P2P) blockchain networks.

---

## 1. Project Overview

In decentralized peer-to-peer networks, edge nodes face novel zero-day attacks and extreme data heterogeneity (non-IID label distribution skew with Dirichlet concentration $\beta_D = 0.1$). Existing federated learning models suffer from severe client drift and dimensional collapse when clients lack specific attack classes, causing conventional open-set algorithms to generate unconstrained false alarms.

This framework implements a 5-pillar mathematical architecture:
1. **Variational Information Bottleneck (VIB) Latent Regularization** ($\mathcal{L}_{\mathrm{VIB}}$) - Compresses representations to a standard Gaussian sphere, eliminating network flow noise.
2. **Decoupled Dual-Branch Neural Architecture** ($E_\theta, C_\phi, H_\psi$) - Isolates known-class classification gradients from anomaly density estimation.
3. **Class-Balanced Focal Cross-Entropy** ($\mathcal{L}_{\mathrm{CB}}$) - Rebalances rare attack classes using sample effective volumes.
4. **Fixed Orthogonal Classifier Anchoring** ($\mathcal{L}_{\mathrm{anchor}}$) - Constrains class weight vectors to pairwise orthogonality on the Stiefel manifold.
5. **Primal-Dual Augmented Lagrangian Consensus** ($\mathcal{L}_{\mathrm{PD}}$) - Provably neutralizes client drift with non-asymptotic sublinear convergence (Theorem 1).
6. **Bayesian Covariance-Weighted Prototype Synthesis** ($\bar{\mu}_{\mathrm{global}, c}$) - Aggregates class cluster centers weighted by inverse local covariance matrices.
7. **3-Stage Hierarchical Edge Inference** - Combines Extreme Value Theory (EVT) Weibull discounting, Free Helmholtz Energy scoring, and Finite-Sample Mondrian Conformal Prediction to mathematically guarantee false alarm rates bounded by $\alpha = 0.05$ (Theorem 2).

---

## 2. Project Structure

```
Code/
├── .gitignore                     # Git ignore rules for data, checkpoints, and cache
├── .python-version                # Python 3.12 specification
├── pyproject.toml                 # UV / Hatchling project configuration
├── README.md                      # Comprehensive project documentation
├── configs/                       # Declarative YAML experiment configurations
│   ├── default.yaml               # Master experiment configuration
│   ├── datasets/                  # B-TAT, B-NAT, and Synthetic P2P configs
│   ├── models/                    # VIB dual-branch and baseline MLP configs
│   ├── federated/                 # Primal-Dual, FedAvg, FedProx configs
│   ├── osr/                       # Mondrian conformal, EVT, Energy configs
│   ├── baselines/                 # Benchmark configurations for competing methods
│   └── ablations/                 # Architectural ablation configurations
├── src/                           # Source code (modular research package)
│   ├── config.py                  # Pydantic schema validation & YAML loader
│   ├── data/                      # Datasets, synthetic generators, Dirichlet partitioner
│   ├── models/                    # VIB stochastic encoder and decoupled heads
│   ├── losses/                    # CB focal loss, VIB loss, Orthogonal, Primal-Dual
│   ├── federated/                 # Client, Server, and Flower-compatible strategies
│   ├── aggregation/               # FedAvg, Bayesian covariance prototypes, dual updates
│   ├── osr/                       # Weibull EVT, Helmholtz energy, Mondrian conformal
│   ├── metrics/                   # Multiclass, open-set AUROC/F1, conformal metrics
│   ├── tracking/                  # Structured JSONL logger, seed setter, experiment profiler
│   ├── visualization/             # High-DPI publication plots & LaTeX tables
│   └── utils/                     # Checkpointing, hardware device, and I/O
├── experiments/                   # Executable research runners
│   ├── run_federated.py           # Main training & open-set evaluation script
│   ├── run_baselines.py           # Unified baseline comparison benchmark runner
│   ├── run_ablations.py           # 6-component ablation study runner
│   └── evaluate_checkpoint.py     # Standalone evaluation for saved checkpoints
├── scripts/                       # Automation pipelines
│   ├── generate_synthetic_data.py # Generates 10k P2P benchmark flows
│   ├── run_all_experiments.ps1    # PowerShell batch runner
│   ├── run_all_experiments.sh     # Bash batch runner
│   └── generate_paper_artifacts.py# Vector plots & LaTeX table generator
├── tests/                         # Comprehensive pytest test suite (100% pass)
├── docs/                          # In-depth architectural & theoretical documentation
│   ├── architecture.md            # System architecture & data flow diagrams
│   ├── methodology_mapping.md     # Formula-by-formula mapping to source code
│   └── reviewer_defense_guide.md  # 28 Reviewer 2 challenges and code proofs
├── results/                       # Generated experiment logs, JSONL, and figures
└── checkpoints/                   # Saved model weights, prototypes, and dual states
```

---

## 3. Requirements

* **OS:** Linux / Windows 10+ / macOS (x86_64 or Apple Silicon)
* **Python:** $\ge 3.11$ (Recommended: 3.12)
* **Package Manager:** `uv` (Recommended) or standard `pip`
* **Hardware:** Commodity CPU or CUDA-capable GPU ($\ge 4$ GB VRAM)

---

## 4. Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone <repository-url>
cd Code
```

---

## 5. Environment Setup (UV)

We recommend using **Astral UV** for deterministic virtual environment management:

```bash
# 1. Create a isolated virtual environment with Python 3.12
uv venv --python 3.12

# 2. Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Synchronize all dependencies including development tools
uv pip install -e ".[dev]"
```

---

## 6. Dataset Preparation

### Synthetic P2P Blockchain Dataset
To generate 10,000 synthetic P2P blockchain network flow samples (43 flow features, extreme non-IID label skew):

```bash
python scripts/generate_synthetic_data.py
```

### Real-World Benchmarks (B-TAT & B-NAT)
Place preprocessed CSV files under `data/`:
* `data/b_tat/b_tat.csv` (Blockchain Transaction and Attack Traffic)
* `data/b_nat/b_nat.csv` (Blockchain Network Attack Telemetry)

Update the dataset path in `configs/datasets/b_tat.yaml` or set `--config configs/datasets/b_tat.yaml`.

---

## 7. Configuration

All hyperparameters are isolated in declarative YAML files under `configs/`:
* Learning rate, optimizer, weight decay (`configs/default.yaml`)
* Client count, active fraction, Dirichlet $\beta_D$ (`configs/federated/primal_dual.yaml`)
* Loss penalties: $\lambda_{\mathrm{vib}}, \lambda_{\mathrm{anchor}}, \rho, \beta_{\mathrm{cb}}$ (`configs/default.yaml`)
* Conformal significance level $\alpha = 0.05$ (`configs/osr/mondrian_conformal.yaml`)

No source code modifications are required to run customized experimental regimes.

---

## 8. Training

Execute federated training and open-set evaluation:

```bash
python experiments/run_federated.py --config configs/default.yaml --rounds 50
```

Real-time structured loss logs are written to both standard output and `results/runs/<run_id>/metrics.jsonl`. Model checkpoints, global prototypes, and dual multipliers are persisted in `checkpoints/`.

---

## 9. Evaluation

To evaluate a stored model checkpoint against unseen zero-day attacks:

```bash
python experiments/evaluate_checkpoint.py --checkpoint checkpoints/checkpoint_best.pt --config configs/default.yaml
```

Outputs:
* Closed-Set Accuracy (%)
* Open-Set AUROC (%)
* Open-Set Macro F1 (%)
* False Alarm Rate (FAR %) with Theorem 2 theoretical bound verification ($\text{FAR} \le \alpha = 5.0\%$)
* Conformal Prediction Set Average Cardinality $|\mathcal{C}(x)|$

---

## 10. Baselines

Run the unified benchmark suite comparing against standard federated baselines:

```bash
python experiments/run_baselines.py --rounds 20
```

Included Baselines:
1. **FedAvg + MSP** (Maximum Softmax Probability)
2. **FedProx + OpenMax** (Bendale & Boult, CVPR 2016)
3. **FedProto + Free Energy** (Liu et al., NeurIPS 2020)
4. **Full Method (Proposed)**

Outputs comparison JSON and ready-to-publish LaTeX table in `results/baselines/table_baselines.tex`.

---

## 11. Ablation Studies

Evaluate the individual contribution of each architectural pillar:

```bash
python experiments/run_ablations.py --rounds 20
```

Evaluated Variants:
* `Full Method`
* `w/o VIB Latent Regularization` ($\lambda_{\mathrm{vib}} = 0$, deterministic encoder)
* `w/o Orthogonal Anchoring` ($\lambda_{\mathrm{anchor}} = 0$)
* `w/o Primal-Dual Consensus` ($\rho = 0$)
* `w/o Covariance Prototype Aggregation` (arithmetic mean)
* `w/o Mondrian Conformal Calibration`
* `w/o Class-Balanced Focal Loss` (standard Cross-Entropy)

Results are recorded in `results/ablations/table_ablations.tex`.

---

## 12. Result Analysis

To inspect metric trajectories and examine structured run logs:

```python
from src.utils.io import load_json

run_history = load_json("results/runs/<run_id>/history.json")
final_results = load_json("results/runs/<run_id>/final_results.json")
print("AUROC:", final_results["auroc"])
print("False Alarm Rate:", final_results["false_alarm_rate"])
```

---

## 13. Visualization

Generate publication-ready figures (300 DPI PNG & PDF vector graphics) and LaTeX tables:

```bash
python scripts/generate_paper_artifacts.py
```

Generated Figures in `results/paper_artifacts/`:
* `fig4_auroc_vs_openness.pdf` (AUROC across increasing openness $\mathcal{O}$)
* `fig4_auroc_vs_openness.png`
* `table1_benchmark_summary.tex` (Master LaTeX table)

---

## 14. Reproducibility

Every run logs complete environment telemetry:
* PyTorch random seed, NumPy seed, and Python hash seed set deterministically
* `torch.backends.cudnn.deterministic = True`
* Git commit hash recorded in `run_metadata.json`
* Hardware configuration (GPU model, CUDA driver, CPU specs) logged

Run the full automated test suite to verify numerical stability:

```bash
pytest -v
```

---

## 15. Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{federated_conformal_osr_2026,
  title={Federated Open-Set Recognition and Conformal Defense in Decentralized Networks},
  author={Research Team},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026}
}
```
