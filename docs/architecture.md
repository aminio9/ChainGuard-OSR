# System Architecture & Codebase Design

## 1. Modular Architecture Overview

```
                      +-----------------------------+
                      |     configs/default.yaml    |
                      +--------------+--------------+
                                     |
                                     v
+---------------------------------------------------------------------------------+
|                                src/data/                                        |
|  - synthetic.py       : Generates P2P flow vectors (43 dimensions)             |
|  - partitioning.py    : Extreme Non-IID Dirichlet partitioner (Beta_D = 0.1)    |
|  - osr_split.py       : Known vs Unknown zero-day split & Openness computation  |
+------------------------------------+--------------------------------------------+
                                     |
             +-----------------------+-----------------------+
             v                                               v
+-----------------------------+               +-----------------------------+
|    Client 1 (src/federated) |  ... (P2P)    |    Client K (src/federated) |
|  - VIB Encoder (Formula 5)  |               |  - VIB Encoder (Formula 5)  |
|  - Decoupled Heads (Form. 3)|               |  - Decoupled Heads (Form. 3)|
|  - Composite Loss (Form. 9) |               |  - Composite Loss (Form. 9) |
|  - Prototypes (mu, Sigma)   |               |  - Prototypes (mu, Sigma)   |
+--------------+--------------+               +--------------+--------------+
               \                                             /
                \------ Upload {theta_k, mu_k, Sigma_k} ----/
                                     |
                                     v
+---------------------------------------------------------------------------------+
|                       Server Orchestrator (src/federated)                       |
|  - FedAvg Parameter Aggregator     : theta_global = sum (N_k/N) theta_k         |
|  - Primal-Dual Multiplier Update  : lambda_k = lambda_k + rho (theta_k - theta)|
|  - Bayesian Covariance Aggregator : mu_global = (sum Sigma^-1)^-1 (sum Sigma^-1)|
+------------------------------------+--------------------------------------------+
                                     |
                        Broadcast theta_global, mu_global
                                     |
                                     v
+---------------------------------------------------------------------------------+
|                       Edge Inference Pipeline (src/osr/)                        |
|                                                                                 |
|  Raw Flow x ---> [Z-Score Normalizer] ---> x in R^43                           |
|                         |                                                       |
|                         v                                                       |
|                  [VIB Encoder]                                                  |
|                         |                                                       |
|                         +---> z = mu(x) in R^64                                 |
|                         |                                                       |
|       +-----------------+-----------------+                                     |
|       |                                   |                                     |
|       v                                   v                                     |
|  [Head C_phi]                      [EVT Weibull CDF]                            |
|       |                                   |                                     |
|  Raw Logits                       Discount Factor omega                         |
|       |                                   |                                     |
|       v                                   v                                     |
|  [Free Energy E(x)]                [Calibrated Probs]                           |
|       |                                   |                                     |
|       +-----------------+-----------------+                                     |
|                         |                                                       |
|                         v                                                       |
|         [Mondrian Conformal Set C_{1-alpha}(x)]                                 |
|                         |                                                       |
|        +----------------+----------------+                                      |
|        |                                 |                                      |
|   C == empty                       C != empty                                   |
|        |                                 |                                      |
|        v                                 v                                      |
|  [ALERT: Zero-Day Attack]          [Known Class: argmax p_c]                    |
+---------------------------------------------------------------------------------+
```

## 2. Directory Layout & Separation of Responsibilities

* `configs/`: Declarative YAML configurations isolating hyperparameters, datasets, baselines, and ablation toggles from executable code.
* `src/data/`: Data loading, synthetic generation, Dirichlet non-IID partitioning, and Open-Set Known/Unknown splitting.
* `src/models/`: Modular neural components: Stochastic VIB encoder, Deterministic encoder, Classification head, and Uncertainty density head.
* `src/losses/`: Pure, mathematically grounded loss functions implementing Class-Balanced focal loss, analytical Gaussian KL divergence, Orthogonal anchoring, and Primal-Dual consensus.
* `src/federated/`: Flower-compatible decentralized coordination: clients, server, and custom aggregation strategies.
* `src/aggregation/`: Parameter averaging, Bayesian inverse covariance prototype synthesis, and dual multiplier updates.
* `src/osr/`: 3-stage hierarchical open-set recognition: Weibull EVT fitting, Free Helmholtz energy scoring, and Mondrian conformal set generation.
* `src/metrics/`: Dedicated, reproducible metrics for multiclass classification, open-set AUROC, false alarm rates, and conformal coverage.
* `src/tracking/`: Structured JSONL logging, global deterministic seed initialization, and hardware environment profiling.
* `src/visualization/`: Automated publication vector plots and LaTeX table generators.
* `experiments/`: Experiment runners for standard federated execution, baseline comparisons, and ablation suites.
* `scripts/`: Batch pipelines for data preparation, cluster runs, and LaTeX artifact extraction.
* `tests/`: 100% pytest-tested suite verifying mathematical invariants, numerical stability, and edge cases.
