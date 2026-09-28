#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Full Research Pipeline (Training, Baselines, Ablations)"
echo "=========================================================="

# 1. Generate Synthetic Benchmark Dataset
echo "[1/4] Generating synthetic benchmark dataset..."
python scripts/generate_synthetic_data.py

# 2. Run Main Federated Experiment
echo "[2/4] Running main federated training and OSR evaluation..."
python experiments/run_federated.py --rounds 20

# 3. Run Baselines
echo "[3/4] Running unified baseline comparison suite..."
python experiments/run_baselines.py --rounds 10

# 4. Run Ablation Suite
echo "[4/4] Running architectural ablation suite..."
python experiments/run_ablations.py --rounds 10

# 5. Generate Paper Artifacts (Figures & Tables)
echo "Generating publication figures and LaTeX tables..."
python scripts/generate_paper_artifacts.py

echo "All experiments completed successfully!"
