# PowerShell script to execute full research pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting Full Research Pipeline (Training, Baselines, Ablations)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Generate Synthetic Benchmark Dataset
Write-Host "[1/4] Generating synthetic benchmark dataset..." -ForegroundColor Yellow
python scripts/generate_synthetic_data.py

# 2. Run Main Federated Experiment
Write-Host "[2/4] Running main federated training and OSR evaluation..." -ForegroundColor Yellow
python experiments/run_federated.py --rounds 20

# 3. Run Baselines
Write-Host "[3/4] Running unified baseline comparison suite..." -ForegroundColor Yellow
python experiments/run_baselines.py --rounds 10

# 4. Run Ablation Suite
Write-Host "[4/4] Running architectural ablation suite..." -ForegroundColor Yellow
python experiments/run_ablations.py --rounds 10

# 5. Generate Paper Artifacts (Figures & Tables)
Write-Host "Generating publication figures and LaTeX tables..." -ForegroundColor Green
python scripts/generate_paper_artifacts.py

Write-Host "All experiments completed successfully!" -ForegroundColor Green
