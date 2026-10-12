#!/usr/bin/env bash
set -e

PYTHON_CMD=""
CANDIDATES=(
    "$CONDA_PREFIX/bin/python"
    "/opt/conda/bin/python3"
    "/opt/conda/bin/python"
    "$HOME/miniconda3/bin/python3"
    "$HOME/miniconda3/bin/python"
    "$HOME/anaconda3/bin/python3"
    "$HOME/anaconda3/bin/python"
    "$(which python3 2>/dev/null)"
    "$(which python 2>/dev/null)"
    "/usr/bin/python3"
)

for cand in "${CANDIDATES[@]}"; do
    if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -c "import torch" &>/dev/null; then
        PYTHON_CMD="$cand"
        echo "  -> Found Python with PyTorch already installed: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    for cand in "${CANDIDATES[@]}"; do
        if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -m pip --version &>/dev/null; then
            PYTHON_CMD="$cand"
            echo "  -> Selected Python with pip: $PYTHON_CMD"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="$(command -v python3)"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="$(command -v python)"
    else
        echo "[ERROR] Python 3.10+ not found! Please install Python."
        exit 1
    fi
fi
echo "Active Python: $PYTHON_CMD"

mkdir -p "$HOME/.local/bin"
ln -sf "$PYTHON_CMD" "$HOME/.local/bin/python"
ln -sf "$PYTHON_CMD" "$HOME/.local/bin/python3"
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/5] Verifying GPU & Hardware Environment ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "[WARNING] No NVIDIA GPU detected. Running in CPU mode."
fi

echo "=== [2/5] Extracting Codebase ==="
for czip in "chainguard_code.zip" "chainguard_osr_code.zip" "fedtros_code.zip"; do
    if [ -f "$czip" ]; then
        unzip -q -o "$czip"
        echo "  -> Code extracted successfully from $czip."
        break
    fi
done

echo "=== [3/5] Extracting Datasets ==="
mkdir -p data/raw
for dzip in \
    /content/drive/MyDrive/ChainGuard-OSR/chainguard_data_all4.zip \
    /content/drive/MyDrive/ChainGuard-OSR/chainguard_data_bnat.zip \
    /content/drive/MyDrive/ChainGuard-OSR/fedtros_data_all4.zip \
    /content/drive/MyDrive/ChainGuard-OSR/fedtros_data_bnat.zip \
    /content/drive/MyDrive/FedTROS/fedtros_data_all4.zip \
    /content/drive/MyDrive/fedtros_data_all4.zip \
    /content/drive/MyDrive/FedTROS/fedtros_data_bnat.zip \
    /content/drive/MyDrive/fedtros_data_bnat.zip \
    /content/fedtros_data_all4.zip \
    /content/fedtros_data_bnat.zip \
    chainguard_data_*.zip \
    fedtros_data_all4.zip \
    fedtros_data_*.zip; do
    if [ -f "$dzip" ]; then
        echo "  -> Extracting dataset archive: $dzip ..."
        unzip -q -o "$dzip" -d data/raw/
    fi
done
find data/raw -mindepth 2 -type f \( -name "*.csv" -o -name "*.json" \) -exec mv -f {} data/raw/ \; 2>/dev/null || true
for csv in b_nat.csv b_tat.csv ton_iot.csv cic_ids2017.csv; do
    if [ -f "$csv" ] && [ ! -f "data/raw/$csv" ]; then
        mv -f "$csv" data/raw/"$csv"
    fi
done
ls -lh data/raw/ 2>/dev/null || true

echo "=== [4/5] Checking and Installing Missing Dependencies ==="
if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
    echo "  [WARNING] 'pip' not found in $PYTHON_CMD. Bootstrapping pip..."
    "$PYTHON_CMD" -m ensurepip --upgrade 2>/dev/null || true
fi

$PYTHON_CMD - << 'EOF'
import sys
import subprocess
import importlib.util

REQUIRED_MAPPING = {
    "torch": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "numpy": "numpy>=1.26.0",
    "scipy": "scipy>=1.12.0",
    "pandas": "pandas>=2.2.0",
    "sklearn": "scikit-learn>=1.4.0",
    "matplotlib": "matplotlib>=3.8.0",
    "seaborn": "seaborn>=0.13.0",
    "yaml": "pyyaml>=6.0.1",
    "pydantic": "pydantic>=2.6.0",
    "rich": "rich>=13.7.0",
    "tqdm": "tqdm>=4.66.0"
}

missing = []
for mod, pkg in REQUIRED_MAPPING.items():
    if importlib.util.find_spec(mod) is None:
        missing.append((mod, pkg))

if missing:
    print(f"Installing {len(missing)} missing packages...")
    extra_flags = []
    try:
        h = subprocess.run([sys.executable, "-m", "pip", "install", "--help"], capture_output=True, text=True).stdout
        if "--break-system-packages" in h:
            extra_flags.append("--break-system-packages")
    except Exception:
        pass

    for mod, pkg in missing:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + extra_flags + pkg.split()
        res = subprocess.run(cmd)
        if res.returncode != 0:
            cmd_user = [sys.executable, "-m", "pip", "install", "-q", "--user"] + extra_flags + pkg.split()
            subprocess.run(cmd_user)

print("[SUCCESS] All dependencies verified and ready for experiments!")
EOF

echo "=== [5/5] Setup Complete! Ready for Execution ==="
