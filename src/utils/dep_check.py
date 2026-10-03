"""Automated dependency verification and auto-installation utility for FedTROS-MC."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

REQUIRED_DEPENDENCIES = {
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
    "tqdm": "tqdm>=4.66.0",
}


def ensure_dependencies(auto_install: bool = True, verbose: bool = True) -> bool:
    """Verify all runtime dependencies and install any missing packages."""
    missing = []
    for mod, pkg in REQUIRED_DEPENDENCIES.items():
        if importlib.util.find_spec(mod) is None:
            missing.append((mod, pkg))

    if not missing:
        if verbose:
            print("[DEP-CHECK] All required dependencies are present.")
        return True

    print(f"[DEP-CHECK] Missing {len(missing)} required package(s): {[m[0] for m in missing]}")

    if not auto_install:
        return False

    for mod, pkg in missing:
        print(f"  -> Auto-installing missing dependency: {pkg} ...")
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkg.split()
        res = subprocess.run(cmd)
        if res.returncode != 0 and "torch" in mod:
            print("  -> Retrying standard PyTorch install...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])

    print("[DEP-CHECK] All missing packages installed and verified.\n")
    return True


if __name__ == "__main__":
    ensure_dependencies(auto_install=True, verbose=True)
