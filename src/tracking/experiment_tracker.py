"""Experiment tracker managing experiment runs, config snapshots, and metric histories."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess

from src.utils.io import ensure_dir, save_json
from src.tracking.reproducibility import get_system_info


class ExperimentTracker:
    """Tracks training runs, metrics per round, and saves experiment snapshots."""

    def __init__(self, run_name: str, base_dir: str | Path = "results/runs"):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_name}_{self.timestamp}"
        self.run_dir = ensure_dir(Path(base_dir) / self.run_id)
        self.history: List[Dict[str, Any]] = []

    def get_git_commit(self) -> Optional[str]:
        """Retrieve current git commit hash if available."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def initialize_run(self, config: Dict[str, Any]) -> None:
        """Save run metadata, config snapshot, and system info."""
        metadata = {
            "run_id": self.run_id,
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": self.get_git_commit(),
            "system_info": get_system_info(),
            "config": config,
        }
        save_json(metadata, self.run_dir / "run_metadata.json")

    def log_round(self, round_idx: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a federated communication round."""
        entry = {"round": round_idx, **metrics}
        self.history.append(entry)

    def finish_run(self, final_results: Optional[Dict[str, Any]] = None) -> None:
        """Finalize the run and persist full history and final results."""
        save_json(self.history, self.run_dir / "history.json")
        if final_results is not None:
            save_json(final_results, self.run_dir / "final_results.json")
