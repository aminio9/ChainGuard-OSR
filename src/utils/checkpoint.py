"""Checkpoint management for saving and restoring federated training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import torch

from src.utils.io import ensure_dir


class CheckpointManager:
    """Manages saving and loading federated training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = ensure_dir(checkpoint_dir)

    def save_checkpoint(
        self,
        round_idx: int,
        model_state_dict: Dict[str, torch.Tensor],
        dual_multipliers: Optional[Dict[int, torch.Tensor]] = None,
        global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        conformal_quantiles: Optional[Dict[int, float]] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        prefix: str = "checkpoint",
    ) -> Path:
        """Save a training checkpoint."""
        state = {
            "round": round_idx,
            "model_state_dict": model_state_dict,
            "dual_multipliers": dual_multipliers,
            "global_prototypes": global_prototypes,
            "conformal_quantiles": conformal_quantiles,
            "metrics": metrics or {},
        }
        filename = f"{prefix}_round_{round_idx:03d}.pt"
        target_path = self.checkpoint_dir / filename
        torch.save(state, target_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{prefix}_best.pt"
            torch.save(state, best_path)

        return target_path

    def load_checkpoint(self, path_or_round: str | Path | int, prefix: str = "checkpoint") -> Dict[str, Any]:
        """Load state dictionary from a specified round or path."""
        if isinstance(path_or_round, int):
            path = self.checkpoint_dir / f"{prefix}_round_{path_or_round:03d}.pt"
        else:
            path = Path(path_or_round)

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint
