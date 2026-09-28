"""Free Helmholtz Energy Out-of-Distribution Calibration (Formula 12)."""

from __future__ import annotations

import torch


def compute_free_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Free Helmholtz Energy score from unnormalized logits (Formula 12).

    Mathematical Formulation:
        E(x; w) = - T * log sum_{c=1}^C exp( f_c(x; w) / T )
                = - T * LogSumExp( logits / T )

    Physical Intuition:
        Known standard traffic flows sit in deep free-energy basins (strongly negative values).
        Out-of-distribution zero-day flows produce low logits, yielding higher / less-negative energy.

    Args:
        logits: Tensor of raw network logits, shape (batch, num_classes).
        temperature: Thermodynamic temperature scale parameter T (default 1.0).

    Returns:
        1D tensor of scalar energy values for each sample in the batch.
    """
    scaled_logits = logits / float(temperature)
    energy = -float(temperature) * torch.logsumexp(scaled_logits, dim=-1)
    return energy
