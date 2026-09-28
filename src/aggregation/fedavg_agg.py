"""Weighted parameter aggregation (FedAvg - Algorithm 1 Line 12)."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch


def aggregate_fedavg_parameters(
    client_params_and_weights: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """Aggregate model parameters across clients weighted by sample volume.

    Mathematical Formulation:
        theta_global = sum_{k=1}^K (N_k / N) * theta_k
    """
    total_samples = sum(num_samples for _, num_samples in client_params_and_weights)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be positive.")

    first_params = client_params_and_weights[0][0]
    global_params: Dict[str, torch.Tensor] = {}

    for key in first_params.keys():
        weighted_sum = torch.zeros_like(first_params[key], dtype=torch.float32)
        for params, num_samples in client_params_and_weights:
            weight = float(num_samples) / float(total_samples)
            weighted_sum += params[key].float() * weight
        global_params[key] = weighted_sum.to(first_params[key].dtype)

    return global_params
