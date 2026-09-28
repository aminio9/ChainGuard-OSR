"""Server-side Primal-Dual multiplier update (Formula 8 & Algorithm 1 Line 13)."""

from __future__ import annotations

from typing import Dict
import torch


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten state dict into 1D continuous tensor."""
    tensors = [v.view(-1).float() for v in state_dict.values()]
    return torch.cat(tensors)


def update_dual_multipliers(
    current_lambdas: Dict[int, torch.Tensor],
    client_params: Dict[int, Dict[str, torch.Tensor]],
    global_params: Dict[str, torch.Tensor],
    rho: float = 0.1,
) -> Dict[int, torch.Tensor]:
    """Execute dual ascent step on server to update Lagrangian multipliers for participating clients.

    Mathematical Formulation (Formula 8 & Algorithm 1 Line 13):
        lambda_k^{(t+1)} = lambda_k^{(t)} + rho * (theta_k^{(t+1)} - theta_global^{(t+1)})

    Args:
        current_lambdas: Dict mapping client_id -> current dual multiplier tensor.
        client_params: Dict mapping client_id -> local encoder state_dict after local training.
        global_params: Aggregated global encoder state_dict for round t+1.
        rho: Quadratic penalty coefficient.

    Returns:
        Updated dictionary mapping client_id -> new dual multiplier tensor.
    """
    if rho <= 0.0:
        return current_lambdas

    flat_global = flatten_state_dict(global_params)
    updated_lambdas: Dict[int, torch.Tensor] = {}

    for client_id, params in client_params.items():
        flat_client = flatten_state_dict(params)

        # Ensure lambda exists on matching device
        if client_id not in current_lambdas or current_lambdas[client_id] is None:
            prev_lambda = torch.zeros_like(flat_global)
        else:
            prev_lambda = current_lambdas[client_id].to(flat_global.device)

        diff = flat_client.to(flat_global.device) - flat_global
        new_lambda = prev_lambda + rho * diff
        updated_lambdas[client_id] = new_lambda.detach()

    # Retain unparticipating clients' multipliers unchanged
    for client_id, prev_l in current_lambdas.items():
        if client_id not in updated_lambdas:
            updated_lambdas[client_id] = prev_l

    return updated_lambdas
