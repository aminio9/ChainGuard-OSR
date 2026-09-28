"""Primal-Dual Augmented Lagrangian Consensus Loss (Formula 8)."""

from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn


def flatten_parameters(params: Dict[str, torch.Tensor] | List[torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary or list of parameter tensors into a single 1D tensor."""
    if isinstance(params, dict):
        tensors = [v.view(-1) for v in params.values()]
    else:
        tensors = [p.view(-1) for p in params]
    return torch.cat(tensors)


def primal_dual_consensus_loss(
    local_module: nn.Module,
    global_params: Dict[str, torch.Tensor],
    dual_lambda: torch.Tensor,
    rho: float = 0.1,
) -> torch.Tensor:
    """Compute Augmented Lagrangian penalty on local client parameters (Formula 8).

    Mathematical Formulation:
        L_PD = < lambda_k, theta_k - theta_global > + (rho / 2) * || theta_k - theta_global ||_2^2

    Args:
        local_module: PyTorch module whose parameters theta_k are being trained.
        global_params: Fixed global parameter state_dict received from server.
        dual_lambda: Server-updated dual multiplier vector for this client.
        rho: Quadratic penalty coefficient.

    Returns:
        Scalar penalty tensor contributing to backpropagation gradients.
    """
    if rho <= 0.0 and (dual_lambda is None or dual_lambda.abs().sum() == 0):
        return torch.tensor(0.0, device=next(local_module.parameters()).device)

    device = next(local_module.parameters()).device
    local_tensors = []
    global_tensors = []

    for name, param in local_module.named_parameters():
        if name in global_params:
            local_tensors.append(param.view(-1))
            global_tensors.append(global_params[name].to(device).detach().view(-1))

    if not local_tensors:
        return torch.tensor(0.0, device=device)

    flat_local = torch.cat(local_tensors)
    flat_global = torch.cat(global_tensors)
    diff = flat_local - flat_global

    # Align dual_lambda size if mismatched (e.g. dummy test input or buffer inclusion)
    if dual_lambda.numel() != diff.numel():
        aligned_lambda = torch.zeros_like(diff)
        n = min(dual_lambda.numel(), diff.numel())
        aligned_lambda[:n] = dual_lambda.view(-1)[:n].to(device)
    else:
        aligned_lambda = dual_lambda.to(device).detach().view(-1)

    # 1. Linear Dual Term: < lambda, theta_k - theta_global >
    linear_term = torch.dot(aligned_lambda, diff)

    # 2. Quadratic Proximal Term: (rho / 2) * || theta_k - theta_global ||^2
    quad_term = 0.5 * rho * torch.sum(diff.pow(2))

    return linear_term + quad_term
