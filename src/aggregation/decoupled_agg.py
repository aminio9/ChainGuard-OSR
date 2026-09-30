"""Decoupled Server Aggregation (LaTeX Section 4.2.3, Equations for Decoupled Aggregation).

Decouples Feature Representation Learning (Encoder theta) from Classification Head (psi):
    1. Global Encoder:
       theta_g^{t+1} = sum_{k=1}^K (n_k / N) * theta_k^{t+1}
    2. Global Classifier Head (Row-wise per class c in C_K):
       psi_{g, c}^{t+1} = sum_{k in K_c} (n_c^{(k)} / N_c) * psi_{k, c}^{t+1}
       where K_c = {k | n_c^{(k)} > 0} and N_c = sum_{k in K_c} n_c^{(k)}.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def aggregate_decoupled_parameters(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int, Union[List[int], np.ndarray, torch.Tensor]]],
    previous_global_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Aggregate client parameters using Decoupled Server Aggregation.

    Args:
        client_updates: List of tuples (client_state_dict, total_samples, samples_per_class).
        previous_global_state: Global model state dict from previous round (fallback for unobserved classes).

    Returns:
        Aggregated global model state dictionary.
    """
    if not client_updates:
        raise ValueError("Client updates list cannot be empty for decoupled aggregation.")

    total_samples = sum(num_samples for _, num_samples, _ in client_updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be positive.")

    first_params = client_updates[0][0]
    global_params: Dict[str, torch.Tensor] = {}

    # Initialize from previous global state or first client params
    if previous_global_state is not None:
        for k, v in previous_global_state.items():
            global_params[k] = v.clone()
    else:
        for k, v in first_params.items():
            global_params[k] = v.clone()

    # Identify classifier keys
    classifier_weight_key = "classifier.linear.weight"
    classifier_bias_key = "classifier.linear.bias"

    # 1. Aggregate Encoder and non-classifier parameters (standard sample-weighted FedAvg)
    for key in first_params.keys():
        if key in (classifier_weight_key, classifier_bias_key):
            continue

        weighted_sum = torch.zeros_like(first_params[key], dtype=torch.float32)
        for params, num_samples, _ in client_updates:
            weight = float(num_samples) / float(total_samples)
            weighted_sum += params[key].float() * weight
        global_params[key] = weighted_sum.to(first_params[key].dtype)

    # 2. Decoupled Aggregation of Classifier Head per class c
    if classifier_weight_key in first_params:
        num_classes = first_params[classifier_weight_key].size(0)
        has_bias = classifier_bias_key in first_params

        new_weight = torch.zeros_like(first_params[classifier_weight_key], dtype=torch.float32)
        new_bias = torch.zeros_like(first_params[classifier_bias_key], dtype=torch.float32) if has_bias else None

        for c in range(num_classes):
            # Find clients that observed class c
            reporting_clients = []
            for params, _, spc in client_updates:
                count_c = int(spc[c]) if c < len(spc) else 0
                if count_c > 0:
                    reporting_clients.append((params, count_c))

            total_c_samples = sum(cnt for _, cnt in reporting_clients)

            if total_c_samples > 0:
                # Class-specific weighted average
                weight_row = torch.zeros_like(first_params[classifier_weight_key][c], dtype=torch.float32)
                for params, cnt in reporting_clients:
                    w_k = float(cnt) / float(total_c_samples)
                    weight_row += params[classifier_weight_key][c].float() * w_k
                new_weight[c] = weight_row

                if has_bias and new_bias is not None:
                    bias_val = torch.tensor(0.0, dtype=torch.float32)
                    for params, cnt in reporting_clients:
                        w_k = float(cnt) / float(total_c_samples)
                        bias_val += params[classifier_bias_key][c].float() * w_k
                    new_bias[c] = bias_val
            else:
                # Fallback: keep previous value
                new_weight[c] = global_params[classifier_weight_key][c].float()
                if has_bias and new_bias is not None:
                    new_bias[c] = global_params[classifier_bias_key][c].float()

        global_params[classifier_weight_key] = new_weight.to(first_params[classifier_weight_key].dtype)
        if has_bias and new_bias is not None:
            global_params[classifier_bias_key] = new_bias.to(first_params[classifier_bias_key].dtype)

    return global_params
