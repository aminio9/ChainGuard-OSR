"""Dirichlet-based non-IID label distribution skew partitioning (Formula 1)."""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import torch
from src.data.base import FlowDataset


def partition_dirichlet_non_iid(
    dataset: FlowDataset,
    num_clients: int,
    beta_dirichlet: float = 0.1,
    min_samples_per_client: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Partition dataset across clients using Dirichlet distribution on class proportions.

    Mathematical Formulation (Formula 1):
        p_k ~ Dir_C(beta_D * 1_C)
    where beta_D is the Dirichlet concentration parameter (e.g. 0.1 for extreme non-IID skew).

    Args:
        dataset: The dataset containing flow vectors and class labels.
        num_clients: Total number of federated clients (K).
        beta_dirichlet: Dirichlet concentration parameter beta_D.
        min_samples_per_client: Ensure every client has at least this number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping client_id (0..K-1) to list of assigned dataset sample indices.
    """
    rng = np.random.default_rng(seed)
    labels = dataset.labels.numpy()
    num_samples = len(labels)
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Group sample indices by class
    class_indices: Dict[int, np.ndarray] = {
        int(c): rng.permutation(np.where(labels == c)[0]) for c in unique_classes
    }

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # Dirichlet allocation per class
    for c in unique_classes:
        c_idx = class_indices[int(c)]
        num_class_samples = len(c_idx)
        if num_class_samples == 0:
            continue

        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(beta_dirichlet, num_clients))
        # Scale proportions to actual sample counts
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * num_class_samples).astype(int)[:-1]

        client_subsets = np.split(c_idx, splits)
        for client_id, subset in enumerate(client_subsets):
            client_indices[client_id].extend(subset.tolist())

    # Ensure every client has minimal samples by re-allocating if needed
    effective_min = min(min_samples_per_client, max(1, num_samples // num_clients))
    for client_id in range(num_clients):
        while len(client_indices[client_id]) < effective_min:
            # Find client with maximum samples
            max_client = max(client_indices.keys(), key=lambda k: len(client_indices[k]))
            if len(client_indices[max_client]) <= effective_min:
                break
            stolen_idx = client_indices[max_client].pop()
            client_indices[client_id].append(stolen_idx)

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices
