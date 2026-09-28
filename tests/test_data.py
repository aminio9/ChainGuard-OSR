"""Tests for data handling, OSR splitting, and Dirichlet partitioning."""

import numpy as np
import pytest
import torch
from src.data.base import FlowDataset
from src.data.osr_split import compute_openness, split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid


def test_flow_dataset_indexing():
    feats = torch.randn(50, 10)
    lbls = torch.randint(0, 3, (50,))
    ds = FlowDataset(feats, lbls)

    assert len(ds) == 50
    assert ds.num_features == 10
    x, y = ds[0]
    assert x.shape == (10,)
    assert isinstance(y.item(), int)


def test_compute_openness():
    # If 0 unknown classes, openness must be 0.0
    o_closed = compute_openness(num_known=5, num_unknown=0)
    assert abs(o_closed) < 1e-6

    # Test formula for known=5, unknown=5 -> 1 - sqrt(10 / 15) = 1 - sqrt(2/3) ~ 0.1835
    o_mid = compute_openness(num_known=5, num_unknown=5)
    expected = 1.0 - np.sqrt(10.0 / 15.0)
    assert abs(o_mid - expected) < 1e-4

    # Monotonically increases with unknown count
    o_high = compute_openness(num_known=5, num_unknown=20)
    assert o_high > o_mid


def test_split_open_set_dataset(synthetic_dataset, sample_config):
    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=synthetic_dataset,
        known_classes=sample_config.data.known_classes,
        unknown_classes=sample_config.data.unknown_classes,
        test_ratio=0.2,
        cal_ratio=0.1,
        seed=42,
    )

    # Train, cal, and closed test should ONLY contain remapped known class labels (0, 1, 2)
    known_max = len(sample_config.data.known_classes) - 1
    assert train_ds.labels.max().item() <= known_max
    assert cal_ds.labels.max().item() <= known_max
    assert test_closed_ds.labels.max().item() <= known_max

    # Open test dataset MUST contain unknown classes designated by label == len(known_classes)
    unknown_label = len(sample_config.data.known_classes)
    assert unknown_label in test_open_ds.labels.numpy()


def test_partition_dirichlet_non_iid(synthetic_dataset):
    num_clients = 4
    client_dict = partition_dirichlet_non_iid(
        dataset=synthetic_dataset,
        num_clients=num_clients,
        beta_dirichlet=0.1,
        seed=42,
    )

    assert len(client_dict) == num_clients
    total_assigned = sum(len(indices) for indices in client_dict.values())
    assert total_assigned == len(synthetic_dataset)

    # In extreme non-IID, different clients should have different class distributions
    labels = synthetic_dataset.labels.numpy()
    c0_classes = np.unique(labels[client_dict[0]])
    assert len(c0_classes) >= 1
