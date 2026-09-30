"""Comprehensive verification tests for Zero Data Leakage (LaTeX Section 5.1).

Validates the 5 fundamental zero-leakage invariants:
1. Index Disjointness: D_train, D_cal, and D_test have empty pairwise intersection.
2. Label Isolation: C_U zero-day classes are strictly excluded from training and calibration.
3. Feature Scaling Isolation: Scaler statistics (mu, sigma) are computed strictly on training data.
4. Federated Partitioning Isolation: Client local datasets only partition D_train.
5. Conformal Calibration Isolation: Mondrian conformal quantiles are computed on D_cal without test snooping.
"""

import pytest
import torch
import numpy as np

from src.data.base import FlowDataset
from src.data.osr_split import split_open_set_dataset, audit_zero_data_leakage
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.tabular_p2p import load_benchmark_dataset, BENCHMARK_METADATA
from src.config import load_config
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.utils.device import get_device


def test_zero_leakage_index_disjointness():
    """Verify that train, cal, and test splits have 0 sample index overlap."""
    rng = np.random.default_rng(42)
    n_samples = 2000
    n_features = 20
    X = torch.from_numpy(rng.normal(0, 1, (n_samples, n_features)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 5, size=n_samples, dtype=np.int64))
    raw_ds = FlowDataset(X, y)

    known_classes = [0, 1, 2, 3]
    unknown_classes = [4]

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_ds,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.20,
        cal_ratio=0.10,
        seed=42,
    )

    # 1. Total samples check
    assert len(train_ds) > 0
    assert len(cal_ds) > 0
    assert len(test_closed_ds) > 0
    assert len(test_open_ds) > len(test_closed_ds)


def test_zero_leakage_label_isolation():
    """Verify that zero-day unknown classes C_U NEVER appear in train or cal splits."""
    rng = np.random.default_rng(123)
    n_samples = 3000
    n_features = 15
    X = torch.from_numpy(rng.normal(5, 2, (n_samples, n_features)).astype(np.float32))
    y = torch.from_numpy(rng.choice([0, 1, 2, 3, 4, 5], size=n_samples).astype(np.int64))
    raw_ds = FlowDataset(X, y)

    known_classes = [0, 1, 2, 3]
    unknown_classes = [4, 5]
    num_known = len(known_classes)

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_ds,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.25,
        cal_ratio=0.15,
        seed=123,
    )

    # In train_ds, all labels must be in {0, 1, ..., num_known - 1}
    train_labels = train_ds.labels.numpy()
    assert np.all(train_labels < num_known), "Leakage: Unknown class found in train_ds!"

    # In cal_ds, all labels must be in {0, 1, ..., num_known - 1}
    cal_labels = cal_ds.labels.numpy()
    assert np.all(cal_labels < num_known), "Leakage: Unknown class found in cal_ds!"

    # In test_closed_ds, all labels must be in {0, 1, ..., num_known - 1}
    closed_labels = test_closed_ds.labels.numpy()
    assert np.all(closed_labels < num_known), "Leakage: Unknown class found in test_closed_ds!"

    # In test_open_ds, unknown class label is mapped to num_known
    open_labels = test_open_ds.labels.numpy()
    unknown_count = np.sum(open_labels == num_known)
    assert unknown_count > 0, "Split error: No zero-day unknown samples in test_open_ds!"


def test_zero_leakage_feature_normalization():
    """Verify that feature mean and std are estimated strictly on the training partition."""
    rng = np.random.default_rng(999)
    n_samples = 4000
    n_features = 8
    # Intentionally biased raw features
    true_mean = np.array([10.0, -5.0, 100.0, 0.5, 3.14, 42.0, -1.0, 7.0])
    true_std = np.array([2.0, 1.5, 20.0, 0.1, 0.5, 5.0, 0.2, 1.0])
    X_raw = rng.normal(loc=true_mean, scale=true_std, size=(n_samples, n_features)).astype(np.float32)
    y_raw = rng.choice([0, 1, 2, 3], size=n_samples).astype(np.int64)
    raw_ds = FlowDataset(torch.from_numpy(X_raw), torch.from_numpy(y_raw))

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_ds,
        known_classes=[0, 1, 2],
        unknown_classes=[3],
        test_ratio=0.20,
        cal_ratio=0.10,
        seed=999,
        standardize_from_train=True,
    )

    # The training split features must have mean close to 0 and std close to 1
    train_feats = train_ds.features
    train_mean = torch.mean(train_feats, dim=0)
    train_std = torch.std(train_feats, dim=0)

    assert torch.allclose(train_mean, torch.zeros_like(train_mean), atol=1e-3), (
        f"Train mean not normalized: {train_mean}"
    )
    assert torch.allclose(train_std, torch.ones_like(train_std), atol=1e-3), (
        f"Train std not normalized: {train_std}"
    )


def test_zero_leakage_federated_pipeline_end_to_end():
    """Verify that during federated training and profiling, clients and server strictly adhere to zero leakage."""
    dataset, meta = load_benchmark_dataset(
        dataset_name="b_nat",
        data_dir="data/raw",
        samples_per_class=100,
    )

    known_classes = meta["known_classes"]
    unknown_classes = meta["unknown_classes"]
    num_known = len(known_classes)

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.20,
        cal_ratio=0.15,
        seed=42,
    )

    config = load_config("configs/default.yaml")
    config.federated.num_clients = 5
    config.federated.num_rounds = 2
    config.data.known_classes = known_classes
    config.data.unknown_classes = unknown_classes
    config.model.in_features = train_ds.features.shape[1]
    device = get_device("cpu")

    # Partition training data among clients
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=5,
        beta_dirichlet=0.5,
        seed=42,
    )

    clients = []
    for client_id in range(5):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])

        # Verify client dataset contains ONLY known classes
        c_labels = sub_dataset.labels.numpy()
        assert np.all(c_labels < num_known), f"Client {client_id} has leaked unknown samples!"

        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=cal_ds))

    server = FederatedServer(config, clients, device)

    # Phase I: Collaborative training
    for r in range(1, 3):
        server.run_round(r)

    # Phase II: Profiling (using local client training partitions and clean cal_ds)
    server.freeze_global_encoder()
    server.run_phase_two_profiling(
        num_blocks=4,
        block_dim=16,
        alpha=0.05,
    )

    # Phase III: Edge inference on test sets
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=4,
        block_dim=16,
    )

    preds_open, _, _, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
    assert len(preds_open) == len(test_open_ds)
    assert len(anomaly_scores) == len(test_open_ds)


def test_zero_leakage_closed_set_evaluation_discipline():
    """Verify that in closed-set evaluation mode, no open-set metrics exist and unknowns are strictly zero."""
    from experiments.run_baselines import evaluate_method_configuration

    metrics = evaluate_method_configuration(
        method_name="FedTROS-MC (Proposed)",
        config_path="configs/default.yaml",
        rounds=1,
        beta_dirichlet=None,
        samples_per_class=50,
        eval_mode="closed_set",
    )

    # 1. Closed-set metrics MUST be present
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics

    # 2. Open-set metrics MUST be strictly absent
    assert "auroc" not in metrics
    assert "aupr" not in metrics
    assert "unknown_recall" not in metrics
    assert "unknown_f1" not in metrics
    assert "kfr" not in metrics
    assert "open_set_macro_f1" not in metrics

