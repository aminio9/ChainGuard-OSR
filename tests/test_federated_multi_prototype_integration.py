"""End-to-end integration test for Federated Multi-Prototype Hyperspherical Profiling (AMP-OAS)."""

import math
import torch
from src.config import AppConfig
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.osr.block_mahalanobis import WireSpeedEdgeDetector


def test_federated_multi_prototype_end_to_end():
    device = torch.device("cpu")

    # Lightweight configuration
    cfg = AppConfig()
    cfg.federated.num_clients = 3
    cfg.federated.client_fraction = 1.0
    cfg.federated.num_rounds = 2
    cfg.federated.local_epochs = 1
    cfg.federated.batch_size = 16
    cfg.data.num_samples = 300
    cfg.data.known_classes = [0, 1, 2]
    cfg.data.unknown_classes = [3]
    cfg.data.in_features = 12
    cfg.model.in_features = 12
    cfg.model.latent_dim = 16
    cfg.model.hidden_dim = [32, 16]
    cfg.experiment.seed = 42

    # Enable Multi-Prototype Profiling
    cfg.osr.use_multi_prototype = True
    cfg.osr.num_prototypes_normal = 5
    cfg.osr.num_prototypes_attack = 3
    cfg.osr.num_blocks = 1
    cfg.osr.block_dim = 16
    cfg.osr.normalize_features = True

    dataset = generate_synthetic_p2p_data(
        num_samples=300,
        in_features=cfg.data.in_features,
        num_classes=4,
        seed=42,
    )

    # Filter to known classes only for training
    known_mask = dataset.labels < 3
    train_feat = dataset.features[known_mask]
    train_labels = dataset.labels[known_mask]
    train_dataset = FlowDataset(train_feat, train_labels)

    client_map = partition_dirichlet_non_iid(
        dataset=train_dataset,
        num_clients=cfg.federated.num_clients,
        beta_dirichlet=0.5,
        seed=42,
    )

    clients = []
    for cid in range(cfg.federated.num_clients):
        sub_idx = client_map[cid]
        idx_t = torch.tensor(sub_idx, dtype=torch.long)
        sub_ds = FlowDataset(train_dataset.features[idx_t], train_dataset.labels[idx_t])
        client = FederatedClient(cid, sub_ds, cfg, device, cal_dataset=sub_ds)
        clients.append(client)

    server = FederatedServer(cfg, clients, device)

    # Stage 1: Representation Learning
    for r in range(1, 3):
        metrics = server.run_round(r)
        assert "loss_total" in metrics

    server.freeze_global_encoder()
    assert server.encoder_frozen

    # Stage 2 & 3: Multi-Prototype Profiling & Conformal Calibration
    profiles = server.run_phase_two_profiling(
        num_blocks=1,
        block_dim=16,
        alpha=0.05,
    )

    assert len(profiles) == len(cfg.data.known_classes)

    for c, prof in profiles.items():
        assert prof.conformal_threshold > 0.0
        assert prof.prototypes is not None
        assert prof.prototypes.ndim == 2
        assert prof.prototypes.size(1) == 16
        target_m = cfg.osr.num_prototypes_normal if c == 0 else cfg.osr.num_prototypes_attack
        assert prof.prototypes.size(0) <= target_m
        print(f"Class {c}: {prof.prototypes.size(0)} prototypes, Conformal tau={prof.conformal_threshold:.4f}")

    # Stage 4: Test Edge Detector
    detector = server.get_edge_detector(num_blocks=1, block_dim=16, normalize_features=True)
    assert isinstance(detector, WireSpeedEdgeDetector)

    # Known sample inference
    x_known = train_dataset.features[:1]
    res_known = detector.predict_sample(x_known)
    assert res_known.candidate_class in cfg.data.known_classes

    # Unknown sample inference (class 3)
    unknown_mask = dataset.labels == 3
    x_unknown = dataset.features[unknown_mask][:5]
    preds, is_unk, cand, scores = detector.predict_batch(x_unknown)
    assert len(preds) == 5

    print("FEDERATED MULTI-PROTOTYPE INTEGRATION TEST PASSED!")


if __name__ == "__main__":
    test_federated_multi_prototype_end_to_end()
