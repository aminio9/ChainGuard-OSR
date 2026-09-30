"""Integration tests verifying complete federated training and 3-Phase protocol execution (LaTeX Algorithms 1 & 2)."""

import numpy as np
import torch
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


def test_three_phase_protocol_end_to_end(sample_config):
    """Verify complete 3-Phase protocol from LaTeX Algorithms 1 & 2."""
    device = torch.device("cpu")
    num_classes = len(sample_config.data.known_classes)

    dataset = generate_synthetic_p2p_data(
        num_samples=200,
        in_features=sample_config.data.in_features,
        num_classes=num_classes,
        seed=42,
    )

    client_map = partition_dirichlet_non_iid(
        dataset=dataset,
        num_clients=sample_config.federated.num_clients,
        beta_dirichlet=0.5,
        seed=42,
    )

    clients = []
    for cid in range(sample_config.federated.num_clients):
        sub_idx = client_map[cid]
        idx_t = torch.tensor(sub_idx, dtype=torch.long)
        sub_ds = FlowDataset(dataset.features[idx_t], dataset.labels[idx_t])
        client = FederatedClient(cid, sub_ds, sample_config, device, cal_dataset=sub_ds)
        clients.append(client)

    server = FederatedServer(sample_config, clients, device)

    # =========================================================================
    # Phase I: Communication rounds with Cosine Annealing & Dynamic Anchor
    # =========================================================================
    for r in range(1, 3):
        metrics = server.run_round(r)
        assert "loss_total" in metrics
        assert "loss_bla" in metrics
        assert "loss_vib" in metrics
        assert "lr_scheduled" in metrics

    # Freeze encoder (Algorithm 1 Line 297)
    server.freeze_global_encoder()
    assert server.encoder_frozen

    # =========================================================================
    # Phase II: Moment Extraction & Total Variance Aggregation (Theorem 1)
    # =========================================================================
    profiles = server.run_phase_two_profiling(
        num_blocks=4,
        block_dim=4,  # sample_config.model.latent_dim = 16 = 4 * 4
        alpha=0.05,
    )
    assert len(profiles) > 0
    for c, p in profiles.items():
        assert p.mu.shape == (sample_config.model.latent_dim,)
        assert len(p.block_covs) == 4
        assert p.conformal_threshold >= 0.0

    # =========================================================================
    # Phase III: Wire-Speed Conformal Edge Detection (Algorithm 2)
    # =========================================================================
    detector = server.get_edge_detector(
        unknown_label=-1,
        num_blocks=4,
        block_dim=4,
    )

    # Single sample inference
    res = detector.predict_sample(dataset.features[0])
    assert res.candidate_class in profiles
    assert res.nonconformity_score >= 0.0
    assert isinstance(res.is_unknown, (bool, np.bool_))

    # Batch inference
    preds, is_unk, cands, scores = detector.predict_batch(dataset.features[:10])
    assert len(preds) == 10
    assert len(is_unk) == 10
    assert len(cands) == 10
    assert len(scores) == 10

    # Verify profiles were broadcast back to each client ("دوباره محلی") (LaTeX Algorithm 2)
    for client in clients:
        assert client.global_profiles is not None
        assert client.edge_detector is not None
        c_preds, c_is_unk, _, _ = client.evaluate_edge_traffic(dataset.features[:5])
        assert len(c_preds) == 5
        assert len(c_is_unk) == 5

