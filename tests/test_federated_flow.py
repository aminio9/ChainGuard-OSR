"""Integration test verifying complete federated training round and edge inference."""

import numpy as np
import torch
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.osr.weibull_evt import WeibullEVTFitter
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.edge_inference import EdgeInferenceEngine


def test_federated_end_to_end_flow(sample_config):
    device = torch.device("cpu")
    dataset = generate_synthetic_p2p_data(
        num_samples=150,
        in_features=sample_config.data.in_features,
        num_classes=len(sample_config.data.known_classes),
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
        client = FederatedClient(cid, sub_ds, sample_config, device)
        clients.append(client)

    server = FederatedServer(sample_config, clients, device)

    # Execute 1 training round
    metrics = server.run_round(1)
    assert "loss_total" in metrics

    # Fit EVT Weibull
    weibull = WeibullEVTFitter(tail_size=10)
    weibull.fit(dataset.features, dataset.labels, server.global_prototypes, model=server.global_model)

    # Calibrate Conformal
    calibrator = MondrianConformalCalibrator(alpha=0.05)
    logits, _, _, _, _ = server.global_model(dataset.features)
    probs = torch.softmax(logits, dim=-1)
    calibrator.calibrate(probs, dataset.labels)

    # Execute Edge Inference
    engine = EdgeInferenceEngine(
        model=server.global_model,
        prototypes=server.global_prototypes,
        weibull_models=weibull.weibull_models,
        conformal_calibrator=calibrator,
    )

    test_flow = dataset.features[0]
    pred, c_set, energy, u_prob = engine.predict_flow(test_flow)
    assert isinstance(pred, (int, np.integer)) if 'np' in globals() else isinstance(pred, int)
