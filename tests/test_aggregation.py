"""Tests for server-side parameter aggregation and Law of Total Variance moment aggregation."""

import torch
from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.total_variance import ClientClassMoment, aggregate_total_variance_moments


def test_fedavg_parameter_aggregation():
    p1 = {"w": torch.tensor([1.0, 1.0])}
    p2 = {"w": torch.tensor([3.0, 3.0])}

    # Weight 1: 10 samples, Weight 2: 30 samples -> Total 40 samples
    # Expected w = (10*1 + 30*3) / 40 = 100 / 40 = 2.5
    client_data = [(p1, 10), (p2, 30)]
    global_p = aggregate_fedavg_parameters(client_data)

    expected = torch.tensor([2.5, 2.5])
    assert torch.allclose(global_p["w"], expected)


def test_total_variance_aggregation():
    # Verify that pooled covariance matches Law of Total Variance
    m1 = ClientClassMoment(
        client_id=0,
        sample_count=50,
        mu=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        block_covs=[torch.eye(2), torch.eye(2)],
        cal_quantile=1.5,
    )
    m2 = ClientClassMoment(
        client_id=1,
        sample_count=50,
        mu=torch.tensor([3.0, 3.0, 3.0, 3.0]),
        block_covs=[torch.eye(2), torch.eye(2)],
        cal_quantile=1.8,
    )

    submissions = {0: [m1, m2]}
    profiles = aggregate_total_variance_moments(
        submissions,
        num_blocks=2,
        block_dim=2,
    )

    assert 0 in profiles
    prof = profiles[0]
    assert torch.allclose(prof.mu, torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert len(prof.block_covs) == 2
    assert prof.total_samples == 100


def test_decoupled_parameter_aggregation():
    from src.aggregation.decoupled_agg import aggregate_decoupled_parameters

    # Client 1: 100 samples, observed class 0 only
    p1 = {
        "encoder.weight": torch.tensor([1.0, 1.0]),
        "classifier.linear.weight": torch.tensor([[10.0, 10.0], [0.0, 0.0]]),
        "classifier.linear.bias": torch.tensor([1.0, 0.0]),
    }
    # Client 2: 100 samples, observed class 1 only
    p2 = {
        "encoder.weight": torch.tensor([3.0, 3.0]),
        "classifier.linear.weight": torch.tensor([[0.0, 0.0], [20.0, 20.0]]),
        "classifier.linear.bias": torch.tensor([0.0, 2.0]),
    }

    client_updates = [
        (p1, 100, [100, 0]),
        (p2, 100, [0, 100]),
    ]

    global_w = aggregate_decoupled_parameters(client_updates)

    # Encoder should be sample-weighted average: (1 + 3) / 2 = 2.0
    assert torch.allclose(global_w["encoder.weight"], torch.tensor([2.0, 2.0]))
    # Classifier row 0 should be from Client 1 only: [10.0, 10.0], bias: 1.0
    assert torch.allclose(global_w["classifier.linear.weight"][0], torch.tensor([10.0, 10.0]))
    assert torch.isclose(global_w["classifier.linear.bias"][0], torch.tensor(1.0))
    # Classifier row 1 should be from Client 2 only: [20.0, 20.0], bias: 2.0
    assert torch.allclose(global_w["classifier.linear.weight"][1], torch.tensor([20.0, 20.0]))
    assert torch.isclose(global_w["classifier.linear.bias"][1], torch.tensor(2.0))

