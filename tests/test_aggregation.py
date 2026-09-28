"""Tests for server-side parameter, prototype, and dual updates."""

import torch
from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.covariance_proto import aggregate_prototypes_covariance
from src.aggregation.primal_dual_agg import update_dual_multipliers, flatten_state_dict


def test_fedavg_parameter_aggregation():
    p1 = {"w": torch.tensor([1.0, 1.0])}
    p2 = {"w": torch.tensor([3.0, 3.0])}

    # Weight 1: 10 samples, Weight 2: 30 samples -> Total 40 samples
    # Expected w = (10*1 + 30*3) / 40 = 100 / 40 = 2.5
    client_data = [(p1, 10), (p2, 30)]
    global_p = aggregate_fedavg_parameters(client_data)

    expected = torch.tensor([2.5, 2.5])
    assert torch.allclose(global_p["w"], expected)


def test_covariance_prototype_aggregation():
    latent_dim = 4
    # Client 1: confident, small variance -> large inverse covariance weight
    mu1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    cov1 = 0.01 * torch.eye(latent_dim)

    # Client 2: noisy, large variance -> small inverse covariance weight
    mu2 = torch.tensor([5.0, 5.0, 5.0, 5.0])
    cov2 = 1.0 * torch.eye(latent_dim)

    submissions = {0: [(mu1, cov1, 100), (mu2, cov2, 100)]}
    global_protos = aggregate_prototypes_covariance(submissions, latent_dim=latent_dim)

    # The synthesized prototype should be pulled strongly toward mu1
    proto_c0 = global_protos[0]
    dist_to_mu1 = torch.norm(proto_c0 - mu1).item()
    dist_to_mu2 = torch.norm(proto_c0 - mu2).item()
    assert dist_to_mu1 < dist_to_mu2


def test_primal_dual_multiplier_update():
    global_dict = {"w": torch.tensor([2.0, 2.0])}
    client_dict = {
        0: {"w": torch.tensor([3.0, 3.0])},  # Client 0 drifted +1.0
        1: {"w": torch.tensor([1.0, 1.0])},  # Client 1 drifted -1.0
    }

    current_lambdas = {
        0: torch.zeros(2),
        1: torch.zeros(2),
    }

    # Step: lambda = lambda + rho * (theta_k - theta_g)
    # rho = 0.1 -> new_lambda_0 = [0.1, 0.1], new_lambda_1 = [-0.1, -0.1]
    new_lambdas = update_dual_multipliers(
        current_lambdas=current_lambdas,
        client_params=client_dict,
        global_params=global_dict,
        rho=0.1,
    )

    assert torch.allclose(new_lambdas[0], torch.tensor([0.1, 0.1]))
    assert torch.allclose(new_lambdas[1], torch.tensor([-0.1, -0.1]))
