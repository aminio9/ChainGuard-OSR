"""Tests for model architectures: VIB stochastic encoder and decoupled heads."""

import torch
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead, DensityHead
from src.models.dual_branch import DecoupledDualBranchNetwork


def test_vib_encoder_training_vs_eval():
    encoder = VIBEncoder(in_features=10, hidden_dim=20, latent_dim=8, dropout=0.0, use_batchnorm=False)
    x = torch.randn(4, 10)

    # 1. Training mode: stochastic reparameterization
    encoder.train()
    z1, mu1, logvar1 = encoder(x)
    z2, mu2, logvar2 = encoder(x)
    # Mu must be deterministic given x
    assert torch.allclose(mu1, mu2, atol=1e-5)
    # But z should differ due to Gaussian noise epsilon
    assert not torch.allclose(z1, z2, atol=1e-5)

    # 2. Evaluation mode: deterministic z == mu
    encoder.eval()
    z_eval, mu_eval, _ = encoder(x)
    assert torch.allclose(z_eval, mu_eval, atol=1e-5)


def test_deterministic_encoder():
    encoder = DeterministicEncoder(in_features=10, hidden_dim=20, latent_dim=8)
    x = torch.randn(4, 10)
    z, mu, logvar = encoder(x)
    assert z.shape == (4, 8)
    assert torch.allclose(z, mu)


def test_classification_and_density_heads():
    clf = ClassificationHead(latent_dim=8, num_classes=3)
    density = DensityHead(latent_dim=8, density_dim=4)

    z = torch.randn(5, 8)
    logits = clf(z)
    d_out = density(z)

    assert logits.shape == (5, 3)
    assert d_out.shape == (5, 4)
    assert clf.weight_matrix.shape == (3, 8)


def test_decoupled_network_parameter_sync(decoupled_model):
    x = torch.randn(4, decoupled_model.encoder.network[0].in_features)
    logits, d_out, z, mu, logvar = decoupled_model(x)

    assert logits.shape == (4, decoupled_model.num_classes)
    assert z.shape == (4, decoupled_model.latent_dim)

    # Test extracting and setting encoder parameters
    encoder_params = decoupled_model.get_encoder_parameters()
    assert len(encoder_params) > 0

    # Modify parameters and restore
    new_model = DecoupledDualBranchNetwork(
        in_features=decoupled_model.encoder.network[0].in_features,
        latent_dim=decoupled_model.latent_dim,
        num_classes=decoupled_model.num_classes,
        hidden_dim=decoupled_model.encoder.network[0].out_features,
        density_dim=decoupled_model.density_dim,
    )
    new_model.set_encoder_parameters(encoder_params)

    for k in encoder_params.keys():
        assert torch.allclose(
            new_model.encoder.state_dict()[k],
            decoupled_model.encoder.state_dict()[k],
        )
