"""Tests for model architectures: VIB stochastic encoder and classification head (LaTeX Section 4.1)."""

import torch
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead
from src.models.vib_network import VIBClassifierNetwork


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


def test_classification_head():
    clf = ClassificationHead(latent_dim=8, num_classes=3)
    z = torch.randn(5, 8)
    logits = clf(z)

    assert logits.shape == (5, 3)
    assert clf.weight_matrix.shape == (3, 8)


def test_vib_classifier_network(decoupled_model):
    x = torch.randn(4, decoupled_model.in_features)
    logits, z, mu, logvar = decoupled_model(x)

    assert logits.shape == (4, decoupled_model.num_classes)
    assert z.shape == (4, decoupled_model.latent_dim)
    assert mu.shape == (4, decoupled_model.latent_dim)

    # Test deterministic classification and feature extraction
    logits_det = decoupled_model.classify(x)
    assert logits_det.shape == (4, decoupled_model.num_classes)

    feat_det = decoupled_model.extract_features(x)
    assert feat_det.shape == (4, decoupled_model.latent_dim)

    # Test extracting and setting encoder parameters
    encoder_params = decoupled_model.get_encoder_parameters()
    assert len(encoder_params) > 0

    new_model = VIBClassifierNetwork(
        in_features=decoupled_model.in_features,
        latent_dim=decoupled_model.latent_dim,
        num_classes=decoupled_model.num_classes,
        hidden_dim=32,
    )
    new_model.set_encoder_parameters(encoder_params)

    for k in encoder_params.keys():
        assert torch.allclose(
            new_model.encoder.state_dict()[k],
            decoupled_model.encoder.state_dict()[k],
        )
