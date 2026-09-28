"""Decoupled dual-branch composite neural architecture (Formula 3 & Algorithm 1)."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.models.base import BaseNetwork
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead, DensityHead


class DecoupledDualBranchNetwork(BaseNetwork):
    """Composite network decoupling feature extraction, known-class classification,

    and density/uncertainty estimation.

    Mathematical Formulation (Formula 3):
        z = E_theta(x)
        y_hat = C_phi(z) in R^C
        h = H_psi(z) in R^{d_h}
    """

    def __init__(
        self,
        in_features: int = 43,
        latent_dim: int = 64,
        num_classes: int = 7,
        hidden_dim: int = 128,
        density_dim: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        stochastic_encoder: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.density_dim = density_dim
        self.stochastic_encoder = stochastic_encoder

        # 1. Feature Encoder E_theta
        if stochastic_encoder:
            self.encoder = VIBEncoder(
                in_features=in_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.encoder = DeterministicEncoder(
                in_features=in_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
            )

        # 2. Known-Class Classifier Head C_phi
        self.classifier = ClassificationHead(latent_dim=latent_dim, num_classes=num_classes)

        # 3. Density / Uncertainty Head H_psi
        if density_dim > 0:
            self.density_head = DensityHead(latent_dim=latent_dim, density_dim=density_dim)
        else:
            self.density_head = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through decoupled branches.

        Returns:
            (logits, density_rep, z, mu, logvar)
        """
        z, mu, logvar = self.encoder(x)
        logits = self.classifier(z)
        density_rep = self.density_head(z) if self.density_head is not None else None
        return logits, density_rep, z, mu, logvar

    def get_encoder_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract encoder state dictionary for federated parameter exchange."""
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def set_encoder_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated global encoder parameters."""
        self.encoder.load_state_dict(state_dict)
