"""Canonical VIB Feature Encoder and Classifier Architecture (LaTeX Section 4.1, Equations 32 & 41).

Model:
    1. Feature Encoder E_theta: Maps flow x in R^{d_x} to latent parameters mu(x), logvar(x) in R^{d_z}.
       Stochastic latent vector z = mu(x) + epsilon * sigma(x) via reparameterization (Equation 41).
    2. Classifier Head C_phi: Linear projection g_phi(z) to unnormalized class logits in R^{|C_K|} (Equation 32).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.base import BaseNetwork
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead


class VIBClassifierNetwork(BaseNetwork):
    """Canonical representation learning backbone for FedTROS-MC (LaTeX Section 4.1)."""

    def __init__(
        self,
        in_features: int = 34,
        latent_dim: int = 64,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
        use_layernorm: bool = True,
        activation: str = "gelu",
        stochastic_encoder: bool = True,
        **kwargs,  # Gracefully ignore any legacy kwargs like density_dim
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.stochastic_encoder = stochastic_encoder

        # 1. Feature Encoder E_theta (Equation 41)
        if stochastic_encoder:
            self.encoder = VIBEncoder(
                in_features=in_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                activation=activation,
            )
        else:
            self.encoder = DeterministicEncoder(
                in_features=in_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                activation=activation,
            )

        # 2. Known-Class Classifier Head C_phi (Equation 32)
        self.classifier = ClassificationHead(latent_dim=latent_dim, num_classes=num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VIB encoder and classification head.

        Returns:
            (logits, z, mu, logvar)
        """
        z, mu, logvar = self.encoder(x)
        logits = self.classifier(z)
        return logits, z, mu, logvar

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic evaluation: compute logits directly from mean latent embedding."""
        if hasattr(self.encoder, "deterministic_forward"):
            mu = self.encoder.deterministic_forward(x)
        else:
            out = self.encoder(x)
            mu = out[1] if isinstance(out, tuple) else out
        return self.classifier(mu)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic evaluation: extract mean embedding vector z = mu(x; theta)."""
        if hasattr(self.encoder, "deterministic_forward"):
            return self.encoder.deterministic_forward(x)
        out = self.encoder(x)
        return out[1] if isinstance(out, tuple) else out

    def get_encoder_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract encoder state dictionary for federated parameter exchange."""
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def set_encoder_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated global encoder parameters."""
        self.encoder.load_state_dict(state_dict)

    def get_classifier_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract classifier head state dictionary for federated parameter exchange."""
        return {k: v.clone() for k, v in self.classifier.state_dict().items()}

    def set_classifier_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated classifier head parameters."""
        self.classifier.load_state_dict(state_dict)

    def get_all_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract full model state dictionary for federated parameter exchange."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_all_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated global model parameters."""
        self.load_state_dict(state_dict)
