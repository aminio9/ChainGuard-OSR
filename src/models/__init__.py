"""Neural network model architectures."""

from __future__ import annotations

from src.models.base import BaseNetwork
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead
from src.models.vib_network import VIBClassifierNetwork

__all__ = [
    "BaseNetwork",
    "VIBEncoder",
    "DeterministicEncoder",
    "ClassificationHead",
    "VIBClassifierNetwork",
]
