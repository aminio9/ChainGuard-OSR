"""Neural network architectures: VIB stochastic encoder and decoupled dual-branch heads."""

from src.models.base import BaseNetwork
from src.models.encoder import VIBEncoder, DeterministicEncoder
from src.models.heads import ClassificationHead, DensityHead
from src.models.dual_branch import DecoupledDualBranchNetwork

__all__ = [
    "BaseNetwork",
    "VIBEncoder",
    "DeterministicEncoder",
    "ClassificationHead",
    "DensityHead",
    "DecoupledDualBranchNetwork",
]
