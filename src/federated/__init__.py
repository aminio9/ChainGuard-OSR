"""Federated learning framework: Edge Client and Coordinator Server."""

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

__all__ = [
    "FederatedClient",
    "FederatedServer",
]
