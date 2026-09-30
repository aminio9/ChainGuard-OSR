"""Federated learning framework: Client, Server, and Coordinator (LaTeX Algorithm 1)."""

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

__all__ = [
    "FederatedClient",
    "FederatedServer",
]
