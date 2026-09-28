"""Federated learning framework: Client, Server, Strategy, and Flower integrations."""

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.strategy import CovariancePrototypeStrategy

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "CovariancePrototypeStrategy",
]
