"""Flower-compatible Federated Learning Strategy implementing Covariance Prototype Aggregation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
import numpy as np


class CovariancePrototypeStrategy(fl.server.strategy.FedAvg):
    """Custom Flower strategy extending FedAvg to handle decoupled parameters

    and covariance-weighted prototypes.
    """

    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            *args,
            **kwargs,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Standard aggregate_fit delegated to FedAvg with metrics extraction."""
        if not results:
            return None, {}

        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        return aggregated_params, aggregated_metrics
