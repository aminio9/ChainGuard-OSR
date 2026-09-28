"""Server-side aggregation algorithms: FedAvg, Bayesian Covariance-Weighted Prototypes, and Dual Updates."""

from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.covariance_proto import aggregate_prototypes_covariance
from src.aggregation.primal_dual_agg import update_dual_multipliers

__all__ = [
    "aggregate_fedavg_parameters",
    "aggregate_prototypes_covariance",
    "update_dual_multipliers",
]
