"""Server-side aggregation algorithms: FedAvg, Byzantine MAD, Law of Total Variance, Bayesian Covariance-Weighted Prototypes, and Dual Updates."""

from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.covariance_proto import aggregate_prototypes_covariance
from src.aggregation.primal_dual_agg import update_dual_multipliers
from src.aggregation.byzantine_mad import (
    ClientClassMoment,
    filter_byzantine_moments,
)
from src.aggregation.total_variance import (
    GlobalClassProfile,
    aggregate_total_variance_moments,
)

__all__ = [
    "aggregate_fedavg_parameters",
    "aggregate_prototypes_covariance",
    "update_dual_multipliers",
    "ClientClassMoment",
    "filter_byzantine_moments",
    "GlobalClassProfile",
    "aggregate_total_variance_moments",
]
