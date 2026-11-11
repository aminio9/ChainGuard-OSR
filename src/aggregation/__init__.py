"""Server-side aggregation algorithms.

Contains:
    - aggregate_fedavg_parameters: Standard sample-weighted parameter averaging.
    - aggregate_decoupled_parameters: Decoupled parameter aggregation for encoder and classifier.
    - ClientClassMoment, GlobalClassProfile, aggregate_total_variance_moments:
      Statistical moment profiling via the Law of Total Variance.
"""

from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.decoupled_agg import aggregate_decoupled_parameters
from src.aggregation.total_variance import (
    ClientClassMoment,
    GlobalClassProfile,
    aggregate_total_variance_moments,
    compute_server_side_oas_shrinkage,
    extract_one_shot_federated_reference_centers,
)

__all__ = [
    "aggregate_fedavg_parameters",
    "aggregate_decoupled_parameters",
    "ClientClassMoment",
    "GlobalClassProfile",
    "aggregate_total_variance_moments",
    "compute_server_side_oas_shrinkage",
    "extract_one_shot_federated_reference_centers",
]
