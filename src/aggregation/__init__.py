"""Server-side aggregation algorithms strictly matching LaTeX Section 4.1 & 4.2.

Contains:
    - aggregate_fedavg_parameters (Phase I, Eq. fed_encoder_agg)
    - ClientClassMoment, GlobalClassProfile, aggregate_total_variance_moments (Phase II Law of Total Variance, Theorem 1)
"""

from src.aggregation.fedavg_agg import aggregate_fedavg_parameters
from src.aggregation.total_variance import (
    ClientClassMoment,
    GlobalClassProfile,
    aggregate_total_variance_moments,
)

__all__ = [
    "aggregate_fedavg_parameters",
    "ClientClassMoment",
    "GlobalClassProfile",
    "aggregate_total_variance_moments",
]
