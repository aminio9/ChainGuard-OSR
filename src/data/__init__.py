"""Data pipelines, synthetic generators, Dirichlet non-IID partitioning, and OSR splitting."""

from src.data.base import FlowDataset
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.tabular_p2p import load_p2p_tabular_dataset, load_benchmark_dataset, BENCHMARK_METADATA
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.osr_split import split_open_set_dataset, compute_openness

__all__ = [
    "FlowDataset",
    "generate_synthetic_p2p_data",
    "load_p2p_tabular_dataset",
    "load_benchmark_dataset",
    "BENCHMARK_METADATA",
    "partition_dirichlet_non_iid",
    "split_open_set_dataset",
    "compute_openness",
]
