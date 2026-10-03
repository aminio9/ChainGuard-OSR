"""Streaming extraction of legacy experiment archives.

This extractor is retained only for pre-publication-bundle migration; new
publication rendering must use :mod:`src.data.fedtros_bundle`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import zstandard as zstd
import tarfile

EXP2 = "e2_iid_openset_fot_dkd_fedos_seed42"
P01 = "e3_noniid_alpha01_closed_dkd_fedos_seed42"
A01 = "e3_noniid_alpha01_closed_fedavg_seed42"
X01 = "e3_noniid_alpha01_closed_fedprox_seed42"
P10 = "e3_noniid_alpha1_closed_dkd_fedos_seed42"
A10 = "e3_noniid_alpha1_closed_fedavg_seed42"
X10 = "e3_noniid_alpha1_closed_fedprox_seed42"


def member(run: str, name: str) -> str:
    return f"outputs/{run}/{name}"


HOST1_MEMBERS = {
    member(P01, "federated_history.csv"),
    member(P01, "processed/client_class_distribution.csv"),
    member(P01, "resolved_config.yaml"),
    member(X01, "federated_history.csv"),
}

HOST2_MEMBERS = {
    member(EXP2, "resolved_config.yaml"),
    member(EXP2, "dkd_fedos_student_latest.pt"),
    member(EXP2, "processed/shared_open_set_test.pt"),
    member(EXP2, "processed/class_names.json"),
    member(EXP2, "open_set_scores.csv"),
    member(EXP2, "open_set_roc_curve.csv"),
    member(EXP2, "open_set_pr_curve.csv"),
    member(EXP2, "before_osr_confusion_matrix.csv"),
    member(EXP2, "after_osr_confusion_matrix.csv"),
    member(EXP2, "fed_digos_prototypes.json"),
    member(EXP2, "open_set_metrics.json"),
    member(P10, "federated_history.csv"),
    member(P10, "resolved_config.yaml"),
    member(P10, "dkd_fedos_student_latest.pt"),
    member(A10, "federated_history.csv"),
    member(A10, "communication_metrics.csv"),
    member(X10, "federated_history.csv"),
    member(X10, "communication_metrics.csv"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_members(archive: Path, wanted: set[str]) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    found: dict[str, bytes] = {}
    provenance: dict[str, dict[str, Any]] = {}
    with archive.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed, read_size=16 * 1024 * 1024)
        with tarfile.open(fileobj=reader, mode="r|", bufsize=16 * 1024 * 1024) as tar:
            for info in tar:
                if not info.isfile() or info.name not in wanted:
                    continue
                extracted = tar.extractfile(info)
                if extracted is None:
                    continue
                data = extracted.read()
                found[info.name] = data
                provenance[info.name] = {
                    "archive_path": str(archive.resolve()),
                    "archive_sha256": sha256_file(archive),
                    "member_name": info.name,
                    "member_size_bytes": info.size,
                    "member_mtime_epoch": info.mtime,
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                }
                if len(found) == len(wanted):
                    break
    return found, provenance
