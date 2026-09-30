"""Export real experiment outputs from D:\\PAPER\\Code into the 31-plot reference contract format."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve

from src.config import load_config
from src.data.base import FlowDataset
from src.data.osr_split import compute_openness, split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.tabular_p2p import load_benchmark_dataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.osr_metrics import compute_osr_metrics
from src.tracking.reproducibility import set_seed
from src.utils.device import get_device


KNOWN_NAMES = ["Normal", "BP", "DoS", "MitM"]


def _run_method_round_curves(
    method_name: str,
    train_ds: FlowDataset,
    cal_ds: FlowDataset,
    test_closed_ds: FlowDataset,
    test_open_ds: FlowDataset,
    client_indices: Dict[int, List[int]],
    rounds: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train FedAvg, FedProx, or FedTROS-MC and record real per-round metrics and client forgetting."""
    set_seed(seed)
    cfg = load_config("configs/default.yaml")
    cfg.federated.num_rounds = rounds
    cfg.federated.num_clients = len(client_indices)
    cfg.model.in_features = int(train_ds.features.shape[1])
    num_known = int(len(np.unique(train_ds.labels.numpy())))
    cfg.data.known_classes = list(range(num_known))

    if method_name in ("FedAvg", "FedProx"):
        cfg.loss.tau_bla = 0.0
        cfg.loss.lambda_vib = 0.0
        cfg.model.stochastic_encoder = False
        cfg.federated.aggregation_strategy = "fedavg"
        if method_name == "FedProx":
            cfg.loss.fedprox_mu = 0.01

    clients = []
    for cid, idxs in client_indices.items():
        idx_t = torch.tensor(idxs, dtype=torch.long)
        sub_ds = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(cid, sub_ds, cfg, device, cal_dataset=cal_ds))

    server = FederatedServer(cfg, clients, device)
    num_params = sum(p.numel() for p in server.global_model.parameters())
    mb_per_round = (2.0 * num_params * 4.0) / (1024.0 * 1024.0)
    if method_name == "FedTROS-MC":
        # Decoupled encoder-only downlink/uplink savings during Phase I
        enc_params = sum(p.numel() for p in server.global_model.encoder.parameters())
        mb_per_round = (2.0 * enc_params * 4.0) / (1024.0 * 1024.0)

    y_closed = test_closed_ds.labels.numpy()
    curve_rows = []
    comm_rows = []
    dyn_rows = []

    for r in range(1, rounds + 1):
        loss_info = server.run_round(r)
        server.global_model.eval()
        with torch.no_grad():
            logits = server.global_model.classify(test_closed_ds.features.to(device))
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        acc = float(np.mean(preds == y_closed) * 100.0)
        mf1 = float(f1_score(y_closed, preds, average="macro", zero_division=0) * 100.0)
        # Client-level F1 dispersion for band
        c_f1s = []
        for cl in clients:
            with torch.no_grad():
                cl_logits = cl.model.classify(test_closed_ds.features.to(device))
                cl_preds = torch.argmax(cl_logits, dim=-1).cpu().numpy()
                c_f1s.append(float(f1_score(y_closed, cl_preds, average="macro", zero_division=0) * 100.0))
        band = float(np.std(c_f1s))
        curve_rows.append({"round": r, "method": method_name, "macro_f1_percent": mf1, "band": band})
        comm_rows.append({
            "round": r,
            "method": method_name,
            "cumulative_mb": float(r * mb_per_round),
            "accuracy_percent": acc,
            "run_id": f"real_{method_name.lower()}_s{seed}",
            "seed": seed,
        })
        dyn_rows.append({
            "round": r,
            "loss_bla": float(loss_info.get("loss_bla", loss_info.get("loss_total", 0.0))),
            "loss_vib": float(loss_info.get("loss_vib", 0.0)),
            "disagreement": float(loss_info.get("loss_bla", loss_info.get("loss_total", 0.0))),
            "temperature": float(loss_info.get("lr_scheduled", 0.001) * 1000.0),
        })

    # Present vs absent/scarce class accuracy across clients after final round
    pres_accs, abs_accs = [], []
    for cl in clients:
        counts = np.asarray(cl.samples_per_class, dtype=float)
        props = counts / max(float(counts.sum()), 1.0)
        with torch.no_grad():
            cl_logits = cl.model.classify(test_closed_ds.features.to(device))
            cl_preds = torch.argmax(cl_logits, dim=-1).cpu().numpy()
        for c_id in range(num_known):
            c_mask = (y_closed == c_id)
            if not np.any(c_mask):
                continue
            c_acc = float(np.mean(cl_preds[c_mask] == y_closed[c_mask]) * 100.0)
            if props[c_id] > 0.15:
                pres_accs.append(c_acc)
            else:
                abs_accs.append(c_acc)

    # Phase II & III OSR evaluation
    server.run_phase_two_profiling(
        num_blocks=cfg.federated.num_blocks,
        block_dim=cfg.federated.block_dim,
        alpha=cfg.osr.conformal_alpha,
        cal_dataset=cal_ds,
    )
    detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=cfg.federated.num_blocks,
        block_dim=cfg.federated.block_dim,
    )
    det = detector.predict_batch_detailed(test_open_ds.features)
    y_open = test_open_ds.labels.numpy()
    osr_m = compute_osr_metrics(
        y_true=y_open,
        y_pred=det["predictions"],
        anomaly_scores=det["normalized_scores"],
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    return {
        "curve_rows": curve_rows,
        "comm_rows": comm_rows,
        "dyn_rows": dyn_rows,
        "final_closed_acc": comm_rows[-1]["accuracy_percent"],
        "final_closed_f1": curve_rows[-1]["macro_f1_percent"],
        "present_acc": float(np.mean(pres_accs)) if pres_accs else comm_rows[-1]["accuracy_percent"],
        "absent_acc": float(np.mean(abs_accs)) if abs_accs else curve_rows[-1]["macro_f1_percent"],
        "osr_metrics": osr_m,
        "detailed": det,
        "y_open": y_open,
        "server": server,
    }
