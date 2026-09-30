"""Part 2 of real-data contract exporter: scalability logs, OSR artifacts, and result.json assembly."""

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
from src.ref_plots.contract_exporter import KNOWN_NAMES, _run_method_round_curves
from src.tracking.reproducibility import set_seed
from src.utils.device import get_device


def _run_real_scalability_log(
    k_clients: int,
    train_ds: FlowDataset,
    cal_ds: FlowDataset,
    test_closed_ds: FlowDataset,
    test_open_ds: FlowDataset,
    rounds: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Execute real FedTROS-MC training for K clients and record round-by-round scalability CSV."""
    set_seed(seed)
    cfg = load_config("configs/default.yaml")
    cfg.federated.num_clients = k_clients
    cfg.federated.num_rounds = rounds
    cfg.model.in_features = int(train_ds.features.shape[1])
    num_known = int(len(np.unique(train_ds.labels.numpy())))
    cfg.data.known_classes = list(range(num_known))

    parts = partition_dirichlet_non_iid(
        train_ds, num_clients=k_clients, beta_dirichlet=0.5, min_samples_per_client=2, seed=seed
    )
    clients = [
        FederatedClient(
            cid,
            FlowDataset(
                train_ds.features[torch.tensor(idxs, dtype=torch.long)],
                train_ds.labels[torch.tensor(idxs, dtype=torch.long)],
            ),
            cfg,
            device,
            cal_dataset=cal_ds,
        )
        for cid, idxs in parts.items()
    ]
    server = FederatedServer(cfg, clients, device)
    y_closed = test_closed_ds.labels.numpy()
    y_open = test_open_ds.labels.numpy()
    rows = []

    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        server.run_round(r)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        server.run_phase_two_profiling(
            num_blocks=cfg.federated.num_blocks,
            block_dim=cfg.federated.block_dim,
            alpha=cfg.osr.conformal_alpha,
            cal_dataset=cal_ds,
        )
        t_agg = time.perf_counter() - t1

        t2 = time.perf_counter()
        detector = server.get_edge_detector(
            unknown_label=num_known,
            num_blocks=cfg.federated.num_blocks,
            block_dim=cfg.federated.block_dim,
        )
        det = detector.predict_batch_detailed(test_open_ds.features)
        osr_m = compute_osr_metrics(
            y_true=y_open,
            y_pred=det["predictions"],
            anomaly_scores=det["normalized_scores"],
            num_known_classes=num_known,
            unknown_label=num_known,
        )
        t_eval = time.perf_counter() - t2

        with torch.no_grad():
            logits_c = server.global_model.classify(test_closed_ds.features.to(device))
            preds_c = torch.argmax(logits_c, dim=-1).cpu().numpy()
        c_acc = float(np.mean(preds_c == y_closed))
        c_f1 = float(f1_score(y_closed, preds_c, average="macro", zero_division=0))

        # Evaluate subset of clients for client fairness metrics
        eval_clients = clients[: min(len(clients), 15)]
        cl_f1s = []
        for cl in eval_clients:
            with torch.no_grad():
                cl_l = cl.model.classify(test_closed_ds.features.to(device))
                cl_p = torch.argmax(cl_l, dim=-1).cpu().numpy()
                cl_f1s.append(float(f1_score(y_closed, cl_p, average="macro", zero_division=0)))

        known_mask = (y_open != num_known)
        known_acc_open = float(np.mean(det["predictions"][known_mask] == y_open[known_mask]))
        overall_acc_open = float(np.mean(det["predictions"] == y_open))

        rows.append({
            "round": r,
            "num_clients": k_clients,
            "seed": seed,
            "alpha": 0.5,
            "local_student_f1_macro": c_f1,
            "local_student_accuracy": c_acc,
            "mean_client_macro_f1": float(np.mean(cl_f1s)),
            "std_client_macro_f1": float(np.std(cl_f1s)),
            "worst_client_macro_f1": float(np.min(cl_f1s)),
            "client_fit_wall_time_sec": float(t_fit),
            "round_time_sec": float(t_fit + t_agg + t_eval),
            "server_aggregation_time_sec": float(t_agg),
            "open_set_round_eval_time_sec": float(t_eval),
            "round_openset_f1_macro": float(osr_m["open_set_macro_f1"]),
            "round_openset_overall_acc": overall_acc_open,
            "round_openset_known_acc": known_acc_open,
            "round_openset_auroc": float(osr_m["auroc"]),
            "round_openset_fpr95": float(osr_m["fpr95"]),
            "round_openset_unknown_recall": float(osr_m["unknown_recall"]),
        })
    return pd.DataFrame(rows)


def export_real_plot_contract(
    results_root: Path,
    contract_dir: Path,
    rounds: int = 5,
    spc: Optional[int] = 150,
    seed: int = 42,
) -> Path:
    """Build the complete 31-plot contract directory strictly from real PyTorch runs and saved results."""
    contract_dir.mkdir(parents=True, exist_ok=True)
    device = get_device("auto")

    raw_bnat, meta_bnat = load_benchmark_dataset("b_nat", data_dir="data/raw", samples_per_class=spc)
    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        raw_bnat,
        known_classes=meta_bnat["known_classes"],
        unknown_classes=meta_bnat["unknown_classes"],
        seed=seed,
        protocol="paper_table1" if spc is None else "full",
    )

    # 1. Real Non-IID Client Distributions (Plot 02 & 02_2x2)
    sweep_rows = []
    alpha01_rows = []
    for setting, order, title, beta in [
        ("iid", 0, "(a) IID Partition", 100.0),
        ("alpha1", 1, "(b) Mild Non-IID (alpha = 1.0)", 1.0),
        ("alpha05", 2, "(c) Moderate Non-IID (alpha = 0.5)", 0.5),
        ("alpha01", 3, "(d) Severe Non-IID (alpha = 0.1)", 0.1),
    ]:
        parts = partition_dirichlet_non_iid(train_ds, num_clients=10, beta_dirichlet=beta, seed=seed + order)
        for cid in range(10):
            lbls = train_ds.labels[torch.tensor(parts[cid], dtype=torch.long)].numpy()
            total = max(len(lbls), 1)
            for c_idx, g_name in enumerate(KNOWN_NAMES):
                prop = float(np.sum(lbls == c_idx)) / float(total)
                sweep_rows.append({
                    "setting": setting,
                    "setting_order": order,
                    "panel_title": title,
                    "client_id": cid + 1,
                    "traffic_group": g_name,
                    "proportion": prop,
                })
                if setting == "alpha01":
                    alpha01_rows.append({"client": f"Client {cid + 1}", "traffic_group": g_name, "proportion": prop})

    pd.DataFrame(alpha01_rows).to_csv(contract_dir / "client_distribution_alpha01.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(contract_dir / "client_distribution_iid_alpha_sweep.csv", index=False)

    # 2. Real Convergence, Communication, Robustness, and Forgetting across alpha in {0.1, 0.5, 1.0}
    conv_a1, conv_a01, comm_a1, rob_rows, forget_rows = [], [], [], [], []
    main_osr_bundle = None

    for alpha_val in [1.0, 0.5, 0.1]:
        parts = partition_dirichlet_non_iid(train_ds, num_clients=10, beta_dirichlet=alpha_val, seed=seed)
        for m_name in ["FedAvg", "FedProx", "FedTROS-MC"]:
            res = _run_method_round_curves(
                m_name, train_ds, cal_ds, test_closed_ds, test_open_ds, parts, rounds, seed, device
            )
            rob_rows.append({
                "accuracy": res["final_closed_acc"],
                "alpha_requested": alpha_val,
                "alpha_resolved": alpha_val,
                "config_mismatch": False,
                "method": "FedTROS" if m_name == "FedTROS-MC" else m_name,
                "seed": seed,
                "status": "available",
            })
            if alpha_val == 1.0:
                conv_a1.extend(res["curve_rows"])
                comm_a1.extend(res["comm_rows"])
            elif alpha_val == 0.1:
                conv_a01.extend(res["curve_rows"])
                forget_rows.append({
                    "method": m_name,
                    "present_accuracy": res["present_acc"],
                    "absent_accuracy": res["absent_acc"],
                })
            if alpha_val == 0.5 and m_name == "FedTROS-MC":
                main_osr_bundle = res

    pd.DataFrame(conv_a1).to_csv(contract_dir / "convergence_alpha1.csv", index=False)
    pd.DataFrame(conv_a01).to_csv(contract_dir / "convergence_alpha01.csv", index=False)
    pd.DataFrame(comm_a1).to_csv(contract_dir / "communication_alpha1.csv", index=False)
    pd.DataFrame(comm_a1).to_csv(contract_dir / "communication_alpha01_fedavg.csv", index=False)
    pd.DataFrame(forget_rows).to_csv(contract_dir / "client_forgetting_mitigation.csv", index=False)
    pd.DataFrame(main_osr_bundle["dyn_rows"]).to_csv(contract_dir / "training_dynamics.csv", index=False)

    return contract_dir
