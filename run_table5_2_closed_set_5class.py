#!/usr/bin/env python3
"""Run genuine 5-class closed-set evaluation for Chapter 5 Table 5-2 on 100% B-NAT.

Invariants:
- All 5 classes (Normal, BP, DoS, MitM, FoT) are known in the training and closed-set test universe.
- Zero classes withheld as unknown (no open-set sequestration).
- 10 clients, 100 communication rounds per model and per non-IID level.
- 4 distribution conditions: IID, Moderate (beta=0.5), Severe (beta=0.1), Critical (beta=0.05).
- Evaluates:
  1. Proposed Framework (VIB + BLA Logit Adjustment + Decoupled Head-Body)
  2. FedAvg (McMahan et al., 2017)
  3. FedProx (Li et al., 2020 with mu=0.01)
- Strict metric calculation:
  - Accuracy (Acc %)
  - Macro F1 (Macro-F1 %)
  - Balanced Accuracy (B-Acc %)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Ensure Code directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.data.tabular_p2p import load_benchmark_dataset
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.baselines.fedavg.fedavg import FedAvgBaseline
from src.baselines.fedprox.fedprox import FedProxBaseline


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("Table5_2_ClosedSet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def evaluate_closed_set(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
    }


def format_persian_latex_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{نتایج طبقه‌بندی مجموعه‌بسته بر حسب درصد در شرایط توزیع یکنواخت و سطوح مختلف ناهمگنی دیریکله روی مجموعه‌داده B-NAT (هر ۵ رده شناخته‌شده) با ۱۰ گره و ۱۰۰ دور ارتباطی}",
        r"\label{tab:ch5-closed-set-eval}",
        r"\begin{tabularx}{\textwidth}{P{3.4cm} >{\hsize=1.36\hsize\raggedleft\arraybackslash}X >{\hsize=0.88\hsize\centering\arraybackslash}X >{\hsize=0.88\hsize\centering\arraybackslash}X >{\hsize=0.88\hsize\centering\arraybackslash}X}",
        r"\toprule",
        r"\textbf{شرایط توزیع} & \textbf{مدل ارزیابی‌شده} & \textbf{\lr{Acc}} & \textbf{\lr{Macro-F1}} & \textbf{\lr{B-Acc}} \\",
        r"\midrule",
    ]

    # IID
    iid_res = results.get("IID", {})
    lines.append(r"یکنواخت & خوا و همکاران~\cite{khoa2024blockchain} & \textbf{۹۸/۴۴} & \textbf{۹۶/۱۰} & -- \\")
    if "FedAvg" in iid_res:
        lines.append(f"یکنواخت & \\lr{{FedAvg}} & {iid_res['FedAvg']['acc']:.2f} & {iid_res['FedAvg']['macro_f1']:.2f} & {iid_res['FedAvg']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "FedProx" in iid_res:
        lines.append(f"یکنواخت & \\lr{{FedProx}} & {iid_res['FedProx']['acc']:.2f} & {iid_res['FedProx']['macro_f1']:.2f} & {iid_res['FedProx']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "Proposed" in iid_res:
        lines.append(f"یکنواخت & \\textbf{{چارچوب پیشنهادی}} & {iid_res['Proposed']['acc']:.2f} & {iid_res['Proposed']['macro_f1']:.2f} & \\textbf{{{iid_res['Proposed']['bal_acc']:.2f}}} \\\\".replace(".", "/"))
    lines.append(r"\midrule")

    # Moderate b=0.5
    mod_res = results.get("Moderate_b0.5", {})
    if "FedAvg" in mod_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.5$}} & \\lr{{FedAvg}} & {mod_res['FedAvg']['acc']:.2f} & {mod_res['FedAvg']['macro_f1']:.2f} & {mod_res['FedAvg']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "FedProx" in mod_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.5$}} & \\lr{{FedProx}} & {mod_res['FedProx']['acc']:.2f} & {mod_res['FedProx']['macro_f1']:.2f} & {mod_res['FedProx']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "Proposed" in mod_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.5$}} & \\textbf{{چارچوب پیشنهادی}} & \\textbf{{{mod_res['Proposed']['acc']:.2f}}} & \\textbf{{{mod_res['Proposed']['macro_f1']:.2f}}} & \\textbf{{{mod_res['Proposed']['bal_acc']:.2f}}} \\\\".replace(".", "/"))
    lines.append(r"\midrule")

    # Severe b=0.1
    sev_res = results.get("Severe_b0.1", {})
    if "FedAvg" in sev_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.1$}} & \\lr{{FedAvg}} & {sev_res['FedAvg']['acc']:.2f} & {sev_res['FedAvg']['macro_f1']:.2f} & {sev_res['FedAvg']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "FedProx" in sev_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.1$}} & \\lr{{FedProx}} & {sev_res['FedProx']['acc']:.2f} & {sev_res['FedProx']['macro_f1']:.2f} & {sev_res['FedProx']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "Proposed" in sev_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.1$}} & \\textbf{{چارچوب پیشنهادی}} & \\textbf{{{sev_res['Proposed']['acc']:.2f}}} & \\textbf{{{sev_res['Proposed']['macro_f1']:.2f}}} & \\textbf{{{sev_res['Proposed']['bal_acc']:.2f}}} \\\\".replace(".", "/"))
    lines.append(r"\midrule")

    # Critical b=0.05
    crit_res = results.get("Critical_b0.05", {})
    if "FedAvg" in crit_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.05$}} & \\lr{{FedAvg}} & {crit_res['FedAvg']['acc']:.2f} & {crit_res['FedAvg']['macro_f1']:.2f} & {crit_res['FedAvg']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "FedProx" in crit_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.05$}} & \\lr{{FedProx}} & {crit_res['FedProx']['acc']:.2f} & {crit_res['FedProx']['macro_f1']:.2f} & {crit_res['FedProx']['bal_acc']:.2f} \\\\".replace(".", "/"))
    if "Proposed" in crit_res:
        lines.append(f"\\mbox{{ناهمگن با $\\beta=0.05$}} & \\textbf{{چارچوب پیشنهادی}} & \\textbf{{{crit_res['Proposed']['acc']:.2f}}} & \\textbf{{{crit_res['Proposed']['macro_f1']:.2f}}} & \\textbf{{{crit_res['Proposed']['bal_acc']:.2f}}} \\\\".replace(".", "/"))

    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    base_out_dir = Path(r"D:\PAPER\Real_Run\master_chapter5_suite")
    log_dir = base_out_dir / "logs"
    ckpt_dir = base_out_dir / "checkpoints_closed_set"
    table_dir = base_out_dir / "tables"
    for d in [log_dir, ckpt_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / "table5_2_closed_set_5class.log")
    logger.info("================================================================================")
    logger.info("STARTING TABLE 5-2 GENUINE 5-CLASS CLOSED-SET CLASSIFICATION SUITE (100 ROUNDS)")
    logger.info("================================================================================")

    device = torch.device("cpu")
    num_clients = 10
    num_rounds = 100
    batch_size = 512
    seed = 42

    set_seed(seed)

    # 1. Load B-NAT (100% of flows: 210,000)
    logger.info("Loading 100% raw B-NAT dataset from data/raw/b_nat.csv...")
    raw_ds, meta = load_benchmark_dataset(
        dataset_name="b_nat",
        data_dir="data/raw",
        samples_per_class=None,
    )
    logger.info(f"Loaded raw dataset: {len(raw_ds)} total flows across {len(meta['label_to_id'])} classes: {meta['label_to_id']}")

    # 2. Strict 5-class closed-set split (NO UNKNOWN CLASSES)
    all_5_classes = [0, 1, 2, 3, 4]
    logger.info(f"Splitting dataset with all 5 classes known: {all_5_classes} and unknown_classes=[]...")
    train_ds, cal_ds, test_closed_ds, _ = split_open_set_dataset(
        dataset=raw_ds,
        known_classes=all_5_classes,
        unknown_classes=[],
        test_ratio=0.20,
        cal_ratio=0.10,
        seed=seed,
    )
    in_features = int(train_ds.features.size(1))
    logger.info(f"Partitions: Train={len(train_ds)}, Cal={len(cal_ds)}, Test={len(test_closed_ds)} | Features={in_features}")

    y_test_true = test_closed_ds.labels.cpu().numpy()
    class_counts_test = {c: int(np.sum(y_test_true == c)) for c in all_5_classes}
    logger.info(f"Test Set Class Counts: {class_counts_test}")

    # 3. Define the 4 Skew Scenarios
    scenarios = [
        ("IID", 100.0),
        ("Moderate_b0.5", 0.5),
        ("Severe_b0.1", 0.1),
        ("Critical_b0.05", 0.05),
    ]

    models_to_run = ["Proposed", "FedAvg", "FedProx"]
    results_file = base_out_dir / "table_5_2_closed_set_5class_results.json"

    # Load existing results if resuming
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for sc_name, beta in scenarios:
        if sc_name not in all_results:
            all_results[sc_name] = {}

        logger.info("--------------------------------------------------------------------------------")
        logger.info(f">>> Scenario: {sc_name} (Dirichlet beta = {beta})")
        logger.info("--------------------------------------------------------------------------------")

        # Partition training set among 10 clients
        client_map = partition_dirichlet_non_iid(
            dataset=train_ds,
            num_clients=num_clients,
            beta_dirichlet=beta,
            seed=seed,
        )
        client_datasets = [
            FlowDataset(train_ds.features[torch.tensor(client_map[cid])], train_ds.labels[torch.tensor(client_map[cid])])
            for cid in range(num_clients)
        ]
        for cid, cds in enumerate(client_datasets):
            logger.info(f"  Client {cid}: {len(cds)} samples | classes present: {sorted(np.unique(cds.labels.numpy()).tolist())}")

        # ---------------------------------------------------------------------
        # MODEL 1: PROPOSED FRAMEWORK (VIB + BLA + Decoupled Head-Body)
        # ---------------------------------------------------------------------
        if "Proposed" not in all_results[sc_name]:
            logger.info(f"[{sc_name}] Training Proposed Framework (VIB + BLA + Decoupled, 100 rounds)...")
            cfg = load_config("configs/default.yaml")
            cfg.data.in_features = in_features
            cfg.data.known_classes = all_5_classes
            cfg.data.unknown_classes = []
            cfg.model.in_features = in_features
            cfg.model.latent_dim = 64
            cfg.model.hidden_dim = [128, 64]
            cfg.federated.num_clients = num_clients
            cfg.federated.num_rounds = num_rounds
            cfg.federated.local_epochs = 1
            cfg.federated.batch_size = batch_size
            cfg.federated.aggregation_strategy = "decoupled"
            cfg.loss.lambda_vib = 0.01
            cfg.loss.tau_bla = 1.0

            clients = [
                FederatedClient(cid, client_datasets[cid], cfg, device)
                for cid in range(num_clients)
            ]
            server = FederatedServer(cfg, clients, device)

            t0 = time.time()
            for r in range(1, num_rounds + 1):
                metrics = server.run_round(r)
                if r % 20 == 0 or r == 1 or r == num_rounds:
                    logger.info(f"  [Proposed | {sc_name}] Round {r:03d}/{num_rounds:03d} | Weighted Loss: {metrics['loss_total']:.4f} (BLA: {metrics.get('loss_bla', 0.0):.4f})")
            t_prop = time.time() - t0

            # Evaluate on 42,000 closed test flows
            server.global_model.eval()
            with torch.no_grad():
                logits, _, _, _ = server.global_model(test_closed_ds.features)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            eval_res = evaluate_closed_set(y_test_true, preds)
            eval_res["train_time_sec"] = t_prop
            all_results[sc_name]["Proposed"] = eval_res
            logger.info(f"  [Proposed | {sc_name}] Finished in {t_prop:.1f}s => Acc: {eval_res['acc']:.2f}% | Macro-F1: {eval_res['macro_f1']:.2f}% | B-Acc: {eval_res['bal_acc']:.2f}%")

            # Save checkpoint
            torch.save(server.global_model.state_dict(), ckpt_dir / f"model_proposed_{sc_name}.pt")

            # Save incremental results
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

        # ---------------------------------------------------------------------
        # MODEL 2: FedAvg (McMahan et al., 2017)
        # ---------------------------------------------------------------------
        if "FedAvg" not in all_results[sc_name]:
            logger.info(f"[{sc_name}] Training FedAvg Baseline (100 rounds)...")
            fedavg = FedAvgBaseline(
                in_features=in_features,
                num_classes=5,
                latent_dim=64,
                hidden_dim=[128, 64],
                device=device,
            )

            t0 = time.time()
            for r in range(1, num_rounds + 1):
                stats = fedavg.fit_round(
                    round_idx=r,
                    client_datasets=client_datasets,
                    epochs=1,
                    batch_size=batch_size,
                    lr=0.001,
                )
                if r % 20 == 0 or r == 1 or r == num_rounds:
                    logger.info(f"  [FedAvg | {sc_name}] Round {r:03d}/{num_rounds:03d} | Train Loss: {stats['train_loss']:.4f}")
            t_avg = time.time() - t0

            fedavg.model.eval()
            with torch.no_grad():
                logits, _ = fedavg.model(test_closed_ds.features)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            eval_res = evaluate_closed_set(y_test_true, preds)
            eval_res["train_time_sec"] = t_avg
            all_results[sc_name]["FedAvg"] = eval_res
            logger.info(f"  [FedAvg | {sc_name}] Finished in {t_avg:.1f}s => Acc: {eval_res['acc']:.2f}% | Macro-F1: {eval_res['macro_f1']:.2f}% | B-Acc: {eval_res['bal_acc']:.2f}%")

            torch.save(fedavg.model.state_dict(), ckpt_dir / f"model_fedavg_{sc_name}.pt")

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

        # ---------------------------------------------------------------------
        # MODEL 3: FedProx (Li et al., 2020)
        # ---------------------------------------------------------------------
        if "FedProx" not in all_results[sc_name]:
            logger.info(f"[{sc_name}] Training FedProx Baseline (mu=0.01, 100 rounds)...")
            fedprox = FedProxBaseline(
                in_features=in_features,
                num_classes=5,
                latent_dim=64,
                hidden_dim=[128, 64],
                mu_prox=0.01,
                device=device,
            )

            t0 = time.time()
            for r in range(1, num_rounds + 1):
                stats = fedprox.fit_round(
                    round_idx=r,
                    client_datasets=client_datasets,
                    epochs=1,
                    batch_size=batch_size,
                    lr=0.001,
                )
                if r % 20 == 0 or r == 1 or r == num_rounds:
                    logger.info(f"  [FedProx | {sc_name}] Round {r:03d}/{num_rounds:03d} | Train Loss: {stats['train_loss']:.4f}")
            t_prx = time.time() - t0

            fedprox.model.eval()
            with torch.no_grad():
                logits, _ = fedprox.model(test_closed_ds.features)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            eval_res = evaluate_closed_set(y_test_true, preds)
            eval_res["train_time_sec"] = t_prx
            all_results[sc_name]["FedProx"] = eval_res
            logger.info(f"  [FedProx | {sc_name}] Finished in {t_prx:.1f}s => Acc: {eval_res['acc']:.2f}% | Macro-F1: {eval_res['macro_f1']:.2f}% | B-Acc: {eval_res['bal_acc']:.2f}%")

            torch.save(fedprox.model.state_dict(), ckpt_dir / f"model_fedprox_{sc_name}.pt")

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    # 4. Export Formatted LaTeX Table 5-2
    latex_tbl = format_persian_latex_table(all_results)
    table_file = table_dir / "table_5_2_closed_set.tex"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(latex_tbl)
    logger.info(f"Saved updated LaTeX Table 5-2 to: {table_file}")

    # Final summary display
    logger.info("================================================================================")
    logger.info("TABLE 5-2 EXPERIMENTS SUCCESSFULLY COMPLETED ACROSS ALL 5 CLASSES:")
    logger.info("================================================================================")
    for sc_name in ["IID", "Moderate_b0.5", "Severe_b0.1", "Critical_b0.05"]:
        logger.info(f"--- {sc_name} ---")
        for m_name in ["FedAvg", "FedProx", "Proposed"]:
            if m_name in all_results.get(sc_name, {}):
                res = all_results[sc_name][m_name]
                logger.info(f"  {m_name:12s} | Acc: {res['acc']:6.2f}% | Macro-F1: {res['macro_f1']:6.2f}% | B-Acc: {res['bal_acc']:6.2f}%")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
