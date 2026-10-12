"""Benchmark evaluation suite comparing canonical federated baselines against proposed FedTROS-MC.

Aligned with LaTeX Section 5.3 & Section 6.2 (Tables 4 & 5).
Baselines:
  - FedAvg (McMahan et al., 2017): Coordinate-wise averaging, empirical cross-entropy, calibrated MSP OSR.
  - FedProx (Li et al., 2020): Proximal parameter-regularized federated learning, Weibull EVT OpenMax OSR.
  - FedProto + Energy: Baseline prototype aggregation with Helmholtz Free Energy OSR.
  - FedTROS-MC (Proposed): 3-Phase protocol with L_BLA, VIB, Decoupled Aggregation,
    Law of Total Variance, and Wire-Speed Block-Diagonal Mahalanobis Edge Detector.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger, format_q1_experiment_log, save_q1_log
from src.utils.device import get_device
from src.utils.io import save_json
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.tabular_p2p import (
    load_p2p_tabular_dataset,
    load_benchmark_dataset,
    BENCHMARK_METADATA,
)
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import (
    compute_confusion_matrix,
    compute_osr_metrics,
    compute_pr_curve_points,
    compute_roc_curve_points,
)
from src.visualization.latex_tables import (
    generate_latex_benchmark_table,
    generate_latex_closed_set_table,
    generate_latex_paper_table2,
)

try:
    from src.osr.conformal import (
        compute_finite_sample_quantile,
        evaluate_conformal_sensitivity_sweep,
    )
except ImportError:
    def compute_finite_sample_quantile(
        scores: np.ndarray | torch.Tensor,
        alpha: float = 0.05,
    ) -> float:
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy().flatten()
        else:
            scores_np = np.asarray(scores).flatten()
        n = len(scores_np)
        if n == 0:
            return float("inf")
        target_rank = np.ceil((n + 1) * (1.0 - alpha))
        if target_rank > n:
            return float(max(float(np.max(scores_np)), 1e-12))
        q_level = float(min(1.0, max(0.0, target_rank / float(n))))
        return float(np.quantile(scores_np, q_level, method="higher"))

    def evaluate_conformal_sensitivity_sweep(
        cal_scores_per_class: Dict[int, np.ndarray | torch.Tensor],
        test_dist_matrix: np.ndarray | torch.Tensor,
        test_labels: np.ndarray | torch.Tensor,
        num_known_classes: int,
        alphas: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if alphas is None:
            alphas = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        if isinstance(test_dist_matrix, torch.Tensor):
            dist_mat = test_dist_matrix.detach().cpu().numpy()
        else:
            dist_mat = np.asarray(test_dist_matrix)
        if isinstance(test_labels, torch.Tensor):
            labels = test_labels.detach().cpu().numpy()
        else:
            labels = np.asarray(test_labels)

        known_mask = (labels < num_known_classes)
        unknown_mask = (labels == num_known_classes)
        n_known = int(known_mask.sum())
        n_unknown = int(unknown_mask.sum())
        results = []
        for a in alphas:
            taus = {}
            for c in range(num_known_classes):
                if c in cal_scores_per_class and len(cal_scores_per_class[c]) > 0:
                    taus[c] = compute_finite_sample_quantile(cal_scores_per_class[c], alpha=a)
                else:
                    taus[c] = float("inf")
            tau_arr = np.array([taus[c] for c in range(num_known_classes)], dtype=np.float64).reshape(1, -1)
            accepted_mask = dist_mat <= tau_arr
            is_unknown = ~np.any(accepted_mask, axis=1)
            kfr = float(is_unknown[known_mask].mean()) if n_known > 0 else 0.0
            cov = 1.0 - kfr
            rec = float(is_unknown[unknown_mask].mean()) if n_unknown > 0 else 0.0
            n_rej = int(is_unknown.sum())
            tp = int((is_unknown & unknown_mask).sum())
            prec = float(tp / n_rej) if n_rej > 0 else 0.0
            f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
            results.append({
                "alpha": float(a),
                "nominal_coverage": float(1.0 - a),
                "empirical_coverage": float(cov),
                "empirical_kfr": float(kfr),
                "zero_day_recall": float(rec),
                "zero_day_precision": float(prec),
                "zero_day_f1": float(f1),
                "thresholds": {int(c): float(taus[c]) for c in range(num_known_classes)},
                "status": "GUARANTEED" if kfr <= (a + 0.005) else "DRIFT",
            })
        return results
from src.baselines import (
    FedAvgBaseline,
    FedProxBaseline,
    SCAFFOLDBaseline,
    FedDynBaseline,
    FedRoDBaseline,
    FedDecorrBaseline,
    FedProtoBaseline,
    FedPDBaseline,
    FedPDPlusPlusBaseline,
    FedOSSBaseline,
    OpenMaxBaseline,
    EnergyBaseline,
    VanillaMahalanobisBaseline,
)
from src.metrics.statistical_tests import (
    CANONICAL_TEN_SEEDS,
    summarize_multi_seed_Comparison,
)

MODULAR_BASELINE_REGISTRY = {
    "FedAvg": FedAvgBaseline,
    "FedProx": FedProxBaseline,
    "SCAFFOLD": SCAFFOLDBaseline,
    "FedDyn": FedDynBaseline,
    "FedRoD": FedRoDBaseline,
    "FedDecorr": FedDecorrBaseline,
    "FedProto": FedProtoBaseline,
    "FedPD": FedPDBaseline,
    "FedPD++": FedPDPlusPlusBaseline,
    "FedOSS": FedOSSBaseline,
    "OpenMax": OpenMaxBaseline,
    "Energy-OOD": EnergyBaseline,
    "Vanilla-Mahalanobis": VanillaMahalanobisBaseline,
}


def evaluate_method_configuration(
    method_name: str,
    config_path: str = "configs/default.yaml",
    rounds: int = 15,
    beta_dirichlet: Optional[float] = None,
    samples_per_class: Optional[int] = None,
    eval_mode: str = "open_set",
    seed: Optional[int] = None,
    scoring_method: str = "native",
    use_mondrian: bool = False,
    dataset_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    beta: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute complete training and 3-phase open-set evaluation for a given method configuration."""
    if beta_dirichlet is None and beta is not None:
        beta_dirichlet = beta

    import copy
    passed_overrides = kwargs.get("overrides", {}) or {}
    overrides: Dict[str, Any] = copy.deepcopy(passed_overrides)
    if beta_dirichlet is not None:
        overrides.setdefault("federated", {})["partition"] = {"beta_dirichlet": beta_dirichlet}

    config = load_config(config_path, overrides=overrides if overrides else None)
    config.federated.num_rounds = rounds
    if batch_size is not None:
        config.federated.batch_size = batch_size
    if seed is not None:
        config.experiment.seed = seed
    if beta_dirichlet is not None:
        config.federated.partition.beta_dirichlet = beta_dirichlet
    else:
        config.federated.partition.strategy = "iid"
        config.federated.partition.beta_dirichlet = 100.0  # Uniform IID partition

    if dataset_name is not None:
        config.data.dataset_name = dataset_name
    if "num_clients" in kwargs and kwargs["num_clients"] is not None:
        config.federated.num_clients = kwargs["num_clients"]

    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    if device.type == "cuda":
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        vram_used = torch.cuda.memory_allocated(device) / (1024**3)
        dev_name = torch.cuda.get_device_name(device)
        print(
            f"\n  +----------------------------------------------------------------------+\n"
            f"  | [HARDWARE ACCELERATION] Device: GPU CUDA ({dev_name})\n"
            f"  | VRAM Total: {vram_total:.2f} GB | VRAM Allocated: {vram_used:.2f} GB\n"
            f"  | In-VRAM Zero-PCIe Slicing & Direct Tensor Buffering: ENABLED\n"
            f"  +----------------------------------------------------------------------+\n",
            flush=True,
        )
    else:
        print(
            f"\n  +----------------------------------------------------------------------+\n"
            f"  | [HARDWARE ACCELERATION] Device: CPU (Warning: GPU not active)\n"
            f"  | In-RAM Contiguous Slicing: ENABLED\n"
            f"  +----------------------------------------------------------------------+\n",
            flush=True,
        )

    # 1. Prepare Data
    target_ds_name = config.data.dataset_name.lower().replace("-", "_")
    spc = samples_per_class if samples_per_class is not None else getattr(config.data, "samples_per_class", None)

    if target_ds_name in BENCHMARK_METADATA:
        raw_dataset, meta = load_benchmark_dataset(
            dataset_name=target_ds_name,
            data_dir="data/raw",
            samples_per_class=spc,
        )
        config.data.known_classes = meta["known_classes"]
        config.data.unknown_classes = meta["unknown_classes"]
        config.data.in_features = meta["in_features"]
    elif target_ds_name == "synthetic_p2p":
        raw_dataset = generate_synthetic_p2p_data(
            num_samples=config.data.num_samples or 3000,
            in_features=config.data.in_features,
            num_classes=config.data.num_classes_total,
            seed=config.experiment.seed,
        )
    else:
        csv_file = Path(config.data.data_dir) / f"{config.data.dataset_name}.csv"
        if not csv_file.exists():
            csv_file = Path("data/raw") / f"{config.data.dataset_name}.csv"

        if csv_file.exists():
            raw_dataset = load_p2p_tabular_dataset(
                csv_path=csv_file,
                target_column=config.data.target_column or "label",
                normalize=config.data.normalize,
            )
        else:
            raw_dataset = generate_synthetic_p2p_data(
                num_samples=config.data.num_samples or 3000,
                in_features=config.data.in_features,
                num_classes=config.data.num_classes_total,
                seed=config.experiment.seed,
            )

    if eval_mode == "closed_set":
        # Strict Closed-Set Discipline: All classes in dataset become known; zero sequestered unknown classes (LaTeX Section 6.2)
        all_classes = sorted(list(set(config.data.known_classes + config.data.unknown_classes)))
        config.data.known_classes = all_classes
        config.data.unknown_classes = []

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
        protocol=os.environ.get("OSR_PROTOCOL", "full"),
    )
    config.model.in_features = train_ds.features.shape[1]

    # 2. Dirichlet Non-IID Partition
    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=config.federated.partition.beta_dirichlet,
        seed=config.experiment.seed,
    )

    num_known = len(config.data.known_classes)

    # Check for modular baseline execution
    clean_key = method_name.split()[0].replace("+", "").strip()
    target_cls = None
    if method_name in MODULAR_BASELINE_REGISTRY:
        target_cls = MODULAR_BASELINE_REGISTRY[method_name]
    elif clean_key in MODULAR_BASELINE_REGISTRY:
        target_cls = MODULAR_BASELINE_REGISTRY[clean_key]

    # Automatically resolve effective scoring method from method_name or config
    eff_scoring = scoring_method
    if eff_scoring == "native":
        cfg_osr = getattr(config, "osr_method", None)
        m_lower = method_name.lower()
        if "openmax" in m_lower or cfg_osr in ["weibull_openmax", "openmax"]:
            eff_scoring = "openmax"
        elif "energy" in m_lower or cfg_osr in ["energy_ood", "energy"]:
            eff_scoring = "energy"
        elif "mahalanobis" in m_lower or cfg_osr in ["mahalanobis", "vanilla_mahalanobis"]:
            eff_scoring = "mahalanobis"
        elif "msp" in m_lower or cfg_osr in ["maximum_softmax_probability", "msp"]:
            eff_scoring = "msp"

    if target_cls is not None:
        client_datasets = [
            FlowDataset(
                train_ds.features[torch.tensor(sub_idx, dtype=torch.long)],
                train_ds.labels[torch.tensor(sub_idx, dtype=torch.long)],
            )
            for sub_idx in client_indices.values()
        ]
        init_kwargs = {
            "in_features": train_ds.features.shape[1],
            "num_classes": num_known,
            "latent_dim": config.model.latent_dim,
            "hidden_dim": config.model.hidden_dim,
            "device": device,
        }
        if target_cls == MODULAR_BASELINE_REGISTRY.get("FedProx"):
            init_kwargs["mu_prox"] = getattr(config.loss, "fedprox_mu", 0.01)

        baseline = target_cls(**init_kwargs)
        ckpt_base_dir = Path(kwargs.get("checkpoint_dir", "checkpoints"))
        t_base_start = time.perf_counter()
        fit_history = baseline.fit(
            client_datasets=client_datasets,
            num_rounds=rounds,
            epochs=config.federated.local_epochs,
            batch_size=config.federated.batch_size,
            lr=config.optimization.learning_rate,
            optimizer_name=config.optimization.optimizer,
            checkpoint_dir=str(ckpt_base_dir / f"{target_ds_name}_b{beta_dirichlet if beta_dirichlet is not None else 'iid'}"),
            resume=kwargs.get("resume", True) and not kwargs.get("force_retrain", False),
        )
        t_base_fit = time.perf_counter() - t_base_start

        t_base_eval_start = time.perf_counter()
        res = baseline.evaluate(
            test_open_ds=test_open_ds,
            test_closed_ds=test_closed_ds,
            cal_ds=cal_ds,
            conformal_alpha=config.osr.conformal_alpha,
            scoring_method=eff_scoring,
            use_mondrian=use_mondrian,
        )
        t_base_eval = time.perf_counter() - t_base_eval_start

        base_model = (
            getattr(baseline, "model", None)
            or getattr(baseline, "global_model", None)
            or getattr(getattr(baseline, "fedavg", None), "model", None)
        )
        num_params_base = sum(p.numel() for p in base_model.parameters()) if base_model is not None else 0
        base_comm_bytes = int(2 * num_params_base * 4 * rounds)

        res["method_name"] = method_name
        res["dataset_name"] = target_ds_name
        res["beta_dirichlet"] = beta_dirichlet
        res["rounds"] = rounds
        res["seed"] = config.experiment.seed
        res["num_clients"] = config.federated.num_clients
        res["num_known_classes"] = num_known
        res["round_history"] = fit_history if isinstance(fit_history, list) else []
        res["fit_wall_time_sec"] = float(t_base_fit)
        res["eval_wall_time_sec"] = float(t_base_eval)
        res["total_wall_time_sec"] = float(t_base_fit + t_base_eval)
        res["total_comm_bytes_per_client"] = base_comm_bytes
        res["n_train"] = len(train_ds)
        res["n_test_closed"] = len(test_closed_ds)
        res["n_test_open"] = len(test_open_ds)

        if eval_mode == "closed_set":
            res["macro_f1"] = res["closed_set_macro_f1"]
            return res
        return res

    cal_client_indices = partition_dirichlet_non_iid(
        dataset=cal_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=config.federated.partition.beta_dirichlet,
        seed=config.experiment.seed + 1000,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])

        cal_sub_indices = cal_client_indices[client_id]
        cal_idx_t = torch.tensor(cal_sub_indices, dtype=torch.long)
        sub_cal_dataset = FlowDataset(cal_ds.features[cal_idx_t], cal_ds.labels[cal_idx_t])

        clients.append(FederatedClient(client_id, sub_dataset, config, device, cal_dataset=sub_cal_dataset))

    server = FederatedServer(config, clients, device)

    # 3. Phase I: Collaborative Federated Training
    ckpt_base_dir = Path(kwargs.get("checkpoint_dir", "checkpoints"))
    method_slug = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    skew_slug = f"b{beta_dirichlet:.2f}" if beta_dirichlet is not None else "iid"
    ckpt_dir = ckpt_base_dir / f"{target_ds_name}_{method_slug}_{skew_slug}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "checkpoint_latest.pt"

    start_round = 1
    resume_enabled = kwargs.get("resume", True) and not kwargs.get("force_retrain", False)
    if resume_enabled and latest_ckpt.exists():
        try:
            saved_state = torch.load(latest_ckpt, map_location=device, weights_only=False)
            saved_sd = saved_state.get("model_state_dict", {})
            first_w = saved_sd.get("encoder.network.0.weight", None)
            saved_in_f = first_w.shape[1] if first_w is not None else saved_state.get("in_features", None)
            curr_in_f = config.model.in_features

            if saved_in_f is not None and saved_in_f != curr_in_f:
                print(
                    f"    [CHECKPOINT INCOMPATIBLE] Found checkpoint with in_features={saved_in_f}, "
                    f"but current model expects in_features={curr_in_f}. Starting fresh with zero leakage.",
                    flush=True,
                )
                start_round = 1
            else:
                server.global_model.load_state_dict(saved_sd)
                saved_r = saved_state.get("round", 0)
                if saved_r < rounds:
                    start_round = saved_r + 1
                    print(f"    [CHECKPOINT RESUME] Resuming {method_name} from Round {start_round}/{rounds} via {latest_ckpt}...", flush=True)
                else:
                    print(f"    [CHECKPOINT COMPLETED] Found completed {saved_r}/{rounds} rounds in {latest_ckpt}. Skipping training loop.", flush=True)
                    start_round = rounds + 1
        except Exception as e:
            print(f"    [CHECKPOINT WARNING] Could not load {latest_ckpt}: {e}. Starting fresh.", flush=True)
            start_round = 1

    t_p1_start = time.perf_counter()
    log_interval = 1 if rounds <= 30 else max(1, rounds // 20)
    test_closed_x = test_closed_ds.features.to(device)
    test_closed_y = test_closed_ds.labels.numpy()
    round_history: List[Dict[str, Any]] = []

    for r in range(start_round, rounds + 1):
        t_r0 = time.perf_counter()
        r_loss = server.run_round(r)
        
        r_entry = dict(r_loss) if isinstance(r_loss, dict) else {"train_loss": float(r_loss)}
        r_entry["round"] = r
        r_entry["elapsed_sec"] = float(time.perf_counter() - t_p1_start)

        # Save periodic checkpoint to disk / Google Drive with full architecture metadata
        if r % log_interval == 0 or r == rounds:
            torch.save({
                "round": r,
                "model_state_dict": server.global_model.state_dict(),
                "in_features": config.model.in_features,
                "num_classes": num_known,
                "dataset_name": target_ds_name,
                "beta_dirichlet": beta_dirichlet,
            }, latest_ckpt)

        if r == 1 or r % log_interval == 0 or r == rounds:
            with torch.no_grad():
                logits_c = server.global_model.classify(test_closed_x)
                preds_c = torch.argmax(logits_c, dim=-1).cpu().numpy()
                c_acc = float(np.mean(preds_c == test_closed_y))
            r_entry["val_closed_accuracy"] = c_acc
            elapsed = time.perf_counter() - t_p1_start
            r_sec = time.perf_counter() - t_r0
            rounds_done = r - start_round + 1
            avg_sec = elapsed / max(rounds_done, 1)
            eta_sec = avg_sec * (rounds - r)
            loss_val = r_entry.get("train_loss", r_entry.get("loss_bla", 0.0))
            vram_str = f" | VRAM: {torch.cuda.memory_allocated(device)/(1024**2):.0f}MB" if device.type == "cuda" else ""
            print(
                f"    [FedTROS-MC] Round {r:3d}/{rounds} | Loss: {loss_val:.4f} | Val Acc: {c_acc * 100:.2f}% | "
                f"Speed: {r_sec:.2f}s/rnd{vram_str} | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s",
                flush=True,
            )
        round_history.append(r_entry)

    t_phase1 = time.perf_counter() - t_p1_start
    # Freeze encoder permanently (Section 4.1 & 4.3)
    server.freeze_global_encoder()

    num_known = len(config.data.known_classes)

    # Phase II: Zero-Leakage Federated Class Profiling via Law of Total Variance
    t_p2_start = time.perf_counter()
    server.run_phase_two_profiling(
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
        alpha=config.osr.conformal_alpha,
        cal_dataset=cal_ds,)
    t_phase2 = time.perf_counter() - t_p2_start

    print(f"\n    [STAGE 2 & 3: MULTI-PROTOTYPE PROFILING & CONFORMAL CALIBRATION]")
    for c, prof in server.global_profiles.items():
        n_protos = prof.prototypes.size(0) if getattr(prof, "prototypes", None) is not None else 1
        print(f"      * Class {c}: {n_protos:2d} prototypes | Conformal tau = {prof.conformal_threshold:.4f} (alpha = {config.osr.conformal_alpha})")

    num_params = sum(p.numel() for p in server.global_model.parameters())
    p1_bytes = int(2 * num_params * 4 * rounds)
    p2_bytes = int(num_known * (config.model.latent_dim + config.federated.num_blocks * (config.federated.block_dim ** 2) + 2) * 4)

    # Phase III: Wire-Speed Block-Diagonal Mahalanobis Edge Detector
    edge_detector = server.get_edge_detector(
        unknown_label=num_known,
        num_blocks=config.federated.num_blocks,
        block_dim=config.federated.block_dim,
    )

    if eval_mode == "closed_set":
        _, _, preds_closed, _ = edge_detector.predict_batch(test_closed_ds.features)
        y_closed_true = test_closed_ds.labels.numpy()
        closed_metrics = compute_classification_metrics(
            y_true=y_closed_true,
            y_pred=preds_closed,
            num_classes=num_known,
        )

        server.global_model.eval()
        with torch.no_grad():
            logits_nn = server.global_model.classify(test_closed_ds.features.to(device))
            preds_nn = torch.argmax(logits_nn, dim=-1).cpu().numpy()
        metrics_nn = compute_classification_metrics(
            y_true=y_closed_true,
            y_pred=preds_nn,
            num_classes=num_known,
        )

        closed_metrics["mahalanobis_accuracy"] = float(closed_metrics["accuracy"])
        closed_metrics["mahalanobis_macro_f1"] = float(closed_metrics["macro_f1"])
        closed_metrics["mahalanobis_class_wise_f1"] = closed_metrics.get("class_wise_f1", [])
        closed_metrics["mahalanobis_confusion_matrix"] = closed_metrics.get("confusion_matrix", [])

        closed_metrics["nn_head_accuracy"] = float(metrics_nn["accuracy"])
        closed_metrics["nn_head_macro_f1"] = float(metrics_nn["macro_f1"])
        closed_metrics["nn_head_class_wise_f1"] = metrics_nn.get("class_wise_f1", [])
        closed_metrics["nn_head_confusion_matrix"] = metrics_nn.get("confusion_matrix", [])

        print(
            f"\n    [CLOSED-SET EVALUATION RESULTS]\n"
            f"      * Neural Classifier Head (Softmax): Accuracy = {metrics_nn['accuracy'] * 100:.2f}% | Macro-F1 = {metrics_nn['macro_f1'] * 100:.2f}%\n"
            f"      * Mahalanobis Edge Profile:        Accuracy = {closed_metrics['accuracy'] * 100:.2f}% | Macro-F1 = {closed_metrics['macro_f1'] * 100:.2f}%\n",
            flush=True,
        )

        closed_metrics["round_history"] = round_history
        closed_metrics["phase1_wall_time_sec"] = float(t_phase1)
        closed_metrics["phase2_wall_time_sec"] = float(t_phase2)
        closed_metrics["total_wall_time_sec"] = float(t_phase1 + t_phase2)
        closed_metrics["phase1_comm_bytes_per_client"] = p1_bytes
        closed_metrics["phase2_comm_bytes_per_client"] = p2_bytes
        closed_metrics["total_comm_bytes_per_client"] = p1_bytes + p2_bytes
        closed_metrics["method_name"] = method_name
        closed_metrics["dataset_name"] = target_ds_name
        closed_metrics["beta_dirichlet"] = beta_dirichlet
        closed_metrics["rounds"] = rounds
        closed_metrics["seed"] = config.experiment.seed
        closed_metrics["num_clients"] = config.federated.num_clients
        closed_metrics["num_classes"] = num_known
        closed_metrics["n_train"] = len(train_ds)
        closed_metrics["n_test_closed"] = len(test_closed_ds)
        return closed_metrics

    # Evaluate on open-set and closed-set test data using regularized Mahalanobis profiles
    det = edge_detector.predict_batch_detailed(test_open_ds.features)
    _, _, preds_closed, _ = edge_detector.predict_batch(test_closed_ds.features)

    y_open_true = test_open_ds.labels.numpy()
    from src.metrics.osr_metrics import compute_binary_auroc, compute_fpr95
    y_bin = (y_open_true == num_known).astype(int)

    metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=det["predictions"],
        anomaly_scores=det["raw_scores"],
        num_known_classes=num_known,
        unknown_label=num_known,
    )
    metrics["auroc_raw"] = float(metrics["auroc"])
    metrics["fpr95_raw"] = float(metrics["fpr95"])
    metrics["auroc_normalized"] = float(compute_binary_auroc(y_bin, det["normalized_scores"]))
    metrics["fpr95_normalized"] = float(compute_fpr95(y_bin, det["normalized_scores"]))

    y_closed_true = test_closed_ds.labels.numpy()
    server.global_model.eval()
    with torch.no_grad():
        logits_nn = server.global_model.classify(test_closed_ds.features.to(device))
        preds_nn = torch.argmax(logits_nn, dim=-1).cpu().numpy()
    metrics_nn = compute_classification_metrics(
        y_true=y_closed_true,
        y_pred=preds_nn,
        num_classes=num_known,
    )
    closed_metrics = compute_classification_metrics(
        y_true=y_closed_true,
        y_pred=preds_closed,
        num_classes=num_known,
    )

    cm_before = compute_confusion_matrix(y_open_true, det["candidate_classes"], labels=list(range(num_known + 1)))
    cm_after = compute_confusion_matrix(y_open_true, det["predictions"], labels=list(range(num_known + 1)))

    metrics["closed_set_accuracy"] = closed_metrics["accuracy"]
    metrics["accuracy"] = closed_metrics["accuracy"]
    metrics["closed_set_balanced_accuracy"] = closed_metrics["balanced_accuracy"]
    metrics["closed_set_macro_f1"] = closed_metrics["macro_f1"]
    metrics["closed_set_class_wise_f1"] = closed_metrics.get("class_wise_f1", [])
    metrics["closed_set_confusion_matrix"] = closed_metrics.get("confusion_matrix", [])

    metrics["nn_head_closed_accuracy"] = float(metrics_nn["accuracy"])
    metrics["nn_head_closed_macro_f1"] = float(metrics_nn["macro_f1"])
    metrics["mahalanobis_closed_accuracy"] = float(closed_metrics["accuracy"])
    metrics["mahalanobis_closed_macro_f1"] = float(closed_metrics["macro_f1"])

    # Pairwise separability (Zero-Day vs each individual known class)
    pairwise_auroc_raw: Dict[int, float] = {}
    pairwise_auroc_norm: Dict[int, float] = {}
    for c in range(num_known):
        m_c = (y_open_true == c) | (y_open_true == num_known)
        if int(m_c.sum()) > 0 and int((y_open_true == c).sum()) > 0 and int((y_open_true == num_known).sum()) > 0:
            pairwise_auroc_raw[int(c)] = float(compute_binary_auroc(y_bin[m_c], det["raw_scores"][m_c]))
            pairwise_auroc_norm[int(c)] = float(compute_binary_auroc(y_bin[m_c], det["normalized_scores"][m_c]))
    metrics["pairwise_auroc_raw"] = pairwise_auroc_raw
    metrics["pairwise_auroc_normalized"] = pairwise_auroc_norm

    # Score distribution statistics for known vs zero-day unknown
    k_mask = (y_bin == 0)
    u_mask = (y_bin == 1)
    raw_k = det["raw_scores"][k_mask]
    raw_u = det["raw_scores"][u_mask]
    norm_k = det["normalized_scores"][k_mask]
    norm_u = det["normalized_scores"][u_mask]

    metrics["score_stats"] = {
        "raw_known_mean": float(np.mean(raw_k)),
        "raw_known_std": float(np.std(raw_k)),
        "raw_known_p95": float(np.percentile(raw_k, 95)),
        "raw_unknown_mean": float(np.mean(raw_u)),
        "raw_unknown_std": float(np.std(raw_u)),
        "raw_unknown_median": float(np.median(raw_u)),
        "norm_known_mean": float(np.mean(norm_k)),
        "norm_known_std": float(np.std(norm_k)),
        "norm_known_p95": float(np.percentile(norm_k, 95)),
        "norm_unknown_mean": float(np.mean(norm_u)),
        "norm_unknown_std": float(np.std(norm_u)),
        "norm_unknown_median": float(np.median(norm_u)),
        "separation_margin_raw": float(np.mean(raw_u) - np.mean(raw_k)),
        "separation_margin_norm": float(np.mean(norm_u) - np.mean(norm_k)),
    }

    # Binned density distributions for instant plotting (KDE / Histogram curves)
    max_score = float(np.percentile(norm_u, 99.0)) if len(norm_u) > 0 else 10.0
    bins = np.linspace(0.0, max(max_score, 5.0), 51)
    h_k, _ = np.histogram(norm_k, bins=bins, density=True)
    h_u, _ = np.histogram(norm_u, bins=bins, density=True)
    metrics["score_distribution_hist"] = {
        "bin_edges": [round(float(b), 4) for b in bins],
        "known_density": [round(float(h), 6) for h in h_k],
        "unknown_density": [round(float(h), 6) for h in h_u],
    }

    # Sampled latent embeddings for t-SNE / PCA / UMAP visualization (Figure generation)
    np.random.seed(config.experiment.seed)
    sample_indices: List[int] = []
    for c in range(num_known + 1):
        c_idx = np.where(y_open_true == c)[0]
        if len(c_idx) > 0:
            sample_count = min(len(c_idx), 250)
            sampled = np.random.choice(c_idx, sample_count, replace=False)
            sample_indices.extend(sampled.tolist())
    sample_indices = sorted(sample_indices)
    metrics["sample_embeddings"] = {
        "indices": sample_indices,
        "y_true": [int(y_open_true[i]) for i in sample_indices],
        "latent_z": [[round(float(v), 5) for v in row] for row in det["latent_z"][sample_indices]],
        "raw_scores": [round(float(det["raw_scores"][i]), 4) for i in sample_indices],
        "normalized_scores": [round(float(det["normalized_scores"][i]), 4) for i in sample_indices],
        "predictions": [int(det["predictions"][i]) for i in sample_indices],
    }

    # Class profiles and conformal metadata audit
    known_names = meta.get("known_class_names", [f"Class_{c}" for c in range(num_known)])
    class_wise_f1 = closed_metrics.get("class_wise_f1", [])
    metrics["class_profiles_metadata"] = {
        int(c): {
            "name": known_names[c] if c < len(known_names) else f"Class_{c}",
            "num_prototypes": prof.prototypes.size(0) if getattr(prof, "prototypes", None) is not None else 1,
            "conformal_tau": float(prof.conformal_threshold),
            "cal_samples": int(server.calibration_counts.get(c, 0)),
            "empirical_kfr": float(metrics.get("per_class_kfr", {}).get(c, 0.0)),
            "coverage": float(metrics.get("per_class_coverage", {}).get(c, 1.0)),
            "closed_f1": float(class_wise_f1[c]) if c < len(class_wise_f1) else 0.0,
            "pairwise_auroc": float(pairwise_auroc_raw.get(c, 0.0)),
        }
        for c, prof in server.global_profiles.items()
    }

    kfr_val = metrics.get("kfr", 0.0) * 100
    oscr_val = metrics.get("oscr", 0.0) * 100
    print(
        f"\n    [OPEN-SET EVALUATION RESULTS (Section 5 / Table 4 Aligned)]\n"
        f"      * Zero-Day AUROC (Raw / Norm):  {metrics.get('auroc', 0.0) * 100:.2f}% / {metrics.get('auroc_normalized', 0.0) * 100:.2f}%\n"
        f"      * Open-Set OSCR:                {oscr_val:.2f}%\n"
        f"      * Zero-Day FPR@95:              {metrics.get('fpr95', 0.0) * 100:.2f}%\n"
        f"      * Conformal KFR:                {kfr_val:.2f}% (Nominal bound <= {config.osr.conformal_alpha * 100:.1f}%)\n"
        f"      * Zero-Day Recall:              {metrics.get('unknown_recall', 0.0) * 100:.2f}% (at alpha={config.osr.conformal_alpha})\n"
        f"      * Known Test Acc (NN Head):     {metrics_nn['accuracy'] * 100:.2f}% (Macro-F1: {metrics_nn['macro_f1'] * 100:.2f}%)\n"
        f"      * Known Test Acc (Maha Edge):   {closed_metrics['accuracy'] * 100:.2f}% (Macro-F1: {closed_metrics['macro_f1'] * 100:.2f}%)\n"
        f"      * Score Separation Margin:      Known: {metrics['score_stats']['raw_known_mean']:.2f} +/- {metrics['score_stats']['raw_known_std']:.2f} | "
        f"Unknown: {metrics['score_stats']['raw_unknown_mean']:.2f} +/- {metrics['score_stats']['raw_unknown_std']:.2f} (Delta={metrics['score_stats']['separation_margin_raw']:.2f})\n",
        flush=True,
    )

    fot_mask = (y_open_true == num_known)
    n_fot = int(fot_mask.sum())
    if n_fot > 0:
        fot_preds = det["predictions"][fot_mask]
        fot_rej = int(np.sum(fot_preds == num_known))
        print(f"    [ZERO-DAY LEAKAGE & ROBUSTNESS AUDIT]")
        print(f"      * Total Sequestered Zero-Day Attacks: {n_fot}")
        print(f"      * Correctly Rejected as Unknown (C+1): {fot_rej} ({fot_rej / n_fot * 100:.2f}%)")
        for c_idx in range(num_known):
            swallowed_c = int(np.sum(fot_preds == c_idx))
            print(f"      * Swallowed by Known Class {c_idx}:    {swallowed_c} ({swallowed_c / n_fot * 100:.2f}%)")
        print("", flush=True)

    cal_scores = getattr(server, "cal_scores_per_class", None)
    if cal_scores is not None and "dist_matrix" in det:
        sweep_results = evaluate_conformal_sensitivity_sweep(
            cal_scores_per_class=cal_scores,
            test_dist_matrix=det["dist_matrix"],
            test_labels=y_open_true,
            num_known_classes=num_known,
            alphas=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        )
        metrics["conformal_sensitivity_sweep"] = sweep_results

        print("    [CONFORMAL SENSITIVITY SWEEP (Section 5.4 / Operating Characteristic Curve)]")
        print(f"    {'Alpha':<6} | {'Nominal Cov':<11} | {'Empirical Cov':<13} | {'KFR (<=alpha)':<13} | {'Zero-Day Rec':<12} | {'Precision':<10} | {'FoT F1':<8} | {'Status':<10}")
        print("    " + "-" * 95)
        for row in sweep_results:
            print(f"    {row['alpha']:<6.2f} | {row['nominal_coverage']*100:<10.1f}% | {row['empirical_coverage']*100:<12.2f}% | {row['empirical_kfr']*100:<12.2f}% | {row['zero_day_recall']*100:<11.2f}% | {row['zero_day_precision']*100:<9.2f}% | {row['zero_day_f1']*100:<7.2f}% | {row['status']:<10}")
        print("    " + "-" * 95 + "\n", flush=True)

    metrics["round_history"] = round_history
    metrics["phase1_wall_time_sec"] = float(t_phase1)
    metrics["phase2_wall_time_sec"] = float(t_phase2)
    metrics["total_wall_time_sec"] = float(t_phase1 + t_phase2)
    metrics["phase1_comm_bytes_per_client"] = p1_bytes
    metrics["phase2_comm_bytes_per_client"] = p2_bytes
    metrics["total_comm_bytes_per_client"] = p1_bytes + p2_bytes
    metrics["confusion_matrix_before"] = cm_before.tolist()
    metrics["confusion_matrix_after"] = cm_after.tolist()
    metrics["thresholds"] = {int(k): float(v) for k, v in server.calibration_thresholds.items()}
    metrics["method_name"] = method_name
    metrics["dataset_name"] = target_ds_name
    metrics["beta_dirichlet"] = beta_dirichlet
    metrics["rounds"] = rounds
    metrics["seed"] = config.experiment.seed
    metrics["num_clients"] = config.federated.num_clients
    metrics["num_known_classes"] = num_known
    metrics["n_train"] = len(train_ds)
    metrics["n_test_closed"] = len(test_closed_ds)
    metrics["n_test_open"] = len(test_open_ds)

    fpr_pts, tpr_pts = compute_roc_curve_points((y_open_true == num_known).astype(int), det["normalized_scores"])
    prec_pts, rec_pts = compute_pr_curve_points((y_open_true == num_known).astype(int), det["normalized_scores"])

    def _sample_curve(x_arr: np.ndarray, y_arr: np.ndarray, num_pts: int = 50) -> Tuple[List[float], List[float]]:
        if len(x_arr) <= num_pts:
            return [round(float(v), 5) for v in x_arr], [round(float(v), 5) for v in y_arr]
        idx = np.linspace(0, len(x_arr) - 1, num_pts, dtype=int)
        return [round(float(x_arr[i]), 5) for i in idx], [round(float(y_arr[i]), 5) for i in idx]

    roc_fpr, roc_tpr = _sample_curve(fpr_pts, tpr_pts)
    pr_prec, pr_rec = _sample_curve(prec_pts, rec_pts)
    metrics["roc_curve"] = {"fpr": roc_fpr, "tpr": roc_tpr}
    metrics["pr_curve"] = {"precision": pr_prec, "recall": pr_rec}

    return metrics


def run_all_baselines(
    rounds: int = 10,
    beta_dirichlet: float = 0.1,
    samples_per_class: Optional[int] = None,
    output_dir: str = "results/baselines",
    tier: Optional[str] = None,
    specific_methods: Optional[List[str]] = None,
    scoring_method: str = "native",
    use_mondrian: bool = False,
    num_seeds: int = 1,
) -> None:
    """Run comparative baseline suite across canonical algorithms under identical data splits."""
    logger = setup_logger(name="baseline_runner")
    logger.info(
        f"Running complete baseline comparison suite (rounds={rounds}, beta={beta_dirichlet}, "
        f"samples_per_class={samples_per_class}, tier={tier}, scoring={scoring_method}, "
        f"mondrian={use_mondrian}, num_seeds={num_seeds})..."
    )

    all_methods = {
        # Category 1: Canonical General FL
        "FedAvg": "configs/baselines/fedavg_msp.yaml",
        "FedProx": "configs/baselines/fedprox_openmax.yaml",
        # Category 2: Non-IID Drift-Corrected FL (SOTA)
        "SCAFFOLD": "configs/baselines/scaffold.yaml",
        "FedDyn": "configs/baselines/feddyn.yaml",
        # Category 3: Balanced Representation FL
        "FedRoD": "configs/baselines/fedrod.yaml",
        "FedDecorr": "configs/baselines/feddecorr.yaml",
        "FedProto": "configs/baselines/fedproto_energy.yaml",
        # Category 4: Direct Federated Open-Set Recognition (FedOSR Competitors)
        "FedPD": "configs/baselines/fedpd.yaml",
        "FedPD++": "configs/baselines/fedpd_plus_plus.yaml",
        "FedOSS": "configs/baselines/fedoss.yaml",
        # Category 5: Centralized OSR Adapted to FL
        "OpenMax": "configs/baselines/fedprox_openmax.yaml",
        "Energy-OOD": "configs/baselines/fedproto_energy.yaml",
        "Vanilla-Mahalanobis": "configs/baselines/vanilla_mahalanobis.yaml",
        # Proposed Framework
        "Proposed (Decoupled VIB-Mahalanobis)": "configs/default.yaml",
    }

    tier_mapping = {
        "tier1": ["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "Proposed (Decoupled VIB-Mahalanobis)"],
        "tier2": ["FedRoD", "FedDecorr", "FedProto", "Proposed (Decoupled VIB-Mahalanobis)"],
        "tier3": ["FedPD", "FedPD++", "FedOSS", "Proposed (Decoupled VIB-Mahalanobis)"],
        "tier4": ["OpenMax", "Energy-OOD", "Vanilla-Mahalanobis", "Proposed (Decoupled VIB-Mahalanobis)"],
        "canonical": [
            "FedAvg",
            "FedProx",
            "SCAFFOLD",
            "FedDyn",
            "FedRoD",
            "FedDecorr",
            "FedPD",
            "FedOSS",
            "Proposed (Decoupled VIB-Mahalanobis)",
        ],
    }

    if specific_methods is not None:
        methods = {k: all_methods[k] for k in specific_methods if k in all_methods}
    elif tier is not None and tier.lower() in tier_mapping:
        selected_keys = tier_mapping[tier.lower()]
        methods = {k: all_methods[k] for k in selected_keys}
    else:
        methods = all_methods

    seeds_to_run = list(CANONICAL_TEN_SEEDS[: max(1, min(num_seeds, len(CANONICAL_TEN_SEEDS)))])
    all_results: Dict[str, Dict[str, Any]] = {}
    multi_seed_records: Dict[str, Dict[str, List[float]]] = {name: {} for name in methods}

    for name, cfg_path in methods.items():
        logger.info(f"--> Evaluating: {name} (config: {cfg_path}) across {len(seeds_to_run)} seed(s)")
        seed_metrics_list = []
        for s_val in seeds_to_run:
            m = evaluate_method_configuration(
                name,
                cfg_path,
                rounds=rounds,
                beta_dirichlet=beta_dirichlet,
                samples_per_class=samples_per_class,
                seed=s_val,
                scoring_method=scoring_method,
                use_mondrian=use_mondrian,
            )
            seed_metrics_list.append(m)
            for k_m, v_m in m.items():
                if isinstance(v_m, (int, float)):
                    multi_seed_records[name].setdefault(k_m, []).append(float(v_m))

        # Aggregate mean & std across seeds
        agg_m = dict(seed_metrics_list[0])
        for k_m, vals in multi_seed_records[name].items():
            agg_m[k_m] = float(np.mean(vals))
            agg_m[f"{k_m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        all_results[name] = agg_m
        logger.info(
            f"    [{name}] AUROC: {agg_m['auroc']*100:.2f}% | "
            f"AUPR: {agg_m['aupr']*100:.2f}% | "
            f"FPR95: {agg_m.get('fpr95', 0.0)*100:.2f}% | "
            f"Unk-Recall: {agg_m['unknown_recall']*100:.2f}% | "
            f"Unk-F1: {agg_m['unknown_f1']*100:.2f}% | "
            f"OS Macro-F1: {agg_m['open_set_macro_f1']*100:.2f}% | "
            f"KFR: {agg_m['kfr']*100:.2f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    tier_suffix = f"_{tier}" if tier else ""
    save_json(all_results, out_p / f"baseline_comparison_beta_{beta_dirichlet}{tier_suffix}.json")

    if len(seeds_to_run) > 1 and "Proposed (Decoupled VIB-Mahalanobis)" in multi_seed_records:
        prop_runs = multi_seed_records["Proposed (Decoupled VIB-Mahalanobis)"]
        base_runs = {k: v for k, v in multi_seed_records.items() if k != "Proposed (Decoupled VIB-Mahalanobis)"}
        stat_summary = summarize_multi_seed_Comparison(prop_runs, base_runs, alpha=0.01)
        save_json(stat_summary, out_p / f"statistical_significance_beta_{beta_dirichlet}{tier_suffix}.json")

    # Save standardized Q1 audit log for peer review
    q1_baselines = {
        name: format_q1_experiment_log(
            dataset="B-NAT",
            alpha_dirichlet=beta_dirichlet,
            round_num=rounds,
            closed_metrics={
                "accuracy": m.get("accuracy", 0.0),
                "balanced_accuracy": m.get("balanced_accuracy", m.get("accuracy", 0.0)),
                "macro_f1": m.get("closed_set_macro_f1", 0.0),
                "class_wise_f1": m.get("class_wise_f1", []),
                "confusion_matrix": m.get("confusion_matrix", []),
            },
            osr_metrics=m,
        )
        for name, m in all_results.items()
    }
    save_json(q1_baselines, out_p / f"q1_baseline_comparison_beta_{beta_dirichlet}{tier_suffix}.json")

    latex_tbl = generate_latex_benchmark_table(
        all_results,
        caption=f"Baseline Comparison under Non-IID Dirichlet Skew ($\\alpha_D = {beta_dirichlet}$)",
        label=f"tab:baseline_comparison_{str(beta_dirichlet).replace('.', '_')}",
    )
    with open(out_p / f"table_baselines_beta_{beta_dirichlet}{tier_suffix}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Baseline benchmarking completed. Table written to {out_p}")


def run_closed_set_baselines(
    rounds: int = 10,
    beta_dirichlet: Optional[float] = None,
    samples_per_class: Optional[int] = None,
    output_dir: str = "results/closed_set",
    tier: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run closed-set classification benchmarks (Section 6.2, Tables 1 & 2).

    Evaluates purely closed-set accuracy, balanced accuracy, and Macro-F1.
    Strictly excludes Phase II/III open-set evaluation and metrics.
    """
    logger = setup_logger(name="closed_set_runner")
    skew_label = "IID" if beta_dirichlet is None else f"alpha_D = {beta_dirichlet}"
    logger.info(
        f"Running Closed-Set Classification Suite ({skew_label}, rounds={rounds}, samples_per_class={samples_per_class})..."
    )

    all_methods = {
        "FedAvg": "configs/baselines/fedavg_msp.yaml",
        "FedProx": "configs/baselines/fedprox_openmax.yaml",
        "SCAFFOLD": "configs/baselines/scaffold.yaml",
        "FedDyn": "configs/baselines/feddyn.yaml",
        "FedRoD": "configs/baselines/fedrod.yaml",
        "FedDecorr": "configs/baselines/feddecorr.yaml",
        "FedProto": "configs/baselines/fedproto_energy.yaml",
        "FedPD": "configs/baselines/fedpd.yaml",
        "FedPD++": "configs/baselines/fedpd_plus_plus.yaml",
        "FedOSS": "configs/baselines/fedoss.yaml",
        "Proposed (Decoupled VIB-Mahalanobis)": "configs/default.yaml",
    }

    if tier == "canonical":
        methods = {
            "FedAvg": all_methods["FedAvg"],
            "FedProx": all_methods["FedProx"],
            "SCAFFOLD": all_methods["SCAFFOLD"],
            "FedDyn": all_methods["FedDyn"],
            "FedRoD": all_methods["FedRoD"],
            "FedDecorr": all_methods["FedDecorr"],
            "FedProto": all_methods["FedProto"],
            "Proposed (Decoupled VIB-Mahalanobis)": all_methods["Proposed (Decoupled VIB-Mahalanobis)"],
        }
    else:
        methods = all_methods

    all_results: Dict[str, Dict[str, float]] = {}
    for name, cfg_path in methods.items():
        logger.info(f"--> Evaluating Closed-Set: {name} (config: {cfg_path})")
        metrics = evaluate_method_configuration(
            name,
            cfg_path,
            rounds=rounds,
            beta_dirichlet=beta_dirichlet,
            samples_per_class=samples_per_class,
            eval_mode="closed_set",
        )
        all_results[name] = metrics
        logger.info(
            f"    [{name}] Accuracy: {metrics['accuracy']*100:.2f}% | "
            f"Balanced Acc: {metrics['balanced_accuracy']*100:.2f}% | "
            f"Macro-F1: {metrics['macro_f1']*100:.2f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    suffix = "iid" if beta_dirichlet is None else f"beta_{beta_dirichlet}"
    save_json(all_results, out_p / f"closed_set_comparison_{suffix}.json")

    # Save standardized Q1 audit log for peer review
    q1_closed = {
        name: format_q1_experiment_log(
            dataset="B-NAT",
            alpha_dirichlet=beta_dirichlet,
            round_num=rounds,
            closed_metrics=m,
        )
        for name, m in all_results.items()
    }
    save_json(q1_closed, out_p / f"q1_closed_set_comparison_{suffix}.json")

    caption = (
        "IID closed-set classification performance on the B-NAT benchmark."
        if beta_dirichlet is None
        else f"Closed-Set Classification Performance ($\\alpha_D = {beta_dirichlet}$)"
    )
    label = "tab:iid_closed_results" if beta_dirichlet is None else f"tab:closed_set_{str(beta_dirichlet).replace('.', '_')}"

    latex_tbl = generate_latex_closed_set_table(
        all_results,
        caption=caption,
        label=label,
    )
    with open(out_p / f"table_closed_set_{suffix}.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Closed-set benchmarking completed. Table written to {out_p}")
    return all_results


def run_closed_set_multi_skew_benchmark(
    rounds: int = 10,
    skews: Optional[list] = None,
    samples_per_class: Optional[int] = None,
    output_dir: str = "results/closed_set",
) -> Dict[float, Dict[str, float]]:
    """Run closed-set multi-skew benchmark for Proposed FedTROS-MC across Dirichlet skews (Table 2)."""
    if skews is None:
        skews = [1.0, 0.5, 0.1]
    logger = setup_logger(name="closed_set_multi_skew_runner")
    logger.info(f"Running Closed-Set Multi-Skew Suite for FedTROS-MC across alpha_D in {skews}...")

    results_by_skew: Dict[float, Dict[str, float]] = {}
    for alpha_d in skews:
        logger.info(f"--> Evaluating Closed-Set FedTROS-MC under alpha_D = {alpha_d}")
        metrics = evaluate_method_configuration(
            "FedTROS-MC (Proposed)",
            "configs/default.yaml",
            rounds=rounds,
            beta_dirichlet=alpha_d,
            samples_per_class=samples_per_class,
            eval_mode="closed_set",
        )
        results_by_skew[alpha_d] = metrics
        logger.info(
            f"    [alpha_D={alpha_d}] Accuracy: {metrics['accuracy']*100:.3f}% | "
            f"Balanced Acc: {metrics['balanced_accuracy']*100:.3f}% | "
            f"Macro-F1: {metrics['macro_f1']*100:.3f}%"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(results_by_skew, out_p / "closed_set_multi_skew_results.json")

    # Save standardized Q1 audit log for peer review
    q1_multi_skew = {
        f"alpha_D_{alpha_d}": format_q1_experiment_log(
            dataset="B-NAT",
            alpha_dirichlet=alpha_d,
            round_num=rounds,
            closed_metrics=m,
        )
        for alpha_d, m in results_by_skew.items()
    }
    save_json(q1_multi_skew, out_p / "q1_closed_set_multi_skew.json")

    latex_tbl = generate_latex_paper_table2(
        results_by_skew,
        caption=r"Closed-set B-NAT classification performance under varying Dirichlet label skew ($\alpha_D \in \{1.0, 0.5, 0.1\}$).",
        label="tab:noniid_closed_results",
    )
    with open(out_p / "table_noniid_closed_results.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Closed-set multi-skew benchmarking completed. Table written to {out_p}")
    return results_by_skew


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Baselines Suite")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per baseline")
    parser.add_argument("--beta", type=float, default=0.1, help="Dirichlet concentration alpha_D")
    parser.add_argument("--samples_per_class", type=int, default=None, help="Samples per class limit (None for full dataset)")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--tier", type=str, default=None, choices=["tier1", "tier2", "tier3", "tier4", "canonical", "all"], help="Filter baselines by tier")
    parser.add_argument("--methods", nargs="+", default=None, help="Specific list of method names to evaluate")
    parser.add_argument("--scoring_method", type=str, default="native", choices=["native", "msp", "energy", "mahalanobis"], help="Unified OSR scoring head")
    parser.add_argument("--use_mondrian", action="store_true", help="Enable Mondrian class-conditional conformal calibration")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of canonical seeds (1, 5, or 10 for Wilcoxon test)")
    args = parser.parse_args()

    run_all_baselines(
        rounds=args.rounds,
        beta_dirichlet=args.beta,
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
        tier=args.tier,
        specific_methods=args.methods,
        scoring_method=args.scoring_method,
        use_mondrian=args.use_mondrian,
        num_seeds=args.num_seeds,
    )

