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
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Any, Dict, Optional
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
from src.metrics.osr_metrics import compute_osr_metrics
from src.visualization.latex_tables import (
    generate_latex_benchmark_table,
    generate_latex_closed_set_table,
    generate_latex_paper_table2,
)
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
    config_path: str,
    rounds: int = 15,
    beta_dirichlet: Optional[float] = None,
    samples_per_class: Optional[int] = None,
    eval_mode: str = "open_set",
    seed: Optional[int] = None,
    scoring_method: str = "native",
    use_mondrian: bool = False,
) -> Dict[str, Any]:
    """Execute complete training and 3-phase open-set evaluation for a given method configuration."""
    overrides: Dict[str, Any] = {}
    if beta_dirichlet is not None:
        overrides.setdefault("federated", {})["partition"] = {"beta_dirichlet": beta_dirichlet}

    config = load_config(config_path, overrides=overrides if overrides else None)
    config.federated.num_rounds = rounds
    if seed is not None:
        config.experiment.seed = seed
    if beta_dirichlet is not None:
        config.federated.partition.beta_dirichlet = beta_dirichlet
    else:
        config.federated.partition.strategy = "iid"
        config.federated.partition.beta_dirichlet = 100.0  # Uniform IID partition

    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    # 1. Prepare Data
    dataset_name = config.data.dataset_name.lower().replace("-", "_")
    spc = samples_per_class if samples_per_class is not None else getattr(config.data, "samples_per_class", None)

    if dataset_name in BENCHMARK_METADATA:
        raw_dataset, meta = load_benchmark_dataset(
            dataset_name=dataset_name,
            data_dir="data/raw",
            samples_per_class=spc,
        )
        config.data.known_classes = meta["known_classes"]
        config.data.unknown_classes = meta["unknown_classes"]
        config.data.in_features = meta["in_features"]
    elif dataset_name == "synthetic_p2p":
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
        # Strict Closed-Set Discipline: Zero sequestered unknown classes (LaTeX Section 6.2)
        config.data.unknown_classes = []

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
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

    if target_cls is not None:
        client_datasets = [
            FlowDataset(
                train_ds.features[torch.tensor(sub_idx, dtype=torch.long)],
                train_ds.labels[torch.tensor(sub_idx, dtype=torch.long)],
            )
            for sub_idx in client_indices.values()
        ]
        baseline = target_cls(
            in_features=train_ds.features.shape[1],
            num_classes=num_known,
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            device=device,
        )
        baseline.fit(
            client_datasets=client_datasets,
            num_rounds=rounds,
            epochs=config.federated.local_epochs,
            batch_size=config.federated.batch_size,
            lr=config.optimization.learning_rate,
            optimizer_name=config.optimization.optimizer,
        )
        res = baseline.evaluate(
            test_open_ds=test_open_ds,
            test_closed_ds=test_closed_ds,
            cal_ds=cal_ds,
            conformal_alpha=config.osr.conformal_alpha,
            scoring_method=scoring_method,
            use_mondrian=use_mondrian,
        )
        if eval_mode == "closed_set":
            return {
                "accuracy": res["accuracy"],
                "balanced_accuracy": res["balanced_accuracy"],
                "macro_f1": res["closed_set_macro_f1"],
            }
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
    for r in range(1, rounds + 1):
        server.run_round(r)

    # Freeze encoder permanently (Section 4.1 & 4.3)
    server.freeze_global_encoder()

    num_known = len(config.data.known_classes)

    if eval_mode == "closed_set":
        # Closed-Set Classification using Stage 2/3 Mahalanobis profiles (since g_psi is discarded after Stage 1)
        server.run_phase_two_profiling(
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
            alpha=config.osr.conformal_alpha,
        )
        edge_detector = server.get_edge_detector(
            unknown_label=num_known,
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
        )
        _, _, preds_closed, _ = edge_detector.predict_batch(test_closed_ds.features)

        y_closed_true = test_closed_ds.labels.numpy()
        closed_metrics = compute_classification_metrics(
            y_true=y_closed_true,
            y_pred=preds_closed,
            num_classes=num_known,
        )
        return closed_metrics

    osr_method = getattr(config, "osr_method", None)
    if osr_method is None and hasattr(config, "osr"):
        osr_method = getattr(config.osr, "osr_method", None)

    if osr_method in ("maximum_softmax_probability", "msp"):
        server.global_model.eval()
        with torch.no_grad():
            x_cal = cal_ds.features.to(device)
            logits_cal = server.global_model.classify(x_cal)
            probs_cal = torch.softmax(logits_cal, dim=-1)
            msp_cal_scores = (1.0 - torch.max(probs_cal, dim=-1)[0]).cpu().numpy()
            tau_msp = float(np.percentile(msp_cal_scores, 100.0 * (1.0 - config.osr.conformal_alpha)))

            x_open = test_open_ds.features.to(device)
            logits_open = server.global_model.classify(x_open)
            probs_open = torch.softmax(logits_open, dim=-1)
            max_prob, cand_class = torch.max(probs_open, dim=-1)
            anomaly_scores = (1.0 - max_prob).cpu().numpy()
            cand_class_np = cand_class.cpu().numpy()
            preds_open = np.where(anomaly_scores > tau_msp, num_known, cand_class_np)

            x_closed = test_closed_ds.features.to(device)
            logits_closed = server.global_model.classify(x_closed)
            preds_closed = torch.argmax(logits_closed, dim=-1).cpu().numpy()

    elif osr_method == "energy_ood":
        server.global_model.eval()
        temp = getattr(config, "energy_temperature", 1.0)
        with torch.no_grad():
            x_cal = cal_ds.features.to(device)
            logits_cal = server.global_model.classify(x_cal)
            energy_cal = (-temp * torch.logsumexp(logits_cal / temp, dim=-1)).cpu().numpy()
            tau_energy = float(np.percentile(energy_cal, 100.0 * (1.0 - config.osr.conformal_alpha)))

            x_open = test_open_ds.features.to(device)
            logits_open = server.global_model.classify(x_open)
            energy_open = (-temp * torch.logsumexp(logits_open / temp, dim=-1)).cpu().numpy()
            cand_class = torch.argmax(logits_open, dim=-1).cpu().numpy()
            anomaly_scores = energy_open
            preds_open = np.where(anomaly_scores > tau_energy, num_known, cand_class)

            x_closed = test_closed_ds.features.to(device)
            logits_closed = server.global_model.classify(x_closed)
            preds_closed = torch.argmax(logits_closed, dim=-1).cpu().numpy()

    elif osr_method == "weibull_openmax":
        from src.osr.weibull_evt import WeibullEVTFitter, compute_evt_probabilities
        server.global_model.eval()
        with torch.no_grad():
            x_cal = cal_ds.features.to(device)
            y_cal = cal_ds.labels
            _, _, _, mu_cal, _ = server.global_model(x_cal)
            prototypes = {}
            for c in range(num_known):
                idx_c = torch.where(y_cal == c)[0]
                if len(idx_c) > 0:
                    prototypes[c] = mu_cal[idx_c].mean(dim=0)
                else:
                    prototypes[c] = torch.zeros(mu_cal.size(1), device=device)

            tail_size = getattr(config, "openmax_tail_size", 20)
            fitter = WeibullEVTFitter(tail_size=tail_size)
            fitter.fit(x_cal, y_cal, prototypes, model=server.global_model)

            x_open = test_open_ds.features.to(device)
            logits_open, _, _, mu_open, _ = server.global_model(x_open)
            evt_probs = compute_evt_probabilities(mu_open, logits_open, prototypes, fitter.weibull_models)
            p_unk = evt_probs[:, 0].cpu().numpy()
            known_probs = evt_probs[:, 1:].cpu().numpy()
            cand_class = np.argmax(known_probs, axis=-1)
            anomaly_scores = p_unk
            preds_open = np.where(p_unk > 0.5, num_known, cand_class)

            x_closed = test_closed_ds.features.to(device)
            logits_closed = server.global_model.classify(x_closed)
            preds_closed = torch.argmax(logits_closed, dim=-1).cpu().numpy()

    else:
        # Phase II: Zero-Leakage Federated Class Profiling via Law of Total Variance
        server.run_phase_two_profiling(
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
            alpha=config.osr.conformal_alpha,
        )

        # Phase III: Wire-Speed Block-Diagonal Mahalanobis Edge Detector
        edge_detector = server.get_edge_detector(
            unknown_label=num_known,
            num_blocks=config.federated.num_blocks,
            block_dim=config.federated.block_dim,
        )

        # Evaluate on open-set and closed-set test data using Mahalanobis profiles (g_psi discarded)
        preds_open, is_unk, cand, anomaly_scores = edge_detector.predict_batch(test_open_ds.features)
        _, _, preds_closed, _ = edge_detector.predict_batch(test_closed_ds.features)

    y_open_true = test_open_ds.labels.numpy()
    metrics = compute_osr_metrics(
        y_true=y_open_true,
        y_pred=preds_open,
        anomaly_scores=anomaly_scores,
        num_known_classes=num_known,
        unknown_label=num_known,
    )

    y_closed_true = test_closed_ds.labels.numpy()
    closed_metrics = compute_classification_metrics(
        y_true=y_closed_true,
        y_pred=preds_closed,
        num_classes=num_known,
    )

    metrics["closed_set_accuracy"] = closed_metrics["accuracy"]
    metrics["accuracy"] = closed_metrics["accuracy"]
    metrics["closed_set_macro_f1"] = closed_metrics["macro_f1"]

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

