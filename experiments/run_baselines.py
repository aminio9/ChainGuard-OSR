import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict
import torch

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.tracking.logger import setup_logger
from src.utils.device import get_device
from src.utils.io import save_json
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.osr.weibull_evt import WeibullEVTFitter
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.edge_inference import EdgeInferenceEngine
from src.metrics.osr_metrics import compute_osr_metrics
from src.visualization.latex_tables import generate_latex_benchmark_table


def evaluate_method_configuration(
    method_name: str,
    config_path: str,
    rounds: int = 15,
) -> Dict[str, float]:
    config = load_config(config_path)
    config.federated.num_rounds = rounds
    set_seed(config.experiment.seed)
    device = get_device(config.experiment.device)

    # Prepare data
    raw_dataset = generate_synthetic_p2p_data(
        num_samples=config.data.num_samples or 3000,
        in_features=config.data.in_features,
        num_classes=config.data.num_classes_total,
        seed=config.experiment.seed,
    )

    train_ds, cal_ds, _, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        test_ratio=config.data.test_split_ratio,
        cal_ratio=config.data.cal_split_ratio,
        seed=config.experiment.seed,
    )

    client_indices = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=config.federated.num_clients,
        beta_dirichlet=config.federated.partition.beta_dirichlet,
        seed=config.experiment.seed,
    )

    clients = []
    for client_id in range(config.federated.num_clients):
        sub_indices = client_indices[client_id]
        idx_t = torch.tensor(sub_indices, dtype=torch.long)
        sub_dataset = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        clients.append(FederatedClient(client_id, sub_dataset, config, device))

    server = FederatedServer(config, clients, device)

    # Train for rounds
    for r in range(1, rounds + 1):
        server.run_round(r)

    # Fit EVT Weibull
    weibull_fitter = WeibullEVTFitter(tail_size=config.osr.weibull_tail_size)
    weibull_fitter.fit(
        features=train_ds.features,
        labels=train_ds.labels,
        prototypes=server.global_prototypes,
        model=server.global_model,
    )

    # Calibrate Conformal
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=256, shuffle=False)
    server.global_model.eval()
    cal_probs_list, cal_labels_list = [], []
    with torch.no_grad():
        for x_b, y_b in cal_loader:
            logits, _, _, _, _ = server.global_model(x_b.to(device))
            cal_probs_list.append(torch.softmax(logits, dim=-1).cpu())
            cal_labels_list.append(y_b)

    all_cal_probs = torch.cat(cal_probs_list, dim=0)
    all_cal_labels = torch.cat(cal_labels_list, dim=0)

    conformal_calibrator = MondrianConformalCalibrator(alpha=config.osr.conformal_alpha)
    conformal_calibrator.calibrate(all_cal_probs, all_cal_labels, num_classes=len(config.data.known_classes))

    # Evaluate
    engine = EdgeInferenceEngine(
        model=server.global_model,
        prototypes=server.global_prototypes,
        weibull_models=weibull_fitter.weibull_models,
        conformal_calibrator=conformal_calibrator,
        unknown_class_id=len(config.data.known_classes),
        energy_temperature=config.osr.energy_temperature,
    )

    preds, _, _, unknown_probs = engine.predict_batch(test_open_ds.features)
    y_test_true = test_open_ds.labels.numpy()

    metrics = compute_osr_metrics(
        y_true=y_test_true,
        y_pred=preds,
        anomaly_scores=unknown_probs,
        num_known_classes=len(config.data.known_classes),
        unknown_label=len(config.data.known_classes),
    )
    return metrics


def run_all_baselines(rounds: int = 10, output_dir: str = "results/baselines") -> None:
    logger = setup_logger(name="baseline_runner")
    logger.info("Running complete baseline comparison suite...")

    methods = {
        "Full Method (Proposed)": "configs/default.yaml",
        "FedAvg + MSP": "configs/baselines/fedavg_msp.yaml",
        "FedProx + OpenMax": "configs/baselines/fedprox_openmax.yaml",
        "FedProto + Energy": "configs/baselines/fedproto_energy.yaml",
    }

    all_results = {}
    for name, cfg_path in methods.items():
        logger.info(f"--> Evaluating baseline: {name}")
        metrics = evaluate_method_configuration(name, cfg_path, rounds=rounds)
        all_results[name] = metrics
        logger.info(f"    AUROC: {metrics['auroc']*100:.2f}% | Macro-F1: {metrics['open_set_macro_f1']*100:.2f}% | FAR: {metrics['false_alarm_rate']*100:.2f}%")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_p / "baseline_comparison.json")

    # Generate and save LaTeX table
    latex_tbl = generate_latex_benchmark_table(all_results)
    with open(out_p / "table_baselines.tex", "w", encoding="utf-8") as f:
        f.write(latex_tbl)

    logger.info(f"Baseline benchmarking completed. Table written to {out_p / 'table_baselines.tex'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per baseline")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    args = parser.parse_args()

    run_all_baselines(rounds=args.rounds, output_dir=args.output_dir)
