import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch

from src.config import load_config
from src.utils.device import get_device
from src.utils.checkpoint import CheckpointManager
from src.models.dual_branch import DecoupledDualBranchNetwork
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.osr_split import split_open_set_dataset
from src.osr.weibull_evt import WeibullEVTFitter
from src.osr.conformal import MondrianConformalCalibrator
from src.osr.edge_inference import EdgeInferenceEngine
from src.metrics.osr_metrics import compute_osr_metrics


def evaluate_checkpoint(checkpoint_path: str, config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    device = get_device(config.experiment.device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DecoupledDualBranchNetwork(
        in_features=config.model.in_features,
        latent_dim=config.model.latent_dim,
        num_classes=len(config.data.known_classes),
        hidden_dim=config.model.hidden_dim,
        density_dim=config.model.density_dim,
        dropout=config.model.dropout,
        use_batchnorm=config.model.use_batchnorm,
        stochastic_encoder=config.model.stochastic_encoder,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    prototypes = ckpt.get("global_prototypes", {})

    # Generate test set
    raw_dataset = generate_synthetic_p2p_data(
        num_samples=2000,
        in_features=config.data.in_features,
        num_classes=config.data.num_classes_total,
        seed=config.experiment.seed,
    )

    train_ds, cal_ds, _, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        seed=config.experiment.seed,
    )

    # EVT
    weibull = WeibullEVTFitter(tail_size=config.osr.weibull_tail_size)
    weibull.fit(train_ds.features, train_ds.labels, prototypes, model=model)

    # Conformal
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=256, shuffle=False)
    model.eval()
    cal_probs = []
    cal_labels = []
    with torch.no_grad():
        for x_b, y_b in cal_loader:
            logits, _, _, _, _ = model(x_b.to(device))
            cal_probs.append(torch.softmax(logits, dim=-1).cpu())
            cal_labels.append(y_b)

    conformal = MondrianConformalCalibrator(alpha=config.osr.conformal_alpha)
    conformal.calibrate(torch.cat(cal_probs), torch.cat(cal_labels))

    engine = EdgeInferenceEngine(
        model=model,
        prototypes=prototypes,
        weibull_models=weibull.weibull_models,
        conformal_calibrator=conformal,
        unknown_class_id=len(config.data.known_classes),
    )

    preds, _, _, scores = engine.predict_batch(test_open_ds.features)
    metrics = compute_osr_metrics(
        y_true=test_open_ds.labels.numpy(),
        y_pred=preds,
        anomaly_scores=scores,
        num_known_classes=len(config.data.known_classes),
        unknown_label=len(config.data.known_classes),
    )

    print("\nCheckpoint Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.config)
