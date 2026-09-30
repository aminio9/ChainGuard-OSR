import argparse
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import numpy as np

from src.config import load_config
from src.utils.device import get_device
from src.models.vib_network import VIBClassifierNetwork
from src.data.synthetic import generate_synthetic_p2p_data
from src.data.osr_split import split_open_set_dataset
from src.metrics.classification import compute_classification_metrics
from src.metrics.osr_metrics import compute_osr_metrics
from src.osr.block_mahalanobis import WireSpeedEdgeDetector


def evaluate_checkpoint(checkpoint_path: str, config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    device = get_device(config.experiment.device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VIBClassifierNetwork(
        in_features=config.model.in_features,
        latent_dim=config.model.latent_dim,
        num_classes=len(config.data.known_classes),
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        use_batchnorm=config.model.use_batchnorm,
        stochastic_encoder=config.model.stochastic_encoder,
    ).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "global_params" in ckpt:
        model.load_state_dict(ckpt["global_params"])

    # Generate or load test set
    raw_dataset = generate_synthetic_p2p_data(
        num_samples=2000,
        in_features=config.data.in_features,
        num_classes=config.data.num_classes_total,
        seed=config.experiment.seed,
    )

    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=raw_dataset,
        known_classes=config.data.known_classes,
        unknown_classes=config.data.unknown_classes,
        seed=config.experiment.seed,
    )

    model.eval()
    with torch.no_grad():
        logits = model.classify(test_closed_ds.features.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        closed_metrics = compute_classification_metrics(
            y_true=test_closed_ds.labels.numpy(),
            y_pred=preds,
            num_classes=len(config.data.known_classes),
        )

    print("\nCheckpoint Evaluation Results (Closed-Set):")
    for k, v in closed_metrics.items():
        print(f"  {k}: {v * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.config)
