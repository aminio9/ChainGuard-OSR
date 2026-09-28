"""Script to generate and save synthetic P2P blockchain network flow dataset CSV."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from src.data.synthetic import generate_synthetic_p2p_data
from src.utils.io import ensure_dir


def main() -> None:
    output_dir = Path("data/synthetic")
    ensure_dir(output_dir)

    print("Generating synthetic P2P network traffic flows (10,000 samples, 43 features)...")
    dataset = generate_synthetic_p2p_data(
        num_samples=10000,
        in_features=43,
        num_classes=10,
        normal_ratio=0.6,
        seed=42,
    )

    features = dataset.features.numpy()
    labels = dataset.labels.numpy()

    cols = [f"flow_feature_{i:02d}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=cols)
    df["label"] = labels

    csv_path = output_dir / "synthetic_p2p.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")


if __name__ == "__main__":
    main()
