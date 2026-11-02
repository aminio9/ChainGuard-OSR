"""Master Orchestrator for Real Ground-Truth 100-Round 10-Client Benchmark on 100% Full B-NAT Data.

Executes:
1. Full 100 rounds of federated training across 10 nodes (Dirichlet beta=0.5).
2. Per-round telemetry logging (loss, validation accuracy, gradient tracking).
3. Checkpoint export: final_model.pt and global_profiles.pt.
4. Comprehensive multi-metric evaluation of Method 1, Method 2 (Proposed), and Baselines.
5. Automated generation and thesis export of publication-grade figures (PDF + 600 DPI PNG):
   - fig_e1_convergence (Loss & Val Accuracy)
   - 03_latent_space_geometry_tsne_pca (Multi-Prototype Normal Modes vs Attacks)
   - 04_score_distribution_and_pr (Conformal Score Density & Alpha Sensitivity)
   - 02_internal_non_iid_distribution_1x4 (Client Data Distribution)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.covariance import OAS
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.lines import Line2D

from src.config import load_config
from src.tracking.reproducibility import set_seed
from src.data.tabular_p2p import load_benchmark_dataset
from src.data.osr_split import split_open_set_dataset
from src.data.partitioning import partition_dirichlet_non_iid
from src.data.base import FlowDataset
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.aggregation.total_variance import (
    ClientClassMoment,
    GlobalClassProfile,
    compute_server_side_oas_shrinkage,
    federated_align_and_average_prototypes,
)
from src.osr.block_mahalanobis import (
    WireSpeedEdgeDetector,
    compute_block_mahalanobis_distances,
)
from src.osr.conformal import compute_right_closed_histogram
from src.metrics.osr_metrics import compute_osr_metrics, compute_fpr95


# ---------------------------------------------------------------------------
# Plotting Style Utilities
# ---------------------------------------------------------------------------
def configure_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10.0,
        "text.color": "#000000",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#000000",
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "axes.axisbelow": True,
        "axes.grid": True,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "grid.color": "#EAEAEA",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#000000",
        "legend.fontsize": 8.5,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def apply_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    seed = 42
    set_seed(seed)
    device = torch.device("cpu")

    num_clients = 10
    num_rounds = 100
    alpha = 0.05
    num_blocks = 4
    block_dim = 16

    # Setup Destination Directories
    out_dir = Path("D:/PAPER/Real_Run/experiment_10c_100r_full")
    ckpt_dir = out_dir / "checkpoints"
    telem_dir = out_dir / "telemetry"
    plots_dir = out_dir / "plots"
    thesis_figs_dir = Path("D:/Thesis/Latex/figs/chapter5")

    for d in [out_dir, ckpt_dir, telem_dir, plots_dir, thesis_figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("MASTER REAL-RUN ORCHESTRATOR: 10 CLIENTS | 100 ROUNDS | 100% B-NAT DATASET")
    print(f"Output Directory: {out_dir}")
    print(f"Thesis Figures Directory: {thesis_figs_dir}")
    print("=" * 95)

    # 1. Load 100% Full B-NAT Dataset
    print("\n[Stage 1/6] Loading 100% full B-NAT dataset from disk cache...")
    dataset, meta = load_benchmark_dataset(
        dataset_name="b_nat",
        data_dir="data/raw",
        samples_per_class=None,
    )
    in_features = meta["in_features"]
    known_classes = meta["known_classes"]
    unknown_classes = meta["unknown_classes"]
    num_known = len(known_classes)
    unknown_label = num_known

    print(f"  Total Flow Records: {len(dataset):,}")
    print(f"  Known Classes ({num_known}): {meta['known_class_names']}")
    print(f"  Zero-Day Sequestered Attack: {meta['unknown_class_names']}")

    # 2. Strict Open-Set Protocol Split (70% Train, 10% Cal, 20% Test)
    print("\n[Stage 2/6] Partitioning Open-Set protocol (70% Train, 10% Cal, 20% Test)...")
    train_ds, cal_ds, test_closed_ds, test_open_ds = split_open_set_dataset(
        dataset=dataset,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        test_ratio=0.2,
        cal_ratio=0.1,
        seed=seed,
    )
    y_test_open = test_open_ds.labels.cpu().numpy()
    y_test_closed = test_closed_ds.labels.cpu().numpy()

    print(f"  Train:       {len(train_ds):,} (Known only)")
    print(f"  Calibration: {len(cal_ds):,} (Known only)")
    print(f"  Closed Test: {len(test_closed_ds):,} (Known only)")
    print(f"  Open Test:   {len(test_open_ds):,} (Known + 15,000 Zero-Day FoT Attacks)")

    # 3. Simulate Dirichlet Non-IID Skew across 10 Clients (beta=0.5)
    print(f"\n[Stage 3/6] Partitioning training data across {num_clients} nodes (Dirichlet beta=0.5)...")
    client_map = partition_dirichlet_non_iid(
        dataset=train_ds,
        num_clients=num_clients,
        beta_dirichlet=0.5,
        seed=seed,
    )
    for cid in range(num_clients):
        print(f"  Node {cid:02d}: {len(client_map[cid]):,} samples")

    cfg = load_config("configs/default.yaml")
    cfg.data.in_features = in_features
    cfg.data.known_classes = known_classes
    cfg.data.unknown_classes = unknown_classes
    cfg.model.in_features = in_features
    cfg.model.latent_dim = 64
    cfg.model.hidden_dim = [128, 64]
    cfg.federated.num_clients = num_clients
    cfg.federated.num_rounds = num_rounds
    cfg.federated.local_epochs = 1
    cfg.federated.batch_size = 512
    cfg.osr.use_multi_prototype = True
    cfg.osr.num_prototypes_normal = 4
    cfg.osr.num_prototypes_attack = 1
    cfg.osr.num_blocks = num_blocks
    cfg.osr.block_dim = block_dim
    cfg.osr.conformal_alpha = alpha

    clients = []
    cal_len_per_client = len(cal_ds) // num_clients
    for cid in range(num_clients):
        sub_idx = client_map[cid]
        idx_t = torch.tensor(sub_idx, dtype=torch.long)
        sub_ds = FlowDataset(train_ds.features[idx_t], train_ds.labels[idx_t])
        cal_sub_idx = slice(cid * cal_len_per_client, (cid + 1) * cal_len_per_client)
        sub_cal_ds = FlowDataset(cal_ds.features[cal_sub_idx], cal_ds.labels[cal_sub_idx])
        c_obj = FederatedClient(cid, sub_ds, cfg, device, cal_dataset=sub_cal_ds)
        clients.append(c_obj)

    server = FederatedServer(cfg, clients, device)

    # 4. Federated Training for 100 Rounds with Telemetry Logging
    print(f"\n[Stage 4/6] Executing 100 Federated VIB Communication Rounds...")
    telemetry_file = telem_dir / "training_telemetry.jsonl"
    round_history = []

    # Validation subset for fast per-round validation tracking (2000 samples)
    val_subset_indices = torch.randperm(len(cal_ds))[:2000]
    val_sub_feats = cal_ds.features[val_subset_indices]
    val_sub_labels = cal_ds.labels[val_subset_indices].numpy()

    t_train_start = time.time()
    with open(telemetry_file, "w", encoding="utf-8") as f_telem:
        for r in range(1, num_rounds + 1):
            t_r0 = time.time()
            metrics = server.run_round(r)
            dur_r = time.time() - t_r0

            # Evaluate validation accuracy
            server.global_model.eval()
            with torch.no_grad():
                logits_val, _, _, _ = server.global_model(val_sub_feats)
                preds_val = torch.argmax(logits_val, dim=1).numpy()
                val_acc = float(np.mean(preds_val == val_sub_labels))

            record = {
                "round": r,
                "duration_sec": float(dur_r),
                "loss_total": float(metrics.get("loss_total", 0.0)),
                "loss_bla": float(metrics.get("loss_bla", 0.0)),
                "loss_vib": float(metrics.get("loss_vib", 0.0)),
                "val_closed_accuracy": float(val_acc),
            }
            round_history.append(record)
            f_telem.write(json.dumps(record) + "\n")
            f_telem.flush()

            if r % 10 == 0 or r == 1 or r == 5:
                print(f"  Round {r:03d}/{num_rounds:03d} [{dur_r:.2f}s] | Loss: {record['loss_total']:.4f} (BLA: {record['loss_bla']:.4f}, VIB: {record['loss_vib']:.4f}) | Val Acc: {val_acc*100:.2f}%")

    print(f"  100 rounds completed in {time.time() - t_train_start:.2f}s")
    server.freeze_global_encoder()

    # 5. Save Final Trained Neural Network
    print("\n[Stage 5/6] Exporting Trained Model Checkpoints & Global Statistical Profiles...")
    model_save_path = ckpt_dir / "final_model.pt"
    torch.save(server.global_model.state_dict(), model_save_path)
    print(f"  Model state dict saved to: {model_save_path}")

    # Phase 2 Profiling: Wang 2025 One-Shot Ref Centers + Voronoi LOTV + OAS Shrinkage
    m2_profiles = server.run_phase_two_profiling(
        num_blocks=num_blocks,
        block_dim=block_dim,
        alpha=alpha,
    )
    profiles_save_path = ckpt_dir / "global_profiles.pt"
    torch.save(m2_profiles, profiles_save_path)
    print(f"  Global statistical profiles saved to: {profiles_save_path}")

    # 6. Comprehensive Multi-Method Evaluation
    print("\n[Stage 6/6] Executing Multi-Method Evaluation across 100% test partitions...")

    def evaluate_detector(preds_open, scores_open, preds_closed):
        osr_m = compute_osr_metrics(
            y_true=y_test_open,
            y_pred=preds_open,
            anomaly_scores=scores_open,
            num_known_classes=num_known,
            unknown_label=unknown_label,
        )
        c_acc = float(np.mean(preds_closed == y_test_closed)) * 100.0
        is_unk_true = (y_test_open >= num_known).astype(int)
        fpr_95 = compute_fpr95(is_unk_true, scores_open) * 100.0
        return {
            "c_acc": c_acc,
            "auroc": osr_m["auroc"] * 100.0,
            "aupr": osr_m["aupr"] * 100.0,
            "fpr95": fpr_95,
            "unk_rec": osr_m["unknown_recall"] * 100.0,
            "unk_prec": osr_m["unknown_precision"] * 100.0,
            "unk_f1": osr_m["unknown_f1"] * 100.0,
            "os_macro_f1": osr_m["open_set_macro_f1"] * 100.0,
            "kfr": osr_m["kfr"] * 100.0,
            "coverage": (1.0 - osr_m["kfr"]) * 100.0,
        }

    all_results = {}

    # --- Method 2: Proposed FedTROS-MC ---
    edge_det = server.get_edge_detector(
        unknown_label=unknown_label,
        num_blocks=num_blocks,
        block_dim=block_dim,
    )
    preds_closed_m2, _, _, _ = edge_det.predict_batch(test_closed_ds.features)
    preds_open_m2, _, _, scores_open_m2 = edge_det.predict_batch(test_open_ds.features)
    all_results["Method 2 (Proposed: Wang Voronoi + Conformal Set)"] = evaluate_detector(
        preds_open=preds_open_m2,
        scores_open=scores_open_m2,
        preds_closed=preds_closed_m2,
    )

    # --- Method 1: Previous FedTROS-MC (حالت ب) ---
    m1_class_moments = {c: [] for c in known_classes}
    for client in clients:
        for c in client.local_classes:
            c = int(c)
            idx = (client.labels_dev == c)
            cnt = int(idx.sum().item())
            if cnt < 3:
                continue
            z_c = client.features_dev[idx]
            z_all = server.global_model.encoder(z_c)
            z_emb = z_all[2] if isinstance(z_all, tuple) and len(z_all) >= 3 else z_all[0]

            mu_c = torch.mean(z_emb, dim=0)
            target_m = 4 if c == 0 else 1
            local_p = None
            local_counts = None
            assigned_c = mu_c.unsqueeze(0).expand(cnt, -1)

            if target_m > 1 and cnt >= target_m:
                km_sample = z_emb.cpu().numpy()
                if len(km_sample) > 5000:
                    km_indices = np.random.choice(len(km_sample), 5000, replace=False)
                    km_sample = km_sample[km_indices]
                km = KMeans(n_clusters=target_m, random_state=42, n_init=3)
                km.fit(km_sample)
                local_p = torch.from_numpy(km.cluster_centers_).to(dtype=z_emb.dtype)
                d_km = torch.cdist(z_emb, local_p)
                lbls = torch.argmin(d_km, dim=1)
                assigned_c = local_p[lbls]
                counts_arr = np.bincount(lbls.cpu().numpy(), minlength=target_m)
                local_counts = [int(x) for x in counts_arr]
            else:
                local_p = mu_c.unsqueeze(0)
                local_counts = [cnt]

            b_covs = []
            for b in range(num_blocks):
                s_idx = b * block_dim
                e_idx = s_idx + block_dim
                diff_b = z_emb[:, s_idx:e_idx] - assigned_c[:, s_idx:e_idx]
                cov_b = torch.matmul(diff_b.t(), diff_b) / float(cnt)
                b_covs.append(cov_b.cpu())

            m1_class_moments[c].append(
                ClientClassMoment(
                    client_id=client.client_id,
                    sample_count=cnt,
                    mu=mu_c.cpu(),
                    block_covs=b_covs,
                    class_id=c,
                    local_prototypes=local_p,
                    local_proto_counts=local_counts,
                    cluster_id=0,
                )
            )

    m1_profiles = {}
    for c, subs in m1_class_moments.items():
        if not subs:
            continue
        tot_n = sum(s.sample_count for s in subs)
        w_mu = sum(float(s.sample_count) * s.mu for s in subs) / float(tot_n)
        b_covs_glob = []
        inv_b_glob = []
        for b in range(num_blocks):
            s_idx = b * block_dim
            e_idx = s_idx + block_dim
            pooled_b = torch.zeros((block_dim, block_dim))
            for s in subs:
                n_k = float(s.sample_count)
                mean_diff = (s.mu[s_idx:e_idx] - w_mu[s_idx:e_idx]).unsqueeze(1)
                disp_b = torch.matmul(mean_diff, mean_diff.t())
                pooled_b += n_k * (s.block_covs[b] + disp_b)
            raw_cov = pooled_b / float(tot_n)
            shrunk, _ = compute_server_side_oas_shrinkage(raw_cov, tot_n, tikhonov_eps=1e-4)
            b_covs_glob.append(shrunk)
            inv_b_glob.append(torch.linalg.pinv(shrunk))

        protos_c = None
        if c == 0:
            local_subs = [(s.local_prototypes, s.local_proto_counts) for s in subs if s.local_prototypes is not None]
            protos_c = federated_align_and_average_prototypes(local_subs, target_num_prototypes=4, fallback_mean=w_mu)

        m1_profiles[c] = GlobalClassProfile(
            class_id=c,
            total_samples=tot_n,
            mu=w_mu,
            block_covs=b_covs_glob,
            inv_block_covs=inv_b_glob,
            prototypes=protos_c,
            proto_inv_block_covs=[inv_b_glob for _ in range(protos_c.size(0))] if protos_c is not None else None,
        )

    # Calibration M1
    m1_cal_scores = {c: [] for c in known_classes}
    for client in clients:
        if client.cal_dataset is None or len(client.cal_dataset) == 0:
            continue
        z_cal = server.global_model.encoder(client.cal_dataset.features)[2]
        dists = compute_block_mahalanobis_distances(z_cal, m1_profiles, num_blocks=num_blocks, block_dim=block_dim)
        for c in known_classes:
            mask_c = (client.cal_dataset.labels == c)
            if mask_c.sum().item() > 0:
                m1_cal_scores[c].append(dists[c][mask_c].detach().cpu().numpy())

    for c in known_classes:
        arrs = m1_cal_scores[c]
        if arrs:
            pooled_s = np.concatenate(arrs)
            s_max = max(float(np.max(pooled_s)), 1e-4)
            edges = np.linspace(0.0, s_max, 2048 + 1)
            hists = [compute_right_closed_histogram(a, edges, include_overflow=False) for a in arrs]
            total_c = np.sum([cnt for cnt, _ in hists], axis=0)
            n_tot = np.sum(total_c)
            r_c = np.ceil((n_tot + 1) * (1 - alpha))
            idx = int(np.searchsorted(np.cumsum(total_c), r_c, side="left"))
            tau = float(edges[min(idx + 1, len(edges) - 1)])
        else:
            tau = 25.0
        m1_profiles[c] = m1_profiles[c]._replace(conformal_threshold=tau)

    z_test_closed = server.global_model.encoder(test_closed_ds.features)[2]
    d_m1_closed = compute_block_mahalanobis_distances(z_test_closed, m1_profiles, num_blocks=num_blocks, block_dim=block_dim)
    preds_closed_m1 = torch.argmin(torch.stack([d_m1_closed[c] for c in known_classes], dim=1), dim=1).cpu().numpy()

    z_test_open = server.global_model.encoder(test_open_ds.features)[2]
    dists_m1 = compute_block_mahalanobis_distances(z_test_open, m1_profiles, num_blocks=num_blocks, block_dim=block_dim)
    dist_mat_m1 = torch.stack([dists_m1[c] for c in known_classes], dim=1)
    tau_tensor_m1 = torch.tensor([m1_profiles[c].conformal_threshold for c in known_classes])

    best_raw_idx = torch.argmin(dist_mat_m1, dim=1)
    cand_dist_m1 = torch.gather(dist_mat_m1, 1, best_raw_idx.unsqueeze(1)).squeeze(1)
    cand_tau_m1 = tau_tensor_m1[best_raw_idx]
    preds_open_m1 = torch.where(cand_dist_m1 > cand_tau_m1, torch.tensor(unknown_label), best_raw_idx).cpu().numpy()

    all_results["Method 1 (Previous: Blind KMeans + Raw Argmin)"] = evaluate_detector(
        preds_open=preds_open_m1,
        scores_open=cand_dist_m1.cpu().numpy(),
        preds_closed=preds_closed_m1,
    )

    # --- Baseline 1: FedAvg + MSP ---
    with torch.no_grad():
        logits_closed = server.global_model.classifier(z_test_closed)
        preds_closed_msp = torch.argmax(logits_closed, dim=1).cpu().numpy()

        logits_open = server.global_model.classifier(z_test_open)
        probs_open = torch.softmax(logits_open, dim=1)
        max_prob, cand_cls = torch.max(probs_open, dim=1)
        msp_scores = (1.0 - max_prob).cpu().numpy()

        z_cal_all = server.global_model.encoder(cal_ds.features)[2]
        probs_cal = torch.softmax(server.global_model.classifier(z_cal_all), dim=1)
        max_prob_cal, _ = torch.max(probs_cal, dim=1)
        tau_msp = float(np.quantile((1.0 - max_prob_cal).cpu().numpy(), 1.0 - alpha))
        preds_open_msp = torch.where(
            torch.from_numpy(msp_scores) > tau_msp,
            torch.tensor(unknown_label),
            cand_cls,
        ).cpu().numpy()

    all_results["FedAvg + MSP (Hendrycks 2017)"] = evaluate_detector(
        preds_open=preds_open_msp,
        scores_open=msp_scores,
        preds_closed=preds_closed_msp,
    )

    # --- Baseline 2: FedAvg + Energy-OOD ---
    with torch.no_grad():
        T_energy = 1.0
        energy_open = -T_energy * torch.logsumexp(logits_open / T_energy, dim=1).cpu().numpy()
        energy_cal = -T_energy * torch.logsumexp(server.global_model.classifier(z_cal_all) / T_energy, dim=1).cpu().numpy()
        tau_energy = float(np.quantile(energy_cal, 1.0 - alpha))
        preds_open_energy = torch.where(
            torch.from_numpy(energy_open) > tau_energy,
            torch.tensor(unknown_label),
            cand_cls,
        ).cpu().numpy()

    all_results["FedAvg + Energy-OOD (Liu 2020)"] = evaluate_detector(
        preds_open=preds_open_energy,
        scores_open=energy_open,
        preds_closed=preds_closed_msp,
    )

    # --- Baseline 3: FedProto ---
    with torch.no_grad():
        proto_list = []
        for c in range(num_known):
            c_mask = (train_ds.labels == c)
            if c_mask.sum() > 0:
                z_c = server.global_model.encoder(train_ds.features[c_mask])[2]
                proto_list.append(torch.mean(z_c, dim=0))
            else:
                proto_list.append(torch.zeros(64))
        protos = torch.stack(proto_list, dim=0)

        d_proto_closed = torch.cdist(z_test_closed, protos)
        preds_closed_proto = torch.argmin(d_proto_closed, dim=1).cpu().numpy()

        d_proto_open = torch.cdist(z_test_open, protos)
        min_d_open, proto_cand = torch.min(d_proto_open, dim=1)
        d_open_np = min_d_open.cpu().numpy()

        d_proto_cal = torch.cdist(z_cal_all, protos)
        min_d_cal, _ = torch.min(d_proto_cal, dim=1)
        tau_proto = float(np.quantile(min_d_cal.cpu().numpy(), 1.0 - alpha))

        preds_open_proto = torch.where(
            min_d_open > tau_proto,
            torch.tensor(unknown_label),
            proto_cand,
        ).cpu().numpy()

    all_results["FedProto (Tan et al. 2022)"] = evaluate_detector(
        preds_open=preds_open_proto,
        scores_open=d_open_np,
        preds_closed=preds_closed_proto,
    )

    # Save Evaluation Results
    eval_json_path = out_dir / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full evaluation JSON written to: {eval_json_path}")

    # Print Table
    print("\n" + "=" * 125)
    print(f"{'EVALUATED METHOD':<42} | {'C-Acc':<7} | {'AUROC':<7} | {'AUPR':<7} | {'FPR95':<7} | {'OS-F1':<7} | {'ZD-Rec':<7} | {'ZD-Prec':<7} | {'ZD-F1':<7} | {'KFR':<6} | {'Cov':<6}")
    print("=" * 125)
    for m_name, res in all_results.items():
        print(
            f"{m_name:<42} | "
            f"{res['c_acc']:>6.2f}% | "
            f"{res['auroc']:>6.2f}% | "
            f"{res['aupr']:>6.2f}% | "
            f"{res['fpr95']:>6.2f}% | "
            f"{res['os_macro_f1']:>6.2f}% | "
            f"{res['unk_rec']:>6.2f}% | "
            f"{res['unk_prec']:>6.2f}% | "
            f"{res['unk_f1']:>6.2f}% | "
            f"{res['kfr']:>5.2f}% | "
            f"{res['coverage']:>5.2f}%"
        )
    print("=" * 125)

    # ---------------------------------------------------------------------------
    # AUTOMATED GENERATION OF PUBLICATION-GRADE FIGURES
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("GENERATING PUBLICATION-GRADE FIGURES (600 DPI PNG & VECTOR PDF)")
    print("-" * 75)
    configure_plot_style()

    # --- Figure 1: Convergence Trajectory (fig_e1_convergence) ---
    print("  [Plot 1/4] Generating Convergence Trajectory Plot (fig_e1_convergence)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.85), layout="constrained")
    acc_rounds = [h["round"] for h in round_history]
    acc_values = [h["val_closed_accuracy"] * 100.0 for h in round_history]
    loss_rounds = [h["round"] for h in round_history]
    loss_total = [h["loss_total"] for h in round_history]

    color_purple = "#7B1FA2"
    ax1.plot(acc_rounds, acc_values, color=color_purple, lw=1.8, label="Validation Accuracy")
    ax1.annotate(f"{acc_values[-1]:.2f}%", xy=(acc_rounds[-1], acc_values[-1]), xytext=(acc_rounds[-1] - 18, acc_values[-1] + 0.35), fontsize=8.5, color=color_purple)
    ax1.set_title("(a) Validation Accuracy vs. Rounds", loc="center")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_xlim(0, num_rounds + 2)
    ax1.legend(loc="lower right")
    apply_spines(ax1)

    ax2.plot(loss_rounds, loss_total, color=color_purple, lw=1.8, label="Total Loss (BLA+VIB)")
    ax2.annotate(f"{loss_total[-1]:.4f}", xy=(loss_rounds[-1], loss_total[-1]), xytext=(loss_rounds[-1] - 18, loss_total[-1] + 0.05), fontsize=8.5, color=color_purple)
    ax2.set_title("(b) Training Loss Trajectory", loc="center")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Loss")
    ax2.set_xlim(0, num_rounds + 2)
    ax2.legend(loc="upper right")
    apply_spines(ax2)

    for p_dest in [plots_dir / "fig_e1_convergence.pdf", plots_dir / "fig_e1_convergence.png", thesis_figs_dir / "fig_e1_convergence.pdf", thesis_figs_dir / "fig_e1_convergence.png"]:
        fig.savefig(str(p_dest))
    plt.close(fig)
    print("    -> Exported: fig_e1_convergence.pdf / .png")

    # --- Figure 2: Latent Space Geometry (03_latent_space_geometry_tsne_pca) ---
    print("  [Plot 2/4] Generating Latent Space Geometry (03_latent_space_geometry_tsne_pca)...")
    # Subsample 1500 test points for clean t-SNE visualization
    tsne_idx = np.random.choice(len(test_open_ds), 1500, replace=False)
    z_sub = z_test_open[tsne_idx].cpu().numpy()
    y_sub = y_test_open[tsne_idx]

    tsne_iso = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=800)
    z_2d_iso = tsne_iso.fit_transform(z_sub)

    # Metric-aligned transform
    mask_kn = (y_sub < num_known)
    oas_fit = OAS().fit(z_sub[mask_kn])
    vals, vecs = np.linalg.eigh(oas_fit.covariance_)
    whitener = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-4))) @ vecs.T
    z_w = (z_sub - oas_fit.location_) @ whitener
    dist_m = cdist(z_w, z_w, metric="euclidean")
    tsne_metric = TSNE(n_components=2, metric="precomputed", perplexity=30, random_state=42, init="random")
    z_2d_metric = tsne_metric.fit_transform(dist_m)

    fig_geo, (gax1, gax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), layout="constrained")

    # Colors for classes
    palette = ["#1565C0", "#00897B", "#D81B60", "#F57C00", "#7B1FA2"]
    class_names = ["Normal (Multi-Mode)", "BP Attack", "DoS Attack", "MitM Attack", "FoT Zero-Day (Unknown)"]

    for c in range(num_known + 1):
        c_mask = (y_sub == c)
        if c_mask.sum() == 0:
            continue
        gax1.scatter(z_2d_iso[c_mask, 0], z_2d_iso[c_mask, 1], s=18, color=palette[c], alpha=0.65, label=class_names[c])
        gax2.scatter(z_2d_metric[c_mask, 0], z_2d_metric[c_mask, 1], s=18, color=palette[c], alpha=0.65, label=class_names[c])

    gax1.set_title("(a) Isotropic Space: Overlapping Boundaries (FedProto Baseline)")
    gax1.set_xlabel("t-SNE Dimension 1")
    gax1.set_ylabel("t-SNE Dimension 2")
    apply_spines(gax1)

    gax2.set_title("(b) Proposed Metric-Aligned Space: Conformal Mode Separation")
    gax2.set_xlabel("t-SNE Dimension 1")
    gax2.set_ylabel("t-SNE Dimension 2")
    gax2.legend(loc="upper right", fontsize=8.0)
    apply_spines(gax2)

    for p_dest in [plots_dir / "03_latent_space_geometry_tsne_pca.pdf", plots_dir / "03_latent_space_geometry_tsne_pca.png", thesis_figs_dir / "03_latent_space_geometry_tsne_pca.pdf", thesis_figs_dir / "03_latent_space_geometry_tsne_pca.png"]:
        fig_geo.savefig(str(p_dest))
    plt.close(fig_geo)
    print("    -> Exported: 03_latent_space_geometry_tsne_pca.pdf / .png")

    # --- Figure 3: Score Distribution & Alpha Sensitivity (04_score_distribution_and_pr) ---
    print("  [Plot 3/4] Generating Score Distribution & PR Trade-off (04_score_distribution_and_pr)...")
    known_mask_open = (y_test_open < num_known)
    unk_mask_open = (y_test_open >= num_known)
    known_scores = scores_open_m2[known_mask_open]
    unk_scores = scores_open_m2[unk_mask_open]

    # Evaluate sweep
    alphas_sweep = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    rec_list = []
    f1_list = []
    kfr_list = []
    for a in alphas_sweep:
        tau_a = np.quantile(known_scores, 1.0 - a)
        pred_unk_a = (scores_open_m2 > tau_a)
        tp = np.sum(pred_unk_a & unk_mask_open)
        rec = float(tp / np.sum(unk_mask_open))
        prec = float(tp / np.sum(pred_unk_a)) if np.sum(pred_unk_a) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        kfr = float(np.mean(pred_unk_a[known_mask_open]))
        rec_list.append(rec * 100)
        f1_list.append(f1 * 100)
        kfr_list.append(kfr * 100)

    fig_score, (sax1, sax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")

    # Histogram Density
    max_s = min(float(np.percentile(scores_open_m2, 99)), 4.0)
    bins = np.linspace(0.0, max_s, 60)
    sax1.hist(known_scores, bins=bins, density=True, color="#1976D2", alpha=0.6, label=r"Known Traffic ($D_c \leq \hat{q}_{\alpha}$)")
    sax1.hist(unk_scores, bins=bins, density=True, color="#D32F2F", alpha=0.6, label=r"Zero-Day Attacks ($D_c > \hat{q}_{\alpha}$)")
    tau_nominal = float(np.quantile(known_scores, 1.0 - alpha))
    sax1.axvline(tau_nominal, color="#000000", linestyle="--", lw=1.8, label=f"Conformal Cutoff $\\hat{{q}}_{{0.05}}={tau_nominal:.2f}$")
    sax1.set_title("(a) Nonconformity Score Density")
    sax1.set_xlabel("Normalized Mahalanobis Score ($D_c / \\hat{q}_c$)")
    sax1.set_ylabel("Empirical Probability Density")
    sax1.legend(loc="upper right", fontsize=8.0)
    apply_spines(sax1)

    # Sensitivity Sweep
    alphas_pct = [a * 100 for a in alphas_sweep]
    sax2.plot(alphas_pct, rec_list, color="#D32F2F", marker="o", lw=1.8, label="Zero-Day Recall (%)")
    sax2.plot(alphas_pct, f1_list, color="#7B1FA2", marker="s", lw=1.8, label="Zero-Day F1-Score (%)")
    sax2.plot(alphas_pct, kfr_list, color="#1976D2", marker="^", lw=1.8, label="Empirical KFR (%)")
    sax2.plot(alphas_pct, alphas_pct, color="#000000", linestyle=":", lw=1.2, label="Nominal Budget $\\alpha$")
    sax2.set_title("(b) Conformal Error Budget Sensitivity")
    sax2.set_xlabel("Significance Level $\\alpha$ (%)")
    sax2.set_ylabel("Metric Value (%)")
    sax2.legend(loc="center right", fontsize=8.0)
    apply_spines(sax2)

    for p_dest in [plots_dir / "04_score_distribution_and_pr.pdf", plots_dir / "04_score_distribution_and_pr.png", thesis_figs_dir / "04_score_distribution_and_pr.pdf", thesis_figs_dir / "04_score_distribution_and_pr.png"]:
        fig_score.savefig(str(p_dest))
    plt.close(fig_score)
    print("    -> Exported: 04_score_distribution_and_pr.pdf / .png")

    # --- Figure 4: Client Non-IID Distribution (02_internal_non_iid_distribution_1x4) ---
    print("  [Plot 4/4] Generating Client Data Distribution (02_internal_non_iid_distribution_1x4)...")
    fig_dist, ax_dist = plt.subplots(figsize=(8.0, 3.8), layout="constrained")
    client_counts = np.zeros((num_clients, num_known))
    for cid in range(num_clients):
        sub_labs = train_ds.labels[client_map[cid]].numpy()
        for c in range(num_known):
            client_counts[cid, c] = np.sum(sub_labs == c)

    bottom_vals = np.zeros(num_clients)
    palette_c = ["#1565C0", "#00897B", "#D81B60", "#F57C00"]
    for c in range(num_known):
        ax_dist.bar(range(num_clients), client_counts[:, c], bottom=bottom_vals, color=palette_c[c], label=meta['known_class_names'][c], edgecolor="#FFFFFF", lw=0.5)
        bottom_vals += client_counts[:, c]

    ax_dist.set_title("Dirichlet Heterogeneous Partition across 10 Blockchain Nodes ($\beta = 0.5$)")
    ax_dist.set_xlabel("Blockchain Validator Node ID")
    ax_dist.set_ylabel("Number of Flow Samples")
    ax_dist.set_xticks(range(num_clients))
    ax_dist.set_xticklabels([f"Node {i}" for i in range(num_clients)])
    ax_dist.legend(loc="upper right")
    apply_spines(ax_dist)

    for p_dest in [plots_dir / "02_internal_non_iid_distribution_1x4.png", thesis_figs_dir / "02_internal_non_iid_distribution_1x4.png"]:
        fig_dist.savefig(str(p_dest), dpi=300)
    plt.close(fig_dist)
    print("    -> Exported: 02_internal_non_iid_distribution_1x4.png")

    # Final summary report
    print("\n" + "=" * 95)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY IN {time.time() - t_start:.2f} SECONDS")
    print(f"Checkpoints exported to: {ckpt_dir}")
    print(f"Plots saved to:          {plots_dir}")
    print(f"Thesis figures updated:  {thesis_figs_dir}")
    print("=" * 95)


if __name__ == "__main__":
    main()
