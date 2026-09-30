"""Additional real-data figure generators for make_plots.py (reads only real experiment files)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    COLORS,
    HEATMAP_CMAP,
    apply_academic_spines,
    auto_contrast_text_color,
    save_dual_figure,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_extended_real_figures(results_root: Path, figures_dir: Path, manifest: Dict[str, Any]) -> None:
    """Generate remaining paper figures strictly from real experiment files under results_root."""
    # 02 & 32: Convergence & Training Loss Decomposition from round_history.csv
    seed_dirs = sorted((results_root / "E2_E3_benchmarks" / "b_nat").glob("seed_*"))
    if seed_dirs:
        rh_path = seed_dirs[0] / "round_history.csv"
        if rh_path.exists():
            rows = _read_csv(rh_path)
            rounds = [int(float(r["round"])) for r in rows]
            val_acc = [float(r.get("val_closed_accuracy", 0.0)) * 100.0 for r in rows]
            total_loss = [float(r.get("loss_total", 0.0)) for r in rows]
            bla_loss = [float(r.get("loss_bla", 0.0)) for r in rows]
            vib_loss = [float(r.get("loss_vib", 0.0)) for r in rows]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
            ax1.plot(rounds, val_acc, marker="o", color=COLORS["ours"], label="FedTROS-MC")
            ax1.set_xlabel("Communication Round $t$")
            ax1.set_ylabel("Closed-Set Accuracy (%)")
            ax1.set_title("Global Validation Accuracy")
            ax1.legend(frameon=True)
            apply_academic_spines(ax1)

            ax2.plot(rounds, total_loss, marker="s", color=COLORS["ours"], label=r"Total $\mathcal{L}$")
            ax2.plot(rounds, bla_loss, linestyle="--", color=COLORS["fedavg"], label=r"$\mathcal{L}_{\mathrm{BLA}}$")
            ax2.plot(rounds, vib_loss, linestyle=":", color=COLORS["fedproto"], label=r"$\mathcal{L}_{\mathrm{VIB}}$")
            ax2.set_xlabel("Communication Round $t$")
            ax2.set_ylabel("Loss")
            ax2.set_title("Phase I Loss Trajectory")
            ax2.legend(frameon=True)
            apply_academic_spines(ax2)
            fig.tight_layout()
            out = save_dual_figure(fig, figures_dir / "02_global_convergence_curves")
            manifest["02_global_convergence_curves"] = {"sources": [str(rh_path)], "outputs": out}

            fig2, ax = plt.subplots(figsize=(7.4, 4.2))
            ax.plot(rounds, total_loss, linewidth=2.2, color=COLORS["ours"], label=r"Total $\mathcal{L}_{\mathrm{total}}$")
            ax.plot(rounds, bla_loss, linewidth=1.8, color=COLORS["fedavg"], label=r"Logit-Adjusted $\mathcal{L}_{\mathrm{BLA}}$")
            ax.plot(rounds, vib_loss, linewidth=1.8, color=COLORS["fedproto"], label=r"KL Information Bottleneck $\mathcal{L}_{\mathrm{VIB}}$")
            ax.set_xlabel("Communication Round $t$")
            ax.set_ylabel("Empirical Training Loss")
            ax.set_title("32: Phase I Training Loss Decomposition")
            ax.legend(frameon=True)
            apply_academic_spines(ax)
            fig2.tight_layout()
            out32 = save_dual_figure(fig2, figures_dir / "32_training_loss_decomposition")
            manifest["32_training_loss_decomposition"] = {"sources": [str(rh_path)], "outputs": out32}

        # 12: Latent Prototype Geometry from diagnostics.npz
        diag_path = seed_dirs[0] / "diagnostics.npz"
        if diag_path.exists():
            npz = np.load(diag_path)
            z = npz["latent_z"]
            y = npz["y_open_subset"]
            # 2D PCA projection of real latent vectors
            z_centered = z - np.mean(z, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
            z_2d = z_centered @ vt[:2].T
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            for cls_id in np.unique(y):
                mask = (y == cls_id)
                lbl = "Unknown Attacks" if cls_id == np.max(y) else f"Known C{cls_id}"
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=14, alpha=0.65, label=lbl)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("12: Real Latent Representation Geometry (PCA Projection)")
            ax.legend(frameon=True, fontsize=8)
            apply_academic_spines(ax)
            fig.tight_layout()
            out12 = save_dual_figure(fig, figures_dir / "12_latent_prototype_geometry")
            manifest["12_latent_prototype_geometry"] = {"sources": [str(diag_path)], "outputs": out12}

    # 11: A2 VIB Hyperparameter Sensitivity
    vib_vals = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    v_labels, v_auroc, v_f1, v_sources = [], [], [], []
    for bv in vib_vals:
        sf = results_root / f"A2_vib_beta_{bv}" / "b_nat" / "summary.json"
        if sf.exists():
            s = _read_json(sf)["summary"]
            v_labels.append(str(bv))
            v_auroc.append(s["auroc"]["mean"] * 100.0)
            v_f1.append(s["open_set_macro_f1"]["mean"] * 100.0)
            v_sources.append(str(sf))
    if v_labels:
        x = np.arange(len(v_labels))
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(x, v_auroc, marker="o", color=COLORS["ours"], label="AUROC (%)")
        ax.plot(x, v_f1, marker="s", color=COLORS["fedavg"], label="Open-Set Macro-F1 (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(v_labels)
        ax.set_xlabel(r"VIB Regularization Weight $\lambda_{\mathrm{vib}}$")
        ax.set_ylabel("Score (%)")
        ax.set_title(r"11: Sensitivity to VIB Regularization Weight $\lambda_{\mathrm{vib}}$")
        ax.legend(frameon=True)
        apply_academic_spines(ax)
        fig.tight_layout()
        out11 = save_dual_figure(fig, figures_dir / "11_hyperparameter_sensitivity")
        manifest["11_hyperparameter_sensitivity"] = {"sources": v_sources, "outputs": out11}

    # 27: A5 Sub-block Pareto Frontier
    blocks = [(64, 1), (8, 8), (4, 16), (2, 32), (1, 64)]
    b_labels, b_auroc, b_bytes, b_sources = [], [], [], []
    for nb, bd in blocks:
        sf = results_root / f"A5_blocks_{nb}x{bd}" / "b_nat" / "summary.json"
        if sf.exists():
            s = _read_json(sf)["summary"]
            b_labels.append(f"B={nb}, d_b={bd}")
            b_auroc.append(s["auroc"]["mean"] * 100.0)
            b_bytes.append(s["phase2_comm_bytes_per_client"]["mean"] / 1024.0)
            b_sources.append(str(sf))
    if b_labels:
        fig, ax1 = plt.subplots(figsize=(7.8, 4.4))
        x = np.arange(len(b_labels))
        ax1.bar(x - 0.18, b_auroc, 0.35, color=COLORS["ours"], label="OSR AUROC (%)")
        ax1.set_ylabel("AUROC (%)")
        ax2 = ax1.twinx()
        ax2.plot(x, b_bytes, marker="D", color=COLORS["unknown"], linewidth=2.0, label="Phase II Payload (KB)")
        ax2.set_ylabel("Phase II Payload per Client (KB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(b_labels)
        ax1.set_title("27: Block-Diagonal Covariance Granularity Trade-Off (A5)")
        apply_academic_spines(ax1)
        fig.tight_layout()
        out27 = save_dual_figure(fig, figures_dir / "27_block_size_pareto_frontier")
        manifest["27_block_size_pareto_frontier"] = {"sources": b_sources, "outputs": out27}

    # 18 & 21: Covariance & Distance Ablations (A3/A4)
    a3_files = sorted((results_root / "A3_A4_cov_dist").glob("covariance_distance_ablations_*.json"))
    if a3_files:
        a3_data = _read_json(a3_files[0])
        cov_d = a3_data.get("A3_covariance_aggregation", {})
        dist_d = a3_data.get("A4_distance_scoring", {})
        if cov_d:
            names = list(cov_d.keys())
            vals = [cov_d[k]["auroc"] * 100.0 for k in names]
            fig, ax = plt.subplots(figsize=(7.4, 4.0))
            ax.barh(names, vals, color=COLORS["ours"])
            ax.set_xlabel("AUROC (%)")
            ax.set_title("21: Covariance Aggregation Strategy Ablation (A3)")
            apply_academic_spines(ax)
            fig.tight_layout()
            out21 = save_dual_figure(fig, figures_dir / "21_covariance_aggregation_ablation")
            manifest["21_covariance_aggregation_ablation"] = {"sources": [str(a3_files[0])], "outputs": out21}
        if dist_d:
            names = list(dist_d.keys())
            vals = [dist_d[k]["auroc"] * 100.0 for k in names]
            fig, ax = plt.subplots(figsize=(7.4, 4.0))
            ax.barh(names, vals, color=COLORS["fedavg"])
            ax.set_xlabel("AUROC (%)")
            ax.set_title("18: Nonconformity Scoring Metric Ablation (A4)")
            apply_academic_spines(ax)
            fig.tight_layout()
            out18 = save_dual_figure(fig, figures_dir / "18_distance_metric_ablation")
            manifest["18_distance_metric_ablation"] = {"sources": [str(a3_files[0])], "outputs": out18}

    # 31: Conformal Alpha Sweep (A6)
    a6_files = sorted((results_root / "A6_conformal").glob("conformal_alpha_sweep_*.json"))
    if a6_files:
        a6_data = _read_json(a6_files[0])
        m_list = a6_data.get("mondrian_class_conditional", [])
        g_list = a6_data.get("global_single_threshold", [])
        if m_list and g_list:
            alphas = [r["target_alpha"] * 100.0 for r in m_list]
            m_kfr = [r["empirical_kfr"] * 100.0 for r in m_list]
            g_kfr = [r["empirical_kfr"] * 100.0 for r in g_list]
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            ax.plot(alphas, alphas, linestyle="--", color=COLORS["threshold"], label=r"Ideal Diagonal ($\mathrm{KFR}=\alpha$)")
            ax.plot(alphas, m_kfr, marker="o", color=COLORS["ours"], label="Mondrian Class-Conditional (Ours)")
            ax.plot(alphas, g_kfr, marker="s", color=COLORS["unknown"], label="Single Global Threshold")
            ax.set_xlabel(r"Nominal Significance Level $\alpha$ (%)")
            ax.set_ylabel("Empirical Known False-Rejection Rate (%)")
            ax.set_title("31: Finite-Sample Conformal Calibration Guarantee (A6)")
            ax.legend(frameon=True)
            apply_academic_spines(ax)
            fig.tight_layout()
            out31 = save_dual_figure(fig, figures_dir / "31_alpha_sweep_conformal_guarantee")
            manifest["31_alpha_sweep_conformal_guarantee"] = {"sources": [str(a6_files[0])], "outputs": out31}

    # 17: LOAO (E6) & 14: Scalability (E7) & 08: Wire-speed latency (E8)
    loao_files = sorted((results_root / "E6_loao").glob("loao_results_*.json"))
    if loao_files:
        l_data = _read_json(loao_files[0]).get("per_holdout_results", {})
        if l_data:
            names = [v["held_out_attack_name"] for v in l_data.values()]
            aurocs = [v["auroc"] * 100.0 for v in l_data.values()]
            f1s = [v["unknown_f1"] * 100.0 for v in l_data.values()]
            x = np.arange(len(names))
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            ax.bar(x - 0.18, aurocs, 0.35, color=COLORS["ours"], label="AUROC (%)")
            ax.bar(x + 0.18, f1s, 0.35, color=COLORS["fedproto"], label="Unknown F1 (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_ylabel("Score (%)")
            ax.set_title("17: Leave-One-Attack-Out Zero-Day Generalization (E6)")
            ax.legend(frameon=True)
            apply_academic_spines(ax)
            fig.tight_layout()
            out17 = save_dual_figure(fig, figures_dir / "17_leave_one_attack_out_matrix")
            manifest["17_leave_one_attack_out_matrix"] = {"sources": [str(loao_files[0])], "outputs": out17}

    scal_file = results_root / "E7_scalability" / "client_scalability_summary.json"
    if scal_file.exists():
        s_data = _read_json(scal_file)
        ks = [v["client_count"] for v in s_data.values()]
        aurocs = [v["auroc"] * 100.0 for v in s_data.values()]
        times = [v["total_wall_time_sec"] for v in s_data.values()]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.0))
        ax1.plot(ks, aurocs, marker="o", color=COLORS["ours"])
        ax1.set_xlabel("Number of Clients $K$")
        ax1.set_ylabel("OSR AUROC (%)")
        ax1.set_title("AUROC vs. Client Population $K$")
        apply_academic_spines(ax1)
        ax2.plot(ks, times, marker="s", color=COLORS["fedavg"])
        ax2.set_xlabel("Number of Clients $K$")
        ax2.set_ylabel("Total Wall-Clock Time (s)")
        ax2.set_title("Execution Time vs. $K$")
        apply_academic_spines(ax2)
        fig.tight_layout()
        out14 = save_dual_figure(fig, figures_dir / "14_client_scalability")
        manifest["14_client_scalability"] = {"sources": [str(scal_file)], "outputs": out14}

    ws_file = results_root / "E8_wire_speed" / "q1_wire_speed_log.json"
    if ws_file.exists():
        w_data = _read_json(ws_file).get("E8_wire_speed_profiling", {})
        if w_data:
            names = list(w_data.keys())
            lat = [w_data[k]["latency_us_per_flow"] for k in names]
            fig, ax = plt.subplots(figsize=(7.6, 4.2))
            ax.barh(names, lat, color=COLORS["ours"])
            ax.set_xlabel(r"Inference Latency ($\mu$s / flow)")
            ax.set_title("08: Edge Gateway Inference Latency Breakdown (E8)")
            apply_academic_spines(ax)
            fig.tight_layout()
            out08 = save_dual_figure(fig, figures_dir / "08_inference_latency_breakdown")
            manifest["08_inference_latency_breakdown"] = {"sources": [str(ws_file)], "outputs": out08}
