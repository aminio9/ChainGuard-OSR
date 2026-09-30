"""Eight Q1 main-paper figures rendered only from a validated publication bundle."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ref_plots.data.fedtros_bundle import PublicationBundle
from src.ref_plots.style import METHOD_STYLES, CLIENT_STYLES, HEATMAP_CMAP, apply_academic_spines, configure_academic_style, save_figure

FIGURE_STEMS={
 "fig01":"01_client_class_support_heatmap",
 "fig02":"02_unknown_score_operating_diagnostic",
 "fig03":"03_noniid_closed_macro_f1",
 "fig04":"04_noniid_osr_operating_tradeoff",
 "fig05":"05_leave_one_attack_out_heatmap",
 "fig06":"06_component_ablation_deltas",
 "fig07":"07_performance_vs_communication",
 "fig08":"08_fixed_data_scalability",
}
ALIASES={
 "macro_f1":["closed_set/macro_f1","federated/global_macro_f1","local_student_f1_macro","macro_f1","openset_f1_macro","open_set/macro_f1"],
 "auroc":["open_set/auroc","openset_auroc","round_openset_auroc","auroc"],
 "auprc":["open_set/auprc","openset_auprc","auprc"],
 "unknown_f1":["open_set/unknown_f1","openset_unknown_f1","unknown_f1"],
 "unknown_recall":["open_set/unknown_recall","openset_unknown_recall","round_openset_unknown_recall"],
 "kfur":["open_set/known_false_unknown_rate","openset_known_false_unknown_rate","known_false_unknown_rate"],
 "fpr95":["open_set/fpr_at_95_tpr","openset_fpr95","round_openset_fpr95"],
}

def _metric(summary:pd.DataFrame,key:str)->pd.DataFrame:
    if summary.empty:return summary
    for candidate in ALIASES[key]:
        part=summary[summary["metric"]==candidate]
        if not part.empty:return part.copy()
    return pd.DataFrame()

def _style(method:str)->dict:
    aliases={"FedTROS-MC":"Proposed","FedTROS":"Proposed","FedTROS-PR":"Proposed","FedPROTEUS":"Proposed","DKD-FedOS":"Proposed","FedAvg-Student":"FedAvg","FedProx-Student":"FedProx"}
    return METHOD_STYLES.get(aliases.get(method,method),{"marker":"D","linestyle":":","linewidth":2.2,"markersize":7,"label":method})

def _col(df:pd.DataFrame,*names:str):
    for n in names:
        if n in df.columns:return n
    return None

def fig01(bundle:PublicationBundle,out:Path):
    """Client class-support heatmaps for IID and three Dirichlet conditions."""
    frames=[]
    if "E1-IID-CS" in bundle.studies:
        iid=bundle.study("E1-IID-CS").client_distribution
        if not iid.empty:
            iid=iid.copy(); iid["partition_label"]="IID"; frames.append(iid)
    if "E3-NIID-CS" in bundle.studies:
        noniid=bundle.study("E3-NIID-CS").client_distribution
        if not noniid.empty:
            noniid=noniid.copy()
            noniid["partition_label"]=noniid["alpha"].map(lambda a:f"Dirichlet α={float(a):g}") if "alpha" in noniid else "Non-IID"
            frames.append(noniid)
    if not frames:
        raise ValueError("E1/E3 client_distribution.csv required for Figure 1")
    df=pd.concat(frames,ignore_index=True,sort=False)
    if "method" in df.columns:
        candidate=df[df["method"].isin(["Proposed", "FedTROS-PR", "FedTROS-MC", "FedTROS"])]
        if not candidate.empty: df=candidate
    if "seed" in df.columns:
        candidate=df[df["seed"]==42]
        if not candidate.empty: df=candidate
    desired=["IID","Dirichlet α=1","Dirichlet α=0.5","Dirichlet α=0.1"]
    labels=[x for x in desired if x in set(df["partition_label"].astype(str))]
    if not labels: labels=list(dict.fromkeys(df["partition_label"].astype(str).tolist()))[:4]
    panels=len(labels); fig,axes=plt.subplots(1,panels,figsize=(4.15*panels,5.0),squeeze=False,layout="constrained")
    meta={"run_id","study","method","dataset","alpha","iid","seed","num_clients","variant","unknown_labels","client_id","partition_label"}
    cls=[c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if not cls: raise ValueError("No numeric class-count columns found in client distribution bundle")
    im=None
    for ax,label in zip(axes.flat,labels):
        sub=df[df["partition_label"].astype(str)==label].copy()
        vals=sub[cls].fillna(0).to_numpy(float); sums=np.maximum(vals.sum(axis=1,keepdims=True),1); vals=vals/sums
        im=ax.imshow(vals,aspect="auto",cmap=HEATMAP_CMAP,vmin=0,vmax=max(0.01,float(vals.max())))
        ax.set_title(label, fontsize=12.0, weight="bold", color="#000000")
        ax.set_xticks(range(len(cls)), cls, rotation=45, ha="right", fontsize=9.5)
        ax.set_ylabel("Client", fontsize=11.0, weight="bold", color="#000000")
        ax.set_yticks(range(len(sub)), [str(x) for x in sub.get("client_id", pd.Series(range(1, len(sub) + 1)))], fontsize=9.5)
        apply_academic_spines(ax)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8)
        cbar.set_label("Client class proportion", fontsize=10.5, color="#000000", weight="bold")
        cbar.outline.set_edgecolor("#000000")
        cbar.outline.set_linewidth(1.0)
        cbar.ax.tick_params(colors="#000000")
    return save_figure(fig, out, FIGURE_STEMS["fig01"])

def _score_arrays(df:pd.DataFrame):
    score=_col(df,"nonconformity_score")
    flag=_col(df,"unknown_flag","is_unknown")
    if score is None: raise ValueError("No open-set score column in E2 scores bundle")
    if flag is not None: y=pd.to_numeric(df[flag],errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        role=_col(df,"known_or_unknown")
        if role is None: raise ValueError("No unknown indicator in E2 scores bundle")
        y=(df[role].astype(str).str.lower()=="unknown").astype(int).to_numpy()
    s=pd.to_numeric(df[score],errors="coerce").fillna(0).to_numpy(float); return y,s

def fig02(bundle:PublicationBundle,out:Path):
    study=bundle.study("E2-IID-OSR")
    df=study.scores
    pr=study.pr
    if df.empty: raise ValueError("E2 scores.csv required for Figure 2")
    if pr.empty: raise ValueError("E2 pr.csv exported by FedTROS is required for Figure 2")
    if "seed" in df:
        candidate=df[df["seed"]==42]
        if not candidate.empty: df=candidate
    if "seed" in pr:
        candidate=pr[pr["seed"]==42]
        if not candidate.empty: pr=candidate
    y,scores=_score_arrays(df); known=scores[y==0]; unknown=scores[y==1]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout="constrained")
    axes[0].hist(known,bins=40,density=True,alpha=.55,label="Known",histtype="stepfilled")
    axes[0].hist(unknown,bins=40,density=True,alpha=.55,label="Unknown",histtype="stepfilled")
    # The operating threshold is a FedTROS-computed scientific result; the plotting
    # repository only renders it.  Prefer the aggregate summary and fall back to a
    # per-sample exported threshold column if present.
    threshold=None
    summary=study.summary
    if not summary.empty and "metric" in summary.columns:
        part=summary[summary["metric"].isin(["prototype_rank/threshold","open_set/rejection_threshold"])]
        if not part.empty and "mean" in part.columns:
            threshold=float(pd.to_numeric(part["mean"],errors="coerce").dropna().iloc[0])
    if threshold is None:
        tcol=_col(df,"tau_alpha")
        if tcol:
            values=pd.to_numeric(df[tcol],errors="coerce").dropna()
            if not values.empty: threshold=float(values.iloc[0])
    if threshold is not None and np.isfinite(threshold):
        axes[0].axvline(threshold,linestyle="--",linewidth=1.6,label=f"Threshold={threshold:.3f}")
    axes[0].set_xlabel("Candidate-class Mahalanobis nonconformity A(x)"); axes[0].set_ylabel("Density"); axes[0].legend(); apply_academic_spines(axes[0])
    precision_col=_col(pr,"precision"); recall_col=_col(pr,"recall")
    if not precision_col or not recall_col: raise ValueError("E2 pr.csv lacks precision/recall columns")
    axes[1].plot(pd.to_numeric(pr[recall_col],errors="coerce"),pd.to_numeric(pr[precision_col],errors="coerce"),linewidth=2.4)
    axes[1].set_xlabel("Unknown recall"); axes[1].set_ylabel("Unknown precision"); axes[1].set_xlim(0,1); axes[1].set_ylim(0,1.02); apply_academic_spines(axes[1])
    return save_figure(fig,out,FIGURE_STEMS["fig02"])

def fig03(bundle:PublicationBundle,out:Path):
    df=_metric(bundle.study("E3-NIID-CS").summary,"macro_f1");
    if df.empty: raise ValueError("E3 Macro-F1 aggregate missing")
    fig,ax=plt.subplots(figsize=(7.2,4.8),layout="constrained")
    for method,sub in df.groupby("method"):
        sub=sub.sort_values("alpha"); st=_style(method); lbl=st.get("label",method); ax.errorbar(sub["alpha"],sub["mean"],yerr=[sub["mean"]-sub["ci95_low"],sub["ci95_high"]-sub["mean"]],marker=st.get("marker","o"),linestyle=st.get("linestyle","-"),linewidth=st.get("linewidth",2.3),label=lbl,capsize=3)
    ax.set_xlabel("Dirichlet α"); ax.set_ylabel("Macro-F1"); ax.legend(); apply_academic_spines(ax); return save_figure(fig,out,FIGURE_STEMS["fig03"])

def fig04(bundle:PublicationBundle,out:Path):
    s=bundle.study("E4-NIID-FOSR").summary; k=_metric(s,"kfur"); r=_metric(s,"unknown_recall")
    if k.empty or r.empty: raise ValueError("E4 K-FUR and unknown recall aggregates required")
    keys=["method","dataset","alpha","num_clients","variant","unknown_labels"]; d=k[keys+["mean"]].rename(columns={"mean":"kfur"}).merge(r[keys+["mean"]].rename(columns={"mean":"recall"}),on=keys)
    fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
    for method,sub in d.groupby("method"):
        st=_style(method); lbl=st.get("label",method); ax.plot(sub["kfur"],sub["recall"],marker=st.get("marker","o"),linestyle=st.get("linestyle","-"),label=lbl)
        for _,row in sub.iterrows(): ax.annotate(f"α={row.alpha:g}",(row.kfur,row.recall),xytext=(4,4),textcoords="offset points",fontsize=8.5,weight="bold",color="#000000")
    ax.set_xlabel("Known false-unknown rate (lower is better)"); ax.set_ylabel("Unknown recall (higher is better)"); ax.legend(); apply_academic_spines(ax); return save_figure(fig,out,FIGURE_STEMS["fig04"])

def fig05(bundle:PublicationBundle,out:Path):
    s=bundle.study("E8-LOAO").summary; keys=["auroc","auprc","fpr95","unknown_recall","unknown_f1","kfur"]; parts=[]
    for key in keys:
        m=_metric(s,key)
        if not m.empty: parts.append(m[["unknown_labels","mean"]].rename(columns={"mean":key}))
    if not parts: raise ValueError("E8 open-set aggregate metrics required")
    d=parts[0]
    for p in parts[1:]: d=d.merge(p,on="unknown_labels",how="outer")
    d=d.drop_duplicates("unknown_labels").set_index("unknown_labels"); vals=d.to_numpy(float)
    # direction-normalized visual intensity only; annotations remain exact values
    disp=vals.copy();
    for j,k in enumerate(d.columns):
        if k in {"fpr95","kfur"}: disp[:,j]=1-disp[:,j]
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained"); im=ax.imshow(disp,aspect="auto",cmap=HEATMAP_CMAP)
    ax.set_yticks(range(len(d)),d.index); ax.set_xticks(range(len(d.columns)),[c.upper() for c in d.columns],rotation=35,ha="right")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            txt_color = "#FFFFFF" if disp[i, j] > 0.55 else "#000000"
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=8.5, weight="bold", color=txt_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Direction-normalized performance", fontsize=10.5, color="#000000", weight="bold")
    cbar.outline.set_edgecolor("#000000")
    cbar.outline.set_linewidth(1.0)
    cbar.ax.tick_params(colors="#000000")
    apply_academic_spines(ax)
    return save_figure(fig, out, FIGURE_STEMS["fig05"])

def fig06(bundle:PublicationBundle,out:Path):
    frames=[]
    metric_priority=("open_set/auroc","closed_set/macro_f1","open_set/unknown_f1")
    for st in ("A1-TEACHER","A2-ANCHOR","A3-TRANSFER","A4-PR"):
        if st not in bundle.studies:
            continue
        d=bundle.study(st).paired_deltas
        if d.empty:
            continue
        chosen=None
        for metric in metric_priority:
            part=d[d["metric"]==metric]
            if not part.empty:
                chosen=part.copy(); break
        if chosen is None:
            continue
        st_disp = "A4-Rejection" if st == "A4-PR" else st
        chosen["label"] = st_disp + ":" + chosen["variant"].astype(str)
        frames.append(chosen)
    if not frames:
        raise ValueError("No FedTROS-computed paired ablation deltas found for A1-A4")
    d=pd.concat(frames,ignore_index=True).sort_values("delta_mean")
    fig,ax=plt.subplots(figsize=(8,max(4,0.38*len(d))),layout="constrained")
    y=np.arange(len(d))
    xerr=[d["delta_mean"]-d["delta_ci95_low"], d["delta_ci95_high"]-d["delta_mean"]]
    ax.errorbar(d["delta_mean"],y,xerr=xerr,fmt="o",capsize=3)
    ax.axvline(0,linewidth=1.0,linestyle="--")
    ax.set_yticks(y,d["label"])
    ax.set_xlabel("Ablation minus full/reference (paired mean ± 95% CI)")
    apply_academic_spines(ax)
    return save_figure(fig,out,FIGURE_STEMS["fig06"])

def fig07(bundle:PublicationBundle,out:Path):
    # FedTROS owns the scientific join between measured communication and the
    # recorded validation/performance series.  The plots repository only renders
    # the versioned efficiency_curve contract.
    df=bundle.study("E7-EFFICIENCY").efficiency_curve
    if df.empty:
        raise ValueError("E7 efficiency_curve.csv required; regenerate the publication bundle with FedTROS")
    required={"method","communication/cumulative_bytes","performance_value"}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"E7 efficiency curve missing columns: {sorted(missing)}")
    fig,ax=plt.subplots(figsize=(7.3,4.8),layout="constrained")
    # Multiple paired seeds are rendered as a mean communication-performance
    # trajectory by round.  This is a visual aggregation of already-computed
    # FedTROS values, not a second scientific metric computation.
    group_keys=["method","round"] if "round" in df.columns else ["method"]
    view=(df.assign(
            _mb=pd.to_numeric(df["communication/cumulative_bytes"],errors="coerce")/(1024**2),
            _perf=pd.to_numeric(df["performance_value"],errors="coerce"),
          )
          .dropna(subset=["_mb","_perf"])
          .groupby(group_keys,as_index=False)
          .agg(_mb=("_mb","mean"),_perf=("_perf","mean")))
    for method,sub in view.groupby("method"):
        sub=sub.sort_values("_mb"); st=_style(method); lbl=st.get("label",method)
        ax.plot(sub["_mb"],sub["_perf"],marker=st.get("marker","o"),
                markevery=max(1,len(sub)//8),linestyle=st.get("linestyle","-"),
                linewidth=st.get("linewidth",2.3),label=lbl)
    metric_labels=sorted({str(x) for x in df.get("performance_metric",pd.Series(dtype=str)).dropna().unique()})
    ylabel="Validation performance" if not metric_labels else f"Performance ({metric_labels[0]})"
    ax.set_xlabel("Cumulative communicated model payload (MiB)")
    ax.set_ylabel(ylabel); ax.legend(); apply_academic_spines(ax)
    return save_figure(fig,out,FIGURE_STEMS["fig07"])

def fig08(bundle:PublicationBundle,out:Path):
    s=bundle.study("E6-SCALE").summary; perf=_metric(s,"auroc");
    if perf.empty: perf=_metric(s,"macro_f1")
    rt=bundle.study("E6-SCALE").runtime
    if perf.empty: raise ValueError("E6 performance aggregate required")
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout="constrained"); p=perf.groupby("num_clients",as_index=False).agg(mean=("mean","mean"),low=("ci95_low","mean"),high=("ci95_high","mean")).sort_values("num_clients"); axes[0].errorbar(p.num_clients,p["mean"],yerr=[p["mean"]-p.low,p.high-p["mean"]],marker="o",capsize=3); axes[0].set_xlabel("Clients"); axes[0].set_ylabel("Open-set performance"); apply_academic_spines(axes[0])
    if not rt.empty:
        tcol=_col(rt,"runtime/round_seconds","round_seconds","total_round_seconds","round_wall_time_sec","wall_time_sec")
        if tcol:
            q=rt.groupby("num_clients")[tcol].median().sort_index(); axes[1].plot(q.index,q.values,marker="s"); axes[1].set_xlabel("Clients"); axes[1].set_ylabel("Median round time (s)")
    apply_academic_spines(axes[1]); return save_figure(fig,out,FIGURE_STEMS["fig08"])

GENERATORS={"fig01":fig01,"fig02":fig02,"fig03":fig03,"fig04":fig04,"fig05":fig05,"fig06":fig06,"fig07":fig07,"fig08":fig08}

def generate_main_figures(bundle:PublicationBundle,output_dir:Path,strict:bool=False)->dict[str,str]:
    configure_academic_style(); output_dir.mkdir(parents=True,exist_ok=True); generated={}; errors={}
    for key,fn in GENERATORS.items():
        try: generated[key]=str(fn(bundle,output_dir))
        except Exception as exc:
            errors[key]=str(exc)
            if strict: raise
    (output_dir/"generation_report.json").write_text(pd.Series({"generated":generated,"errors":errors}).to_json(indent=2),encoding="utf-8")
    return generated
