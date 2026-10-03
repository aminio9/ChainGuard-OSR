"""Auditable, file-only ingestion of experiment outputs; no training imports or imputation."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve

METRICS = {
    'known_accuracy': 'Known accuracy (%)', 'known_macro_f1': 'Known macro-F1 (%)',
    'os_macro_f1': 'Open-set macro-F1 (%)', 'auroc': 'Unknown AUROC (%)',
    'auprc': 'Unknown average precision (%)', 'unknown_f1': 'Unknown F1 (%)',
    'unknown_recall': 'Unknown recall (%)', 'kfr': 'Known false rejection (%)',
    'known_accuracy_after': 'Known accuracy after rejection (%)', 'fpr95': 'FPR at 95% TPR (%)',
    'auroc_mah': 'Mahalanobis AUROC (%)', 'auroc_rec': 'Reconstruction AUROC (%)',
}

def read_json(p):
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def numeric_items(d, prefix=''):
    for k,v in d.items():
        key=f'{prefix}/{k}' if prefix else k
        if isinstance(v,dict): yield from numeric_items(v,key)
        elif isinstance(v,(int,float)) and not isinstance(v,bool): yield key,v

def recompute(s):
    unknown=s.known_or_unknown.eq('unknown').to_numpy()
    known=~unknown
    y=s.true_label.to_numpy(); candidate=s.candidate_pred.to_numpy()
    rejected=s.final_reject.astype(str).str.lower().isin(['true','1']).to_numpy()
    final=s.final_pred.to_numpy()
    # Collapse the declared unknown set; never infer unknown status from class order.
    y_os=np.where(unknown,-1,y); p_os=np.where(rejected,-1,final)
    out={'n_test':len(s),'n_known':int(known.sum()),'n_unknown':int(unknown.sum()),
         'known_accuracy':accuracy_score(y[known],candidate[known]),
         'known_macro_f1':f1_score(y[known],candidate[known],average='macro',zero_division=0),
         'known_accuracy_after':float((p_os[known]==y[known]).mean()),
         'os_macro_f1':f1_score(y_os,p_os,average='macro',zero_division=0),
         'unknown_f1':f1_score(unknown,rejected,zero_division=0),
         'unknown_recall':float(rejected[unknown].mean()) if unknown.any() else np.nan,
         'kfr':float(rejected[known].mean())}
    score=s.nonconformity_score.to_numpy(float)
    if np.unique(unknown).size==2 and np.isfinite(score).all():
        out.update(auroc=roc_auc_score(unknown,score),auprc=average_precision_score(unknown,score))
        fpr,tpr,_=roc_curve(unknown,score)
        out['fpr95']=float(fpr[np.flatnonzero(tpr>=.95)[0]])
        for col,key in [('score_mah','auroc_mah'),('score_rec','auroc_rec')]:
            if col in s and np.isfinite(s[col]).all():out[key]=roc_auc_score(unknown,s[col])
    out['reject_rule_mismatches']=int(np.sum(rejected != (score>=s.tau_alpha.to_numpy())))
    return out

class VerifiedRuns:
    def __init__(self, source, destination, snapshot=True):
        self.source=Path(source); self.root=Path(destination)
        self.root.mkdir(parents=True,exist_ok=True)
        self.paths={}; self.configs={}; self.manifests={}; self.scores={}; self.calibrations={}
        self.rows=[]; scalar=[]; checks=[]; provenance=[]; inventory=[]; datasets=[]
        for p in sorted(self.source.rglob('*')):
            if p.is_file(): inventory.append({'path':str(p.relative_to(self.source)),'bytes':p.stat().st_size,'extension':p.suffix})
        pd.DataFrame(inventory).to_csv(self.root/'source_inventory.csv',index=False)
        manifests=sorted(p for p in self.source.rglob('run_manifest.json') if p.parent.name!='metadata')
        for i,p in enumerate(manifests):
            m=read_json(p); rid=m['run_id']; d=p.parent
            c=yaml.safe_load((d/'resolved_config.yaml').read_text(encoding='utf-8'))
            self.configs[rid]=c;self.manifests[rid]=m
            selected=[]
            for f in d.rglob('*'):
                if f.is_file() and f.suffix in {'.csv','.json','.yaml'} and ('checkpoints' not in f.parts) and ('artifacts' not in f.parts):
                    # Avoid mirrored copies; retain canonical root manifests and actual metrics/OSR data.
                    rel=f.relative_to(d)
                    if rel.parts[0]=='config' or (rel.parts[0]=='metadata' and (d/rel.name).exists()):continue
                    selected.append(f)
            target=self.root/'runs'/rid
            for f in selected:
                rel=f.relative_to(d); out=target/rel
                digest=sha(f)
                if snapshot:
                    out.parent.mkdir(parents=True,exist_ok=True)
                    with f.open('rb') as src, out.open('wb') as dst:
                        shutil.copyfileobj(src,dst,1048576)
                    if sha(out)!=digest:raise RuntimeError(f'Snapshot changed while copying {f}')
                provenance.append({'run_id':rid,'source':str(f),'snapshot':str(out.relative_to(self.root)),'sha256':digest,'bytes':f.stat().st_size})
            self.paths[rid]=target if snapshot else d
            row={k:m.get(k) for k in ['run_id','status','method','dataset','seed','num_clients','num_rounds','iid','alpha','split_hash','config_hash','git_commit','git_dirty']}
            row['study']=m.get('study_id');row['variant']=c.get('experiment',{}).get('variant','canonical')
            row['unknown_labels']='|'.join(m.get('unknown_labels',[]))
            row['known_labels']='|'.join(m.get('known_labels',[]))
            row['feature_source']=c.get('open_set',{}).get('prototype',{}).get('feature_source')
            row['score_mode']=c.get('open_set',{}).get('score_mode')
            for name in ['final_metrics','evaluation_metrics','pipeline_timing']:
                metric=read_json(d/'metrics'/f'{name}.json')
                for key,val in numeric_items(metric):scalar.append({'run_id':rid,'file':name,'metric':key,'value':val})
            raw=read_json(d/'metrics/final_metrics.json')
            row['total_seconds']=raw.get('runtime/total_seconds',np.nan)
            dm=read_json(d/'data_manifest.json')
            datasets.append({'run_id':rid,'dataset':row['dataset'],'study':row['study'],
                **{k:dm.get(k) for k in ['feature_dim','num_train_samples','num_validation_samples','num_closed_test_samples','num_open_test_samples','num_known_samples','num_unknown_samples']},
                'source_class_counts':json.dumps(dm.get('source_class_counts',{})), 'unknown_labels':row['unknown_labels']})
            if m['status']=='COMPLETED' and (d/'osr/test_scores.csv').exists():
                s=pd.read_csv(d/'osr/test_scores.csv');self.scores[rid]=s
                row.update(recompute(s))
                aliases={'open_set/auroc':'auroc','open_set/auprc':'auprc','open_set/KFR':'kfr','open_set/macro_f1':'os_macro_f1','open_set/unknown_f1':'unknown_f1','open_set/unknown_recall':'unknown_recall','open_set/known_accuracy_before':'known_accuracy','open_set/known_accuracy_after':'known_accuracy_after','open_set/fpr95':'fpr95'}
                for key,outkey in aliases.items():
                    if key in raw:checks.append({'run_id':rid,'check':'reported_vs_recomputed','metric':key,'reported':raw[key],'recomputed':row.get(outkey),'abs_delta':abs(raw[key]-row.get(outkey,np.nan))})
                cp=d/'osr/calibration_scores.csv'
                if cp.exists():
                    cal=pd.read_csv(cp); self.calibrations[rid]=cal
                    a=float(cal.alpha.iloc[0]);rank=int(np.ceil((len(cal)+1)*(1-a)))
                    tau=np.sort(cal.nonconformity_score.to_numpy())[rank-1] if rank<=len(cal) else np.inf
                    row.update(calibration_size=len(cal),conformal_alpha=a,tau=float(s.tau_alpha.iloc[0]),calibration_reject_rate=float((cal.nonconformity_score>=tau).mean()),threshold_abs_delta=abs(tau-s.tau_alpha.iloc[0]))
                    # Check whether the emitted fused score can be reconstructed from calibration moments.
                    if {'score_mahalanobis','score_reconstruction'}.issubset(cal) and {'score_mah','score_rec'}.issubset(s):
                        w=float(c.get('open_set',{}).get('ensemble_recon_weight',.5))
                        cm=cal.score_mahalanobis;cr=cal.score_reconstruction
                        fused=(1-w)*(s.score_mah-cm.mean())/max(cm.std(ddof=0),1e-8)+w*(s.score_rec-cr.mean())/max(cr.std(ddof=0),1e-8)
                        row['ensemble_reconstruction_max_error']=float(np.max(np.abs(fused-s.nonconformity_score)))
            elif m['status']=='COMPLETED':
                for key,val in raw.items():
                    if key in ['closed_set/accuracy','closed_set/macro_f1']:
                        row[{'closed_set/accuracy':'known_accuracy','closed_set/macro_f1':'known_macro_f1'}[key]]=val
            self.rows.append(row)
            print(f'Ingested {i+1}/{len(manifests)}: {rid}',flush=True)
        self.frame=pd.DataFrame(self.rows);self.datasets=pd.DataFrame(datasets)
        self.frame.to_csv(self.root/'run_summary.csv',index=False)
        self.datasets.to_csv(self.root/'dataset_runs.csv',index=False)
        pd.DataFrame(scalar).to_csv(self.root/'all_scalar_metrics.csv',index=False)
        pd.DataFrame(checks).to_csv(self.root/'metric_reconciliation.csv',index=False)
        pd.DataFrame(provenance).to_csv(self.root/'source_checksums.csv',index=False)
        (self.root/'variables.json').write_text(json.dumps({'metric_units':'fractions in data; percent in figures/tables','dirichlet_symbol':'a','conformal_symbol':'epsilon','unknown_positive':True,'metrics':METRICS},indent=2))

    def csv(self,rid,rel):
        p=self.paths[rid]/rel
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def history(self,rid):
        h=self.csv(rid,'metrics/federated_history.csv')
        if {'phase','metric_name','round','metric_value'}.issubset(h):
            h['key']=h.phase+'/'+h.metric_name
            return h.pivot(index='round',columns='key',values='metric_value').reset_index()
        return h
