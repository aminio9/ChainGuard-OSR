"""Versioned loader for the FedTROS-PR publication bundle contract."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

SCHEMA_NAME="fedtros_pr_publication_bundle"
SCHEMA_NAMES={"fedtros_pr_publication_bundle","fedtros_mc_publication_bundle"}
# Canonical publication rendering is deliberately strict: schema v1 contains
# Prototype-Rank science and must never be interpreted as v2.  Legacy callers
# can opt in explicitly via ``allow_legacy=True``.
SUPPORTED_SCHEMA_VERSIONS={2}


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

@dataclass(frozen=True)
class StudyData:
    study_id:str
    root:Path
    def csv(self,name:str,required:bool=False)->pd.DataFrame:
        p=self.root/name
        if not p.exists():
            if required: raise FileNotFoundError(f"Required {self.study_id} bundle file missing: {p}")
            return pd.DataFrame()
        return pd.read_csv(p)
    @property
    def summary(self)->pd.DataFrame: return self.csv("summary.csv")
    @property
    def summary_runs(self)->pd.DataFrame: return self.csv("summary_runs.csv")
    @property
    def round_curves(self)->pd.DataFrame: return self.csv("round_curves.csv")
    @property
    def scores(self)->pd.DataFrame: return self.csv("scores.csv")
    @property
    def test_scores(self)->pd.DataFrame: return self.csv("test_scores.csv")
    @property
    def calibration_scores(self)->pd.DataFrame: return self.csv("calibration_scores.csv")
    @property
    def prototype_selection(self)->pd.DataFrame: return self.csv("prototype_selection.csv")
    @property
    def prototype_summary(self)->pd.DataFrame: return self.csv("prototype_summary.csv")
    @property
    def prototype_assignments(self)->pd.DataFrame: return self.csv("prototype_assignments.csv")
    @property
    def latent_projection(self)->pd.DataFrame: return self.csv("latent_projection.csv")
    @property
    def conformal_metadata(self)->dict:
        p=self.root/"conformal_metadata.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    @property
    def roc(self)->pd.DataFrame: return self.csv("roc.csv")
    @property
    def pr(self)->pd.DataFrame: return self.csv("pr.csv")
    @property
    def client_metrics(self)->pd.DataFrame: return self.csv("client_metrics.csv")
    @property
    def client_distribution(self)->pd.DataFrame: return self.csv("client_distribution.csv")
    @property
    def communication(self)->pd.DataFrame: return self.csv("communication.csv")
    @property
    def efficiency_curve(self)->pd.DataFrame: return self.csv("efficiency_curve.csv")
    @property
    def runtime(self)->pd.DataFrame: return self.csv("runtime.csv")
    @property
    def paired_deltas(self)->pd.DataFrame: return self.csv("paired_deltas.csv")

class PublicationBundle:
    def __init__(self,path:str|Path,verify_checksums:bool=True,allow_legacy:bool=False):
        self.root=Path(path).expanduser().resolve()
        mp=self.root/"manifest.json"
        if not mp.exists(): raise FileNotFoundError(f"Publication bundle manifest not found: {mp}")
        self.manifest=json.loads(mp.read_text(encoding="utf-8"))
        if self.manifest.get("schema_name") not in SCHEMA_NAMES: raise ValueError(f"Unsupported bundle schema: {self.manifest.get('schema_name')}")
        version=int(self.manifest.get("schema_version",-1))
        if version==1 and allow_legacy:
            self.legacy=True
        elif version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported bundle version: {self.manifest.get('schema_version')}; canonical publication mode requires schema v2")
        else:
            self.legacy=False
        if self.manifest.get("method") not in {"FedTROS-PR","FedTROS-MC","FedTROS"}: raise ValueError(f"Bundle method {self.manifest.get('method')!r} is not a recognized canonical FedTROS export")
        if not self.legacy:
            required={"detector":"adaptive_multicenter_lw_split_conformal",
                      "feature_source":"student_hidden_l2","alpha":0.05,
                      "aggregation_gamma":0.5,"legacy_osr_branch":False,
                      "synthetic_boundary":False}
            for key,val in required.items():
                got=self.manifest.get(key)
                if got != val:
                    raise ValueError(f"Canonical v2 manifest field {key}={got!r} expected {val!r}")
        if verify_checksums: self.verify_checksums()
    @property
    def studies(self)->list[str]: return list(self.manifest.get("studies_present",[]))
    def study(self,study_id:str,required:bool=True)->StudyData:
        p=self.root/study_id
        if required and not p.is_dir(): raise FileNotFoundError(f"Study {study_id} absent from bundle {self.root}")
        return StudyData(study_id,p)
    def verify_checksums(self)->None:
        failures=[]
        for rel,info in self.manifest.get("files",{}).items():
            p=self.root/rel
            if not p.exists(): failures.append(f"missing:{rel}"); continue
            expected=str(info.get("sha256",""))
            if expected and _sha256(p)!=expected: failures.append(f"checksum:{rel}")
        if failures: raise ValueError("Publication bundle integrity failure: "+", ".join(failures[:10]))

def load_publication_bundle(path:str|Path,verify_checksums:bool=True,allow_legacy:bool=False)->PublicationBundle:
    return PublicationBundle(path,verify_checksums=verify_checksums,allow_legacy=allow_legacy)
