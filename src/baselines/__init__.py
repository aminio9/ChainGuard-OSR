"""Modular baseline implementations for Federated Learning and Open-Set Recognition.

Organized into canonical Q1 IEEE TIFS/TDSC categories:
- Category 1 (Canonical FL): FedAvg (2017), FedProx (2020)
- Category 2 (Non-IID Drift-Corrected FL): SCAFFOLD (ICML 2020), FedDyn (ICLR 2021)
- Category 3 (Balanced Representation FL): FedRoD (ICLR 2022), FedDecorr (ICLR 2023), FedProto (AAAI 2022)
- Category 4 (Specialized Fed-OSR Competitors): FedPD (ICCV 2023), FedPD++ (IJCV 2026), FedOSS (IEEE TMI 2024)
- Category 5 (Centralized OSR Adapted to FL): OpenMax (CVPR 2016), Energy-OOD (NeurIPS 2020), Vanilla-Mahalanobis (NeurIPS 2018)
"""

from __future__ import annotations

from src.baselines.base import BaseBaseline, compute_feature_svd_spectrum
from src.baselines.fedavg.fedavg import FedAvgBaseline
from src.baselines.fedprox.fedprox import FedProxBaseline
from src.baselines.scaffold.scaffold import SCAFFOLDBaseline
from src.baselines.feddyn.feddyn import FedDynBaseline
from src.baselines.fedrod.fedrod import FedRoDBaseline
from src.baselines.feddecorr.feddecorr import FedDecorrBaseline
from src.baselines.fedproto.fedproto import FedProtoBaseline
from src.baselines.fedpd.fedpd import FedPDBaseline
from src.baselines.fedpd_plus_plus.fedpd_plus_plus import FedPDPlusPlusBaseline
from src.baselines.fedoss.fedoss import FedOSSBaseline
from src.baselines.openmax.openmax import OpenMaxBaseline
from src.baselines.energy.energy import EnergyBaseline
from src.baselines.mahalanobis.mahalanobis import VanillaMahalanobisBaseline

__all__ = [
    "BaseBaseline",
    "compute_feature_svd_spectrum",
    "FedAvgBaseline",
    "FedProxBaseline",
    "SCAFFOLDBaseline",
    "FedDynBaseline",
    "FedRoDBaseline",
    "FedDecorrBaseline",
    "FedProtoBaseline",
    "FedPDBaseline",
    "FedPDPlusPlusBaseline",
    "FedOSSBaseline",
    "OpenMaxBaseline",
    "EnergyBaseline",
    "VanillaMahalanobisBaseline",
]
