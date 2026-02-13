from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BusinessMetrics:
    # Métriques orientées décision
    auto_coverage: float
    acc_high_conf: float
    cost_total: float
    fn: int
    fp: int
    revisions: int


def slice_by_confidence(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, thr: float) -> float:
    # Accuracy sur la tranche conf>=thr
    mask = conf >= thr
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def compute_auto_coverage(conf: np.ndarray, thr_high: float) -> float:
    # % cas gérés automatiquement (conf>=0.85 par ex)
    return float((conf >= thr_high).mean())


def compute_cost(fn: int, fp: int, revisions: int) -> float:
    # Coût du sujet: (FN*1000) + (FP*100) + (Revision*50)
    return float(fn * 1000 + fp * 100 + revisions * 50)


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    thr_high: float = 0.85,
    revision_mask: np.ndarray | None = None,
    positive_class: int | None = None,
) -> BusinessMetrics:
    # Calcule couverture, acc, coût (FP/FN nécessitent une définition binaire)
    auto_cov = compute_auto_coverage(conf, thr_high)
    acc_high = slice_by_confidence(y_true, y_pred, conf, thr_high)

    # Par défaut: revisions = cas non auto
    if revision_mask is None:
        revisions = int((conf < thr_high).sum())
    else:
        revisions = int(revision_mask.sum())

    # FP/FN: à définir selon ton mapping (ex: "tumeur" vs "notumor")
    fn = 0
    fp = 0
    if positive_class is not None:
        y_true_bin = (y_true == positive_class).astype(int)
        y_pred_bin = (y_pred == positive_class).astype(int)
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())

    cost = compute_cost(fn, fp, revisions)
    return BusinessMetrics(
        auto_coverage=auto_cov,
        acc_high_conf=acc_high,
        cost_total=cost,
        fn=fn,
        fp=fp,
        revisions=revisions,
    )
