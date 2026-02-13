from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BusinessMetrics:
    auto_coverage: float
    acc_high_conf: float
    high_conf_count: int
    cost_total: float
    fn: int
    fp: int
    revisions: int


@dataclass(frozen=True)
class ConfidenceOperatingPoint:
    threshold: float
    coverage: float
    accuracy: float


def slice_by_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    thr: float,
) -> float:
    mask = conf >= thr
    if int(mask.sum()) == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def compute_auto_coverage(conf: np.ndarray, thr_high: float) -> float:
    return float((conf >= thr_high).mean())


def compute_cost(fn: int, fp: int, revisions: int) -> float:
    return float(fn * 1000 + fp * 100 + revisions * 50)


def compute_tumor_fn_fp(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    notumor_index: int = 2,
) -> Tuple[int, int]:
    true_tumor = y_true != notumor_index
    pred_tumor = y_pred != notumor_index

    fn = int((true_tumor & ~pred_tumor).sum())
    fp = int((~true_tumor & pred_tumor).sum())
    return fn, fp


def accuracy_by_confidence_bands(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    bands: Sequence[Tuple[float, float]] = ((0.0, 0.5), (0.5, 0.65), (0.65, 0.85), (0.85, 1.01)),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for low, high in bands:
        mask = (conf >= low) & (conf < high)
        key = f"[{low:.2f},{high:.2f})"
        out[key] = float((y_true[mask] == y_pred[mask]).mean()) if int(mask.sum()) > 0 else float("nan")
    return out


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    thr_high: float = 0.85,
    revision_mask: np.ndarray | None = None,
    notumor_index: int = 2,
) -> BusinessMetrics:
    auto_cov = compute_auto_coverage(conf, thr_high)
    high_conf_mask = conf >= thr_high
    high_conf_count = int(high_conf_mask.sum())
    acc_high = slice_by_confidence(y_true, y_pred, conf, thr_high)

    if revision_mask is None:
        revisions = int((~high_conf_mask).sum())
    else:
        revisions = int(np.asarray(revision_mask).sum())

    fn, fp = compute_tumor_fn_fp(y_true, y_pred, notumor_index=notumor_index)
    cost = compute_cost(fn, fp, revisions)

    return BusinessMetrics(
        auto_coverage=auto_cov,
        acc_high_conf=acc_high,
        high_conf_count=high_conf_count,
        cost_total=cost,
        fn=fn,
        fp=fp,
        revisions=revisions,
    )


def evaluate_high_confidence_operating_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.85, 0.88, 0.90, 0.92, 0.95),
) -> Dict[float, ConfidenceOperatingPoint]:
    out: Dict[float, ConfidenceOperatingPoint] = {}
    for threshold in thresholds:
        mask = conf >= threshold
        coverage = float(mask.mean())
        accuracy = float((y_true[mask] == y_pred[mask]).mean()) if int(mask.sum()) > 0 else float("nan")
        out[float(threshold)] = ConfidenceOperatingPoint(
            threshold=float(threshold),
            coverage=coverage,
            accuracy=accuracy,
        )
    return out
