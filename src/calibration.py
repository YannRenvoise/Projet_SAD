from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: Any

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(X)


def calibrate_sklearn_classifier(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    method: str = "isotonic",
) -> CalibratedModel:
    from sklearn.calibration import CalibratedClassifierCV

    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("method doit etre 'isotonic' ou 'sigmoid'.")

    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )
    except ImportError:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )

    calibrator.fit(X_calib, y_calib)
    return CalibratedModel(base_model=model, calibrator=calibrator)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature doit etre strictement positive.")

    z = logits / temperature
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _negative_log_likelihood(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def temperature_scaling_fit(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    temperatures: Iterable[float] | None = None,
) -> float:
    if logits.ndim != 2:
        raise ValueError("logits doit etre de forme (n_samples, n_classes).")
    if len(logits) != len(y_true):
        raise ValueError("logits et y_true doivent avoir la meme taille.")
    if len(logits) == 0:
        raise ValueError("logits vide.")

    if temperatures is None:
        temperatures = np.logspace(np.log10(0.05), np.log10(10.0), 250)

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in temperatures:
        probs = apply_temperature(logits, float(temperature))
        nll = _negative_log_likelihood(probs, y_true)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    if probabilities.ndim != 2:
        raise ValueError("probabilities doit etre de forme (n_samples, n_classes).")
    if len(probabilities) != len(y_true):
        raise ValueError("probabilities et y_true doivent avoir la meme taille.")
    if n_bins <= 0:
        raise ValueError("n_bins doit etre > 0.")

    conf = probabilities.max(axis=1)
    pred = probabilities.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(probabilities)
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        in_bin = (conf > left) & (conf <= right) if i > 0 else (conf >= left) & (conf <= right)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(conf[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        ece += abs(avg_acc - avg_conf) * (count / n_samples)

    return float(ece)
