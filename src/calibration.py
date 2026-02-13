from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

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


def calibrate_with_best_method(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    methods: Sequence[str] = ("sigmoid", "isotonic"),
    eval_split: float = 0.4,
    random_state: int = 42,
) -> tuple[CalibratedModel, Dict[str, Dict[str, float]], str]:
    if not methods:
        raise ValueError("methods ne peut pas etre vide.")
    if not (0.0 < eval_split < 1.0):
        raise ValueError("eval_split doit etre strictement entre 0 et 1.")
    if len(X_calib) != len(y_calib):
        raise ValueError("X_calib et y_calib doivent avoir la meme taille.")
    if len(y_calib) < 2:
        raise ValueError("y_calib doit contenir au moins 2 echantillons.")

    allowed_methods = {"sigmoid", "isotonic"}
    unique_methods: list[str] = []
    seen_methods: set[str] = set()
    for method in methods:
        if method not in allowed_methods:
            raise ValueError(f"method invalide: {method}. Utilise 'sigmoid' ou 'isotonic'.")
        if method not in seen_methods:
            unique_methods.append(method)
            seen_methods.add(method)

    from sklearn.model_selection import train_test_split

    classes, counts = np.unique(y_calib, return_counts=True)
    n_classes = len(classes)
    n_samples = len(y_calib)
    n_eval = int(np.ceil(eval_split * n_samples))
    n_fit = n_samples - n_eval

    # Split holdout uniquement si chaque classe peut apparaitre dans les deux sous-ensembles.
    holdout_feasible = (
        counts.min() >= 2
        and n_eval >= n_classes
        and n_fit >= n_classes
    )

    if holdout_feasible:
        X_cfit, X_ceval, y_cfit, y_ceval = train_test_split(
            X_calib,
            y_calib,
            test_size=eval_split,
            random_state=random_state,
            stratify=y_calib,
        )
    else:
        # Fallback stable: pas de holdout quand le split ne peut pas etre representatif.
        X_cfit, y_cfit = X_calib, y_calib
        X_ceval, y_ceval = X_calib, y_calib

    evaluations: Dict[str, Dict[str, float]] = {}
    best_method = None
    best_nll = float("inf")
    best_ece = float("inf")

    for method in unique_methods:
        calibrated = calibrate_sklearn_classifier(
            model=model,
            X_calib=X_cfit,
            y_calib=y_cfit,
            method=method,
        )
        probs = calibrated.predict_proba(X_ceval)
        nll = _negative_log_likelihood(probs, y_ceval)
        ece = expected_calibration_error(probs, y_ceval)
        evaluations[method] = {
            "nll": float(nll),
            "ece": float(ece),
        }

        better = (nll < best_nll) or (abs(nll - best_nll) <= 1e-8 and ece < best_ece)
        if better:
            best_nll = float(nll)
            best_ece = float(ece)
            best_method = method

    if best_method is None:
        raise RuntimeError("Impossible de selectionner une calibration.")

    best_model = calibrate_sklearn_classifier(
        model=model,
        X_calib=X_calib,
        y_calib=y_calib,
        method=best_method,
    )
    return best_model, evaluations, best_method


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
