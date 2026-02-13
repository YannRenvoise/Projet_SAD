from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class CalibratedModel:
    # Wrapper simple
    base_model: Any
    calibrator: Any

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Probas calibrées
        return self.calibrator.predict_proba(X)


def calibrate_sklearn_classifier(
    model: Any,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    method: str = "isotonic",
) -> CalibratedModel:
    # Calibration d'un modèle déjà fit (API sklearn récente)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
    )
    cal.fit(X_calib, y_calib)
    return CalibratedModel(base_model=model, calibrator=cal)


def temperature_scaling_fit(logits: np.ndarray, y_true: np.ndarray) -> float:
    # Ajuste une température sur logits (simple)
    # TODO: optimiser T (ex: minimiser NLL) sur un set de calibration
    return 1.0


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    # Applique T sur logits puis softmax
    z = logits / max(temperature, 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
