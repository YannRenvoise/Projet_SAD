from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TrainResult:
    # Résultat d'entraînement basique
    model: Any
    metrics: Dict[str, float]


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, *, max_iter: int = 2000) -> TrainResult:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return TrainResult(model=clf, metrics={})



def train_mlp_classifier(X_train: np.ndarray, y_train: np.ndarray, *, hidden_layers: Tuple[int, ...] = (256, 128)) -> TrainResult:
    # MLP scikit-learn (simple)
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
    )
    mlp.fit(X_train, y_train)
    return TrainResult(model=mlp, metrics={})


def build_cnn_torch(num_classes: int = 4, image_size: Tuple[int, int] = (224, 224)):
    # CNN PyTorch minimal (à adapter)
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(128, n_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return SimpleCNN(num_classes)
