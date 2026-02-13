from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    history: Dict[str, List[float]] = field(default_factory=dict)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int = 2000,
    random_state: int = 42,
) -> TrainResult:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    return TrainResult(model=clf, metrics={"train_accuracy": train_acc})


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layers: Tuple[int, ...] = (256, 128),
    max_iter: int = 120,
    random_state: int = 42,
) -> TrainResult:
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=random_state,
    )
    mlp.fit(X_train, y_train)
    train_acc = float(mlp.score(X_train, y_train))
    return TrainResult(model=mlp, metrics={"train_accuracy": train_acc})


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch n'est pas installe. Installe les dependances via `pip install -r requirements.txt`."
        ) from exc
    return torch


def _resolve_device(device: str | None):
    torch = _require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cnn_torch(num_classes: int = 4, dropout_p: float = 0.30):
    _require_torch()
    import torch.nn as nn

    class DecisionCNN(nn.Module):
        def __init__(self, n_classes: int, drop_p: float):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop_p),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop_p),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(drop_p),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return DecisionCNN(num_classes, dropout_p)


def train_cnn_classifier(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> TrainResult:
    torch = _require_torch()

    dev = _resolve_device(device)
    model = model.to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state: Dict[str, Any] | None = None
    best_val_acc = -1.0

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += batch_size

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))

        model.eval()
        val_loss_acc = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                logits = model(inputs)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_acc += float(loss.item()) * batch_size
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_total += batch_size

        val_loss = val_loss_acc / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "final_train_accuracy": history["train_acc"][-1],
        "final_val_accuracy": history["val_acc"][-1],
    }
    return TrainResult(model=model, metrics=metrics, history=history)


def predict_cnn_logits(model, data_loader, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    torch = _require_torch()
    dev = _resolve_device(device)
    model = model.to(dev)
    model.eval()

    logits_batches: List[np.ndarray] = []
    labels_batches: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dev)
            logits = model(inputs).cpu().numpy()
            logits_batches.append(logits)
            labels_batches.append(labels.numpy())

    if not logits_batches:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(logits_batches, axis=0),
        np.concatenate(labels_batches, axis=0),
    )
