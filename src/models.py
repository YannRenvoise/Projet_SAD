from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    history: Dict[str, List[float]] = field(default_factory=dict)


def _resolve_pca_components(n_samples: int, n_features: int, requested: int | None) -> int | None:
    if requested is None or requested <= 0:
        return None
    max_allowed = max(2, min(n_samples - 1, n_features))
    return min(requested, max_allowed)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int = 2000,
    random_state: int = 42,
    class_weight: str | dict | None = "balanced",
    c_value: float = 2.0,
    pca_components: int | None = 256,
) -> TrainResult:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    n_comp = _resolve_pca_components(X_train.shape[0], X_train.shape[1], pca_components)
    if n_comp is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_comp,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight=class_weight,
                C=c_value,
                random_state=random_state,
            ),
        )
    )

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    train_acc = float(pipeline.score(X_train, y_train))
    return TrainResult(
        model=pipeline,
        metrics={
            "train_accuracy": train_acc,
            "pca_components": float(n_comp or 0),
        },
    )


def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layers: Tuple[int, ...] = (512, 256),
    max_iter: int = 220,
    random_state: int = 42,
    alpha: float = 1e-4,
    pca_components: int | None = 384,
) -> TrainResult:
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    n_comp = _resolve_pca_components(X_train.shape[0], X_train.shape[1], pca_components)
    if n_comp is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_comp,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size=64,
                learning_rate_init=5e-4,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=12,
                random_state=random_state,
            ),
        )
    )

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    train_acc = float(pipeline.score(X_train, y_train))
    return TrainResult(
        model=pipeline,
        metrics={
            "train_accuracy": train_acc,
            "pca_components": float(n_comp or 0),
        },
    )


from . import _require_torch


def _resolve_device(device: str | None):
    torch = _require_torch()
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cnn_torch(
    num_classes: int = 4,
    dropout_p: float = 0.30,
    backbone: str = "custom",
    pretrained: bool = False,
):
    _require_torch()
    import torch.nn as nn

    if backbone == "resnet18":
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("torchvision est requis pour backbone='resnet18'.") from exc

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if backbone != "custom":
        raise ValueError("backbone doit etre 'custom' ou 'resnet18'.")

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
                nn.MaxPool2d(2),
                nn.Dropout2d(drop_p),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(drop_p),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return DecisionCNN(num_classes, dropout_p)


def compute_class_weights_from_loader(train_loader, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, labels in train_loader:
        arr = labels.numpy().astype(np.int64)
        counts += np.bincount(arr, minlength=num_classes)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def train_cnn_classifier(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int = 14,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    class_weights: np.ndarray | None = None,
    label_smoothing: float = 0.05,
    early_stopping_patience: int | None = 5,
    grad_clip_norm: float | None = 1.0,
    device: str | None = None,
) -> TrainResult:
    torch = _require_torch()

    dev = _resolve_device(device)
    model = model.to(dev)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=dev)

    criterion = torch.nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_state: Dict[str, Any] | None = None
    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += batch_size

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        model.eval()
        val_loss_acc = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_acc += float(loss.item()) * batch_size
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_total += batch_size

        val_loss = val_loss_acc / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = float(val_acc)
            best_epoch = epoch + 1
            epochs_without_improve = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improve += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improve >= early_stopping_patience
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": float(best_epoch),
        "final_train_accuracy": history["train_acc"][-1],
        "final_val_accuracy": history["val_acc"][-1],
    }
    return TrainResult(model=model, metrics=metrics, history=history)


def predict_cnn_logits(
    model,
    data_loader,
    *,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    torch = _require_torch()
    dev = _resolve_device(device)
    model = model.to(dev)
    model.eval()

    logits_batches: List[np.ndarray] = []
    labels_batches: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dev, non_blocking=True)
            logits = model(inputs).cpu().numpy()
            logits_batches.append(logits)
            labels_batches.append(labels.numpy())

    if not logits_batches:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return (
        np.concatenate(logits_batches, axis=0),
        np.concatenate(labels_batches, axis=0),
    )
