from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    train_dir: Path
    test_dir: Path
    class_names: Tuple[str, ...]
    image_size: Tuple[int, int] = (224, 224)


def get_default_config(project_root: Path) -> DataConfig:
    train_dir = project_root / "data" / "Training"
    test_dir = project_root / "data" / "Testing"
    class_names = ("glioma", "meningioma", "notumor", "pituitary")
    return DataConfig(
        data_dir=project_root / "data",
        train_dir=train_dir,
        test_dir=test_dir,
        class_names=class_names,
    )


def list_images_by_class(
    root_dir: Path,
    class_names: Sequence[str],
    *,
    suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    allowed = set(suffixes)
    for class_name in class_names:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            out[class_name] = []
            continue
        out[class_name] = sorted(
            [path for path in class_dir.rglob("*") if path.suffix.lower() in allowed]
        )
    return out


def compute_class_counts(images_by_class: Dict[str, List[Path]]) -> Dict[str, int]:
    return {class_name: len(paths) for class_name, paths in images_by_class.items()}


def load_image_cv2(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    import cv2

    try:
        data = path.read_bytes()
    except Exception as exc:
        raise ValueError(f"Impossible de lire le fichier {path}: {type(exc).__name__}") from exc

    encoded = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32, copy=False)


def normalize_0_1(img: np.ndarray) -> np.ndarray:
    return img / 255.0


def make_sklearn_features(img: np.ndarray) -> np.ndarray:
    return img.reshape(-1)


def build_sklearn_dataset(
    images_by_class: Dict[str, List[Path]],
    class_names: Sequence[str],
    *,
    image_size: Tuple[int, int],
    n_per_class: int | None = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    rng = np.random.default_rng(random_state)
    features: List[np.ndarray] = []
    labels: List[int] = []
    used_paths: List[Path] = []
    skipped = 0

    for class_index, class_name in enumerate(class_names):
        class_paths = list(images_by_class.get(class_name, []))
        if shuffle:
            rng.shuffle(class_paths)
        if n_per_class is not None:
            class_paths = class_paths[:n_per_class]

        for path in class_paths:
            try:
                img = normalize_0_1(load_image_cv2(path, image_size))
            except ValueError:
                skipped += 1
                continue
            features.append(make_sklearn_features(img))
            labels.append(class_index)
            used_paths.append(path)

    if not features:
        raise RuntimeError(
            "Dataset vide. Verifie le dossier data/Training et data/Testing et les noms de classes."
        )
    if skipped > 0:
        print(f"[build_sklearn_dataset] images ignorees car illisibles: {skipped}")

    X = np.stack(features, axis=0).astype(np.float32, copy=False)
    y = np.array(labels, dtype=np.int64)
    return X, y, used_paths


def stratified_train_calibration_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_torch_dataloaders_from_imagefolder(
    train_dir: Path,
    test_dir: Path,
    *,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
):
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "torch et torchvision sont requis pour la partie CNN."
        ) from exc

    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split doit etre dans ]0, 1[.")

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    raw_train = datasets.ImageFolder(str(train_dir))
    n_samples = len(raw_train)
    if n_samples == 0:
        raise RuntimeError(f"Aucune image detectee dans {train_dir}.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split_index = int(round((1.0 - val_split) * n_samples))
    split_index = min(max(split_index, 1), n_samples - 1)
    train_indices = indices[:split_index].tolist()
    val_indices = indices[split_index:].tolist()

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(str(train_dir), transform=train_transform),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(str(train_dir), transform=eval_transform),
        val_indices,
    )
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader, raw_train.classes
