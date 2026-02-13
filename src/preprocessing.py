from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    # Chemins et paramètres data
    data_dir: Path
    train_dir: Path
    test_dir: Path
    class_names: List[str]
    image_size: Tuple[int, int] = (224, 224)


def get_default_config(project_root: Path) -> DataConfig:
    # Convention projet: data/Training et data/Testing
    train_dir = project_root / "data" / "Training"
    test_dir = project_root / "data" / "Testing"
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    return DataConfig(
        data_dir=project_root / "data",
        train_dir=train_dir,
        test_dir=test_dir,
        class_names=class_names,
    )


def list_images_by_class(root_dir: Path, class_names: List[str]) -> Dict[str, List[Path]]:
    # Retourne la liste des images pour chaque classe
    out: Dict[str, List[Path]] = {}
    for c in class_names:
        c_dir = root_dir / c
        out[c] = sorted([p for p in c_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return out


def compute_class_counts(images_by_class: Dict[str, List[Path]]) -> Dict[str, int]:
    # Comptage simple
    return {k: len(v) for k, v in images_by_class.items()}


def load_image_cv2(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    # Lecture robuste (support chemins Unicode Windows) + resize + float32
    import cv2  # import local
    import numpy as np

    try:
        data = path.read_bytes()
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier: {path} ({type(e).__name__})")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Image illisible (imdecode): {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)




def normalize_0_1(img: np.ndarray) -> np.ndarray:
    # Normalisation basique [0,1]
    return img / 255.0


def make_sklearn_features(img: np.ndarray) -> np.ndarray:
    # Baseline: flatten (simple)
    return img.reshape(-1)
