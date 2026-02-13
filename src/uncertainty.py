from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def calculer_incertitude_mc_dropout(image_tensor, model, n_iter: int = 20) -> Dict[str, float]:
    # Estime incertitude via MC Dropout (PyTorch attendu)
    import torch

    model.train()  # active dropout
    probs: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_iter):
            logits = model(image_tensor)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)

    probs_arr = np.concatenate(probs, axis=0)  # (n_iter, n_classes)
    mean_p = probs_arr.mean(axis=0)
    std_p = probs_arr.std(axis=0)

    # Mesures simples: entropie et variance
    eps = 1e-9
    entropy = float(-(mean_p * np.log(mean_p + eps)).sum())

    return {
        "entropy": entropy,
        "mean_max_prob": float(mean_p.max()),
        "std_max_prob": float(std_p[mean_p.argmax()]),
    }
