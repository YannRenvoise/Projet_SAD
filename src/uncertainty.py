from __future__ import annotations

from typing import Dict, List

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch n'est pas installe. Installe les dependances via `pip install -r requirements.txt`."
        ) from exc
    return torch


def _activate_dropout_layers(model) -> None:
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def calculer_incertitude_mc_dropout(image_tensor, model, n_iter: int = 20) -> Dict[str, float]:
    torch = _require_torch()
    if n_iter < 2:
        raise ValueError("n_iter doit etre >= 2 pour estimer une incertitude.")

    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device)

    was_training = model.training
    model.eval()
    _activate_dropout_layers(model)

    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_iter):
            logits = model(image_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("MC Dropout attend un batch de taille 1.")
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probabilities.append(probs)

    model.train(was_training)

    probs_arr = np.stack(probabilities, axis=0)
    mean_probs = probs_arr.mean(axis=0)
    std_probs = probs_arr.std(axis=0)

    eps = 1e-12
    predictive_entropy = float(-(mean_probs * np.log(mean_probs + eps)).sum())
    expected_entropy = float(-(probs_arr * np.log(probs_arr + eps)).sum(axis=1).mean())
    mutual_information = predictive_entropy - expected_entropy

    return {
        "entropy_predictive": predictive_entropy,
        "entropy_expected": expected_entropy,
        "mutual_information": float(mutual_information),
        "mean_max_prob": float(mean_probs.max()),
        "std_max_prob": float(std_probs[mean_probs.argmax()]),
    }
