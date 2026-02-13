from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DecisionThresholds:
    high: float = 0.85
    medium: float = 0.65
    low: float = 0.50
    notumor_safety: float = 0.95


@dataclass(frozen=True)
class DecisionOutput:
    predicted_class: str
    max_prob: float
    certainty_level: str
    decision: str
    action: str
    priority: str
    needs_human_review: bool
    attention_notes: List[str]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _is_notumor(label: str) -> bool:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    return normalized in {"notumor", "no tumor", "pas de tumeur"}


def determine_urgency(predicted_class: str) -> str:
    if predicted_class == "glioma":
        return "[!] URGENT - Prise en charge sous 12h"
    if predicted_class == "meningioma":
        return "SURVEILLANCE - Controle programme"
    if predicted_class == "pituitary":
        return "SPECIALISE - Endocrinologie/Neurochirurgie"
    return "RASSURER - Validation de routine"


def predire_avec_confiance(
    image: np.ndarray,
    model,
    class_names: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 1:
        image_array = image_array.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(image_array), dtype=np.float64)
    else:
        logits = model(image_array)
        if hasattr(logits, "detach") and hasattr(logits, "cpu") and hasattr(logits, "numpy"):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probabilities = _softmax(logits)

    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError("Le modele doit retourner une seule prediction a la fois.")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Le nombre de classes ne correspond pas aux probabilites retournees.")

    probs = probabilities[0]
    scores_by_class = {
        class_names[i]: float(probs[i]) for i in range(len(class_names))
    }
    predicted_class = max(scores_by_class, key=scores_by_class.get)
    return predicted_class, scores_by_class


def _certainty_level(max_prob: float, thresholds: DecisionThresholds) -> str:
    if max_prob >= thresholds.high:
        return "ELEVE [OK]"
    if max_prob >= thresholds.medium:
        return "MOYEN [REVISION]"
    if max_prob >= thresholds.low:
        return "FAIBLE [REVISION]"
    return "TRES FAIBLE [ALERTE]"


def generer_recommandation(
    probabilities: Dict[str, float],
    thresholds: DecisionThresholds,
) -> DecisionOutput:
    if not probabilities:
        raise ValueError("probabilities ne peut pas etre vide.")

    predicted_class, max_prob = max(probabilities.items(), key=lambda kv: kv[1])
    max_prob = float(max_prob)
    notes: List[str] = []

    if max_prob >= thresholds.high:
        decision = "Diagnostic automatique valide"
        action = "Rapport envoye au medecin traitant"
        priority = determine_urgency(predicted_class)
        needs_review = False
    elif max_prob >= thresholds.medium:
        decision = "Diagnostic probable - Revision recommandee"
        action = "Validation par radiologue junior"
        priority = "Normale (48h)"
        needs_review = True
    elif max_prob >= thresholds.low:
        decision = "Cas incertain"
        action = "Revision par radiologue senior"
        priority = "Elevee (24h)"
        needs_review = True
    else:
        decision = "Incertitude elevee"
        action = "Double lecture obligatoire + IRM complementaire"
        priority = "Urgente (12h)"
        needs_review = True

    if _is_notumor(predicted_class) and max_prob < thresholds.notumor_safety:
        notes.append("Verification obligatoire (risque faux negatif)")
        action = "Verification obligatoire par radiologue senior"
        priority = "Elevee (24h)"
        needs_review = True

    if predicted_class == "glioma":
        notes.append("Suspicion tumeur maligne")

    certainty_level = _certainty_level(max_prob, thresholds)
    return DecisionOutput(
        predicted_class=predicted_class,
        max_prob=max_prob,
        certainty_level=certainty_level,
        decision=decision,
        action=action,
        priority=priority,
        needs_human_review=needs_review,
        attention_notes=notes,
    )
