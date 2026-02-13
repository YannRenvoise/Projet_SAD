from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DecisionThresholds:
    # Seuils du sujet
    high: float = 0.85
    medium: float = 0.65
    low: float = 0.50
    notumor_safety: float = 0.95  # seuil spécifique faux négatifs


@dataclass(frozen=True)
class DecisionOutput:
    # Sortie décisionnelle
    predicted_class: str
    max_prob: float
    decision: str
    action: str
    priority: str
    needs_human_review: bool
    attention_notes: List[str]


def determine_urgency(predicted_class: str) -> str:
    # Priorité clinique selon classe (conforme au sujet)
    if predicted_class == "glioma":
        return "URGENT"
    if predicted_class == "meningioma":
        return "SURVEILLANCE"
    if predicted_class == "pituitary":
        return "SPECIALISE"
    return "RASSURER"


def generer_recommandation(
    probabilities: Dict[str, float],
    thresholds: DecisionThresholds,
) -> DecisionOutput:
    # Applique les règles de décision du sujet
    predicted_class, max_prob = max(probabilities.items(), key=lambda kv: kv[1])
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

    # Tolérance asymétrique faux négatifs
    if predicted_class == "notumor" and max_prob < thresholds.notumor_safety:
        notes.append("Verification obligatoire (risque faux negatif)")
        needs_review = True

    return DecisionOutput(
        predicted_class=predicted_class,
        max_prob=float(max_prob),
        decision=decision,
        action=action,
        priority=priority,
        needs_human_review=needs_review,
        attention_notes=notes,
    )
