from __future__ import annotations

from datetime import date
from typing import Dict

from .decision_engine import DecisionOutput


def format_percent(x: float) -> str:
    # Pourcentage court
    return f"{100.0 * x:.1f}%"


def creer_rapport_decision(
    patient_id: str,
    scores_by_class: Dict[str, float],
    decision: DecisionOutput,
    *,
    report_date: str | None = None,
) -> str:
    # Génère le rapport textuel
    if report_date is None:
        report_date = date.today().strftime("%d/%m/%Y")

    lines = []
    lines.append("RAPPORT AUTOMATISE")
    lines.append("========================================")
    lines.append("RAPPORT D'AIDE A LA DECISION")
    lines.append("========================================")
    lines.append(f"Patient ID: {patient_id} Date: {report_date}")
    lines.append("")
    lines.append("PREDICTION PRINCIPALE")
    lines.append("---------------------")
    lines.append(f"Classe: {decision.predicted_class}")
    lines.append(f"Confiance: {format_percent(decision.max_prob)}")
    lines.append("")
    lines.append("SCORES PAR CLASSE")
    lines.append("-----------------")
    for k, v in sorted(scores_by_class.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {k}: {format_percent(v)}")
    lines.append("")
    lines.append("RECOMMANDATIONS CLINIQUES")
    lines.append("--------------------------")
    lines.append(f"Diagnostic: {decision.decision}")
    lines.append(f"Action: {decision.action}")
    lines.append(f"Priorite: {decision.priority}")
    lines.append(f"Revision humaine: {'Oui' if decision.needs_human_review else 'Optionnelle'}")
    lines.append("")
    lines.append("ELEMENTS D'ATTENTION")
    lines.append("---------------------")
    if decision.attention_notes:
        for n in decision.attention_notes:
            lines.append(f"- {n}")
    else:
        lines.append("- Aucun")
    lines.append("========================================")
    return "\n".join(lines)
