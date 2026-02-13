from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable

from .decision_engine import DecisionOutput


def format_percent(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def creer_rapport_decision(
    patient_id: str,
    scores_by_class: Dict[str, float],
    decision: DecisionOutput,
    *,
    report_date: str | None = None,
) -> str:
    if report_date is None:
        report_date = date.today().strftime("%d/%m/%Y")

    lines = [
        "RAPPORT AUTOMATISE",
        "========================================",
        "RAPPORT D'AIDE A LA DECISION",
        "========================================",
        f"Patient ID: {patient_id} Date: {report_date}",
        "",
        "PREDICTION PRINCIPALE",
        "---------------------",
        f"Classe: {decision.predicted_class}",
        f"Confiance: {format_percent(decision.max_prob)}",
        f"Niveau de certitude: {decision.certainty_level}",
        "",
        "SCORES PAR CLASSE",
        "-----------------",
    ]

    for class_name, score in sorted(scores_by_class.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {class_name}: {format_percent(score)}")

    lines.extend(
        [
            "",
            "RECOMMANDATIONS CLINIQUES",
            "--------------------------",
            f"Diagnostic: {decision.decision}",
            f"Action: {decision.action}",
            f"Priorite: {decision.priority}",
            (
                "Revision humaine: Oui - obligatoire"
                if decision.needs_human_review
                else "Revision humaine: Optionnelle (validation finale)"
            ),
            "",
            "ELEMENTS D'ATTENTION",
            "---------------------",
        ]
    )

    if decision.attention_notes:
        lines.extend([f"- {note}" for note in decision.attention_notes])
    else:
        lines.append("- Aucun")

    lines.append("========================================")
    return "\n".join(lines)


def save_reports_to_file(reports: Iterable[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(reports).strip() + "\n"
    output_path.write_text(content, encoding="utf-8")
