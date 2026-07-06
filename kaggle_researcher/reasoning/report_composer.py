from __future__ import annotations

import json
import re
from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    ReviewResult,
    ValidationResult,
)


SECTION_HEADINGS = [
    "Executive summary",
    "Тип соревнования и интерпретация метрики",
    "Анатомия датасета (по доступным описаниям, без EDA на реальных данных)",
    "Стратегия валидации",
    "Риски утечки и shake-up",
    "Разведка по публичным notebooks",
    "Паттерны похожих прошлых соревнований",
    "План baseline",
    "План feature engineering",
    "План моделей",
    "План ансамблирования",
    "Очередь экспериментов с приоритетами",
    "Стратегия выбора финальных сабмитов",
    "Чего не делать",
    "План первых 48 часов",
]

FORBIDDEN_REPORT_PHRASES = (
    "we analyzed train/test",
    "i analyzed train/test",
    "train/test data was analyzed",
    "train/test analysis was performed",
    "real eda showed",
    "eda showed",
    "we ran eda",
    "i ran eda",
    "adversarial validation found",
    "notebook execution showed",
    "we executed notebooks",
    "i executed notebooks",
    "confirmed leakage",
    "leakage confirmed",
    "leakage found",
    "chain-of-thought",
    "chain of thought",
)


def format_section_for_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


async def compose_report(
    competition_desc: str,
    plan_data: PlanData,
    domain_patterns: list[dict[str, Any]],
    validation_result: ValidationResult,
    leakage_result: LeakageRiskResult,
    metric_result: MetricResult,
    experiments: list[ExperimentItem],
    lb_audit: LeaderboardAuditResult,
    review: ReviewResult,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
) -> str:
    if client is None:
        raise ValueError("client is required")
    review_payload = review.model_dump(mode="json")
    payload = {
        "competition_desc": competition_desc,
        "plan_data": plan_data.model_dump(),
        "domain_patterns": domain_patterns,
        "validation_result": validation_result.model_dump(),
        "leakage_result": leakage_result.model_dump(),
        "metric_result": metric_result.model_dump(),
        "experiments": [item.model_dump() for item in experiments],
        "leaderboard_audit": lb_audit.model_dump(),
        "review_result": review_payload,
        "review_revised_sections_for_prompt": {
            name: format_section_for_prompt(section)
            for name, section in review_payload.get("revised_sections", {}).items()
        },
        "required_sections": SECTION_HEADINGS,
    }
    report_text = await client.chat_text(
        model=model,
        system_prompt=(
            "Compose the final Kaggle analyst v4 roadmap as markdown-like text. "
            "Use exactly the 15 required section headings in order. Include confidence "
            "where relevant. Include 'Чего не делать'. Do not claim real EDA, train/test "
            "analysis, notebook execution, or confirmed leakage. Do not include chain-of-thought. "
            "In the validation section, when temporal/stability signals exist, write this policy: "
            "Primary: out-of-time holdout on the latest periods plus rolling/expanding temporal CV. "
            "Secondary: StratifiedGroupKFold only as a robustness check if it does not violate "
            "chronological order. Explicitly say not to train on future periods and validate on "
            "past periods. Include provenance markers for key claims using this style: "
            "_Provenance: Kaggle + heuristic; not verified on data._ Tag validation, leakage, "
            "metric, model, feature, leaderboard, and do-not-do claims."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        timeout=180,
        max_tokens=6000,
    )
    validate_composed_report(report_text)
    return report_text


def validate_composed_report(report_text: str) -> None:
    found_headings = _extract_required_heading_lines(report_text)
    if found_headings != SECTION_HEADINGS:
        raise RuntimeError(
            "Composed report must contain exactly the 15 required section headings in order. "
            f"Found {len(found_headings)} matching headings."
        )
    lowered = report_text.lower()
    forbidden_phrase = next(
        (phrase for phrase in FORBIDDEN_REPORT_PHRASES if phrase in lowered),
        None,
    )
    if forbidden_phrase is not None:
        raise RuntimeError(
            "Composed report contains a forbidden data-execution or reasoning claim: "
            f"{forbidden_phrase!r}"
        )


def _extract_required_heading_lines(report_text: str) -> list[str]:
    required_by_lower = {heading.lower(): heading for heading in SECTION_HEADINGS}
    found: list[str] = []
    for line in report_text.splitlines():
        normalized = _normalize_heading_line(line)
        if normalized.lower() in required_by_lower:
            found.append(required_by_lower[normalized.lower()])
    return found


def _normalize_heading_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^\d+[\.)]\s+", "", text)
    return text.strip()
