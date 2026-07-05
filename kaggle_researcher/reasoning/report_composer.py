from __future__ import annotations

import json
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
    "Анатомия датасета",
    "Стратегия валидации",
    "Риски утечки и shake-up",
    "Разведка по публичным notebooks",
    "Паттерны похожих прошлых соревнований",
    "План baseline",
    "План feature engineering",
    "План моделей",
    "План ансамблирования",
    "Очередь экспериментов",
    "Стратегия выбора финальных сабмитов",
    "Чего не делать",
    "План первых 48 часов",
]


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
    payload = {
        "competition_desc": competition_desc,
        "plan_data": plan_data.model_dump(),
        "domain_patterns": domain_patterns,
        "validation_result": validation_result.model_dump(),
        "leakage_result": leakage_result.model_dump(),
        "metric_result": metric_result.model_dump(),
        "experiments": [item.model_dump() for item in experiments],
        "leaderboard_audit": lb_audit.model_dump(),
        "review_result": review.model_dump(),
        "required_sections": SECTION_HEADINGS,
    }
    return await client.chat_text(
        model=model,
        system_prompt=(
            "Compose the final Kaggle analyst v4 roadmap as markdown-like text. "
            "Use exactly the 15 required section headings in order. Include confidence "
            "where relevant. Include 'Чего не делать'. Do not claim real EDA, train/test "
            "analysis, notebook execution, or confirmed leakage. Do not include chain-of-thought."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        timeout=180,
        max_tokens=6000,
    )
