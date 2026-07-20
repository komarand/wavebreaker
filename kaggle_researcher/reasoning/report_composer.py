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
    report_text = _canonicalize_report_headings(
        await client.chat_text(
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
    )
    try:
        validate_composed_report(report_text)
    except RuntimeError as exc:
        if "15 required section headings" not in str(exc):
            raise
        report_text = _canonicalize_report_headings(
            await _repair_report_sections(
                client=client,
                model=model,
                report_text=report_text,
                payload=payload,
            )
        )
        try:
            validate_composed_report(report_text)
        except RuntimeError as repair_exc:
            if "15 required section headings" not in str(repair_exc):
                raise
            report_text = repair_composed_report(
                report_text,
                competition_desc=competition_desc,
                plan_data=payload["plan_data"],
                domain_patterns=domain_patterns,
                validation_result=payload["validation_result"],
                leakage_result=payload["leakage_result"],
                metric_result=payload["metric_result"],
                experiments=payload["experiments"],
                lb_audit=payload["leaderboard_audit"],
                review=payload["review_result"],
            )
            validate_composed_report(report_text)
    return report_text


async def _repair_report_sections(
    client: DeepSeekClient,
    model: str,
    report_text: str,
    payload: dict[str, Any],
) -> str:
    return await client.chat_text(
        model=model,
        system_prompt=(
            "Repair the markdown roadmap structure only. Return the full report with exactly "
            "the required 15 headings in the required order. Preserve the existing substantive "
            "content as much as possible. Do not add new facts. Do not claim real EDA, train/test "
            "analysis, notebook execution, confirmed leakage, or chain-of-thought."
        ),
        user_prompt=json.dumps(
            {
                "required_sections": SECTION_HEADINGS,
                "original_inputs": payload,
                "report_to_repair": report_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        timeout=180,
        max_tokens=6000,
    )


def validate_composed_report(report_text: str) -> None:
    found_headings = _extract_required_heading_lines(_canonicalize_report_headings(report_text))
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


def extract_report_sections(report_text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    current_lines: list[str] = []
    preamble: list[str] = []

    for line in report_text.splitlines():
        canonical = normalize_report_heading(line)
        if canonical is None:
            if current_heading is None:
                if line.strip():
                    preamble.append(line)
            else:
                current_lines.append(line)
            continue
        if current_heading is not None:
            sections.setdefault(current_heading, []).append("\n".join(current_lines).strip())
        elif preamble:
            sections.setdefault(SECTION_HEADINGS[0], []).append("\n".join(preamble).strip())
            preamble = []
        current_heading = canonical
        current_lines = []

    if current_heading is not None:
        sections.setdefault(current_heading, []).append("\n".join(current_lines).strip())
    elif preamble:
        sections.setdefault(SECTION_HEADINGS[0], []).append("\n".join(preamble).strip())

    return {
        heading: "\n\n".join(part for part in parts if part).strip()
        for heading, parts in sections.items()
        if any(part for part in parts)
    }


def normalize_report_heading(line: str) -> str | None:
    return _canonical_heading_for_line(line)


def repair_composed_report(
    malformed_text: str,
    *,
    competition_desc: str,
    plan_data: dict[str, Any],
    domain_patterns: list[dict[str, Any]],
    validation_result: dict[str, Any],
    leakage_result: dict[str, Any],
    metric_result: dict[str, Any],
    experiments: list[dict[str, Any]],
    lb_audit: dict[str, Any],
    review: dict[str, Any],
) -> str:
    recovered = extract_report_sections(malformed_text)
    blocks: list[str] = []
    for index, heading in enumerate(SECTION_HEADINGS):
        content = recovered.get(heading, "").strip()
        if content:
            body = _repaired_section_note(recovered=True)
            body.extend(_clean_section_content(content))
        else:
            body = _fallback_section_body(
                index=index,
                competition_desc=competition_desc,
                plan_data=plan_data,
                domain_patterns=domain_patterns,
                validation_result=validation_result,
                leakage_result=leakage_result,
                metric_result=metric_result,
                experiments=experiments,
                lb_audit=lb_audit,
                review=review,
            )
        blocks.append(f"## {heading}\n" + "\n".join(body).strip())
    return "\n\n".join(blocks) + "\n"


def _repaired_section_note(*, recovered: bool) -> list[str]:
    reason = (
        "the recovered LLM content was placed under the canonical heading"
        if recovered
        else "the LLM response did not provide a valid canonical section"
    )
    return [
        (
            "Repair note: This section was reconstructed because "
            f"{reason}."
        )
    ]


def _fallback_section_body(
    *,
    index: int,
    competition_desc: str,
    plan_data: dict[str, Any],
    domain_patterns: list[dict[str, Any]],
    validation_result: dict[str, Any],
    leakage_result: dict[str, Any],
    metric_result: dict[str, Any],
    experiments: list[dict[str, Any]],
    lb_audit: dict[str, Any],
    review: dict[str, Any],
) -> list[str]:
    lines = _repaired_section_note(recovered=False)
    task_type = plan_data.get("task_type") or "unknown"
    metric = plan_data.get("metric") or "unknown"
    domain = plan_data.get("domain") or "unknown"

    if index == 0:
        lines.append(
            f"- Conservative summary: task type `{task_type}`, metric `{metric}`, domain `{domain}`."
        )
        warnings = lb_audit.get("warnings") or []
        if warnings:
            lines.append(f"- Top warning: {_first_text(warnings)}")
        lines.append("- No train/test data analysis is claimed in this source-based report.")
    elif index == 1:
        lines.append(f"- Metric: `{metric}`.")
        lines.extend(_dict_bullets(metric_result, keys=("metric_explanation", "surrogate_loss_suggestion")))
        lines.append(f"- Needs calibration: `{metric_result.get('needs_calibration', 'unknown')}`.")
    elif index == 2:
        lines.append("- Dataset anatomy is based on source descriptions only; no train/test files were inspected here.")
        lines.append(f"- Competition description snapshot: {_short_text(competition_desc)}")
    elif index == 3:
        lines.extend(_dict_bullets(validation_result, keys=("recommended_cv", "likely_split", "validation_risk", "reasoning")))
    elif index == 4:
        lines.extend(_dict_bullets(leakage_result, keys=("risk_level",)))
        lines.extend(_list_bullets("Possible issue", leakage_result.get("possible_issues")))
        lines.extend(_list_bullets("Recommended check", leakage_result.get("recommended_checks")))
        lines.extend(_dict_bullets(lb_audit, keys=("shake_up_risk", "public_lb_trust")))
    elif index == 5:
        lines.append("- Public notebooks were analyzed only as text/source material; notebooks were not executed.")
    elif index == 6:
        if domain_patterns:
            for item in domain_patterns[:3]:
                family = item.get("competition_family") or item.get("family") or "similar competition"
                validation = item.get("typical_validation") or item.get("validation") or "validation pattern not specified"
                lines.append(f"- `{family}`: {validation}.")
        else:
            lines.append("- No similar-competition pattern evidence was available.")
    elif index == 7:
        lines.append("- Start with a minimal, honest baseline aligned with the selected validation policy.")
        lines.append("- Treat baseline output as a sanity check, not as final optimization.")
    elif index == 8:
        lines.append("- Derive feature work from source-supported hypotheses and validation constraints.")
        lines.append("- Avoid claiming feature effects until they are measured in a later data-execution step.")
    elif index == 9:
        lines.append("- Use models appropriate for the task type and metric.")
        lines.append("- Prefer simple, auditable candidates before heavier experiments.")
    elif index == 10:
        lines.append("- Delay ensembling until single-model baselines and validation are stable.")
    elif index == 11:
        if experiments:
            for experiment in experiments:
                lines.append(
                    f"- `{experiment.get('priority', 'P?')}` {experiment.get('experiment', 'Experiment')}: "
                    f"{experiment.get('why', 'No rationale recorded.')}"
                )
        else:
            lines.append("- No structured experiments were available.")
    elif index == 12:
        lines.extend(_dict_bullets(lb_audit, keys=("submission_selection_rule", "public_lb_trust", "shake_up_risk")))
    elif index == 13:
        lines.extend(_list_bullets("Avoid", review.get("unnecessary_experiments")))
        lines.extend(_list_bullets("Unsupported claim", review.get("unsupported_claims")))
        if len(lines) == 1:
            lines.append("- Avoid unsupported claims and public leaderboard overfitting.")
    elif index == 14:
        lines.append("- First 48 hours: verify metric, validation, leakage checks, and baseline feasibility.")
        lines.append("- Convert source-based risks into measurable experiments before making strong claims.")
    else:
        lines.append("- No structured fallback content was available for this section.")
    return lines


def _clean_section_content(content: str) -> list[str]:
    lines = [line.rstrip() for line in content.splitlines()]
    cleaned = [
        _sanitize_recovered_line(line)
        for line in lines
        if normalize_report_heading(line) is None
    ]
    return cleaned or ["- Recovered section content was empty after heading normalization."]


def _sanitize_recovered_line(line: str) -> str:
    lowered = line.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_REPORT_PHRASES):
        return "- Repair note: An unsupported data-execution claim from the LLM response was omitted."
    return line


def _dict_bullets(payload: dict[str, Any], *, keys: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        label = key.replace("_", " ")
        lines.append(f"- {label}: {_short_text(value)}")
    return lines


def _list_bullets(label: str, values: Any) -> list[str]:
    if not values:
        return []
    if not isinstance(values, list):
        values = [values]
    return [f"- {label}: {_short_text(value)}" for value in values[:5]]


def _first_text(values: Any) -> str:
    if isinstance(values, list) and values:
        return _short_text(values[0])
    return _short_text(values)


def _short_text(value: Any, *, limit: int = 220) -> str:
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text or "not specified"


def _extract_required_heading_lines(report_text: str) -> list[str]:
    found: list[str] = []
    for line in report_text.splitlines():
        canonical = _canonical_heading_for_line(line)
        if canonical is not None:
            found.append(canonical)
    return found


def _canonicalize_report_headings(report_text: str) -> str:
    lines: list[str] = []
    for line in report_text.splitlines():
        canonical = _canonical_heading_for_line(line)
        if canonical is None:
            lines.append(line)
        else:
            lines.append(f"## {canonical}")
    return "\n".join(lines)


def _canonical_heading_for_line(line: str) -> str | None:
    normalized = _normalize_heading_line(line)
    if not normalized:
        return None
    lookup = _heading_lookup()
    for candidate in _heading_variants(normalized):
        canonical = lookup.get(_heading_key(candidate))
        if canonical is not None:
            return canonical
    return None


def _heading_variants(value: str) -> list[str]:
    variants = [value]
    try:
        repaired = value.encode("cp1251").decode("utf-8")
    except UnicodeError:
        repaired = ""
    if repaired and repaired not in variants:
        variants.append(repaired)
    return variants


def _heading_lookup() -> dict[str, str]:
    lookup = {_heading_key(heading): heading for heading in SECTION_HEADINGS}
    aliases = {
        SECTION_HEADINGS[1]: [
            "Тип соревнования и метрика",
            "Тип соревнования и интерпретация metric",
        ],
        SECTION_HEADINGS[2]: [
            "Анатомия датасета",
            "Данные и анатомия датасета",
            "Анатомия данных",
        ],
        SECTION_HEADINGS[4]: [
            "Риски утечки и shakeup",
            "Риски утечки и shake up",
            "Утечки и shake-up",
        ],
        SECTION_HEADINGS[5]: [
            "Разведка публичных notebooks",
            "Разведка по публичным ноутбукам",
        ],
        SECTION_HEADINGS[6]: [
            "Паттерны похожих соревнований",
            "Паттерны прошлых соревнований",
        ],
        SECTION_HEADINGS[11]: [
            "Очередь экспериментов",
            "План экспериментов",
            "Experiment queue",
        ],
        SECTION_HEADINGS[12]: [
            "Стратегия финальных сабмитов",
            "Выбор финальных сабмитов",
        ],
        SECTION_HEADINGS[13]: [
            "Что не делать",
            "Do not do",
        ],
        SECTION_HEADINGS[14]: [
            "Первые 48 часов",
            "План на первые 48 часов",
            "48-hour action plan",
        ],
    }
    for canonical, variants in aliases.items():
        for variant in variants:
            lookup[_heading_key(variant)] = canonical
    return lookup


def _heading_key(value: str) -> str:
    text = value.strip().lower().replace("ё", "е")
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.replace("shake-up", "shakeup").replace("shake up", "shakeup")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_heading_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^\d+[\.)]\s+", "", text)
    return text.strip()
