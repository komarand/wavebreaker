from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from kaggle_researcher.reasoning.report_composer import (
    FORBIDDEN_REPORT_PHRASES,
    SECTION_HEADINGS,
    _canonicalize_report_headings,
    _extract_required_heading_lines,
)
from kaggle_researcher.schemas import RetrievedDocument


KEY_REASONING_OUTPUTS = {
    "metric",
    "metric_result",
    "validation",
    "validation_result",
    "leakage",
    "leakage_result",
    "leaderboard",
    "leaderboard_audit",
    "review",
    "review_result",
}

CONFIDENCE_OUTPUTS = KEY_REASONING_OUTPUTS
EVIDENCE_OUTPUTS = KEY_REASONING_OUTPUTS | {"experiments", "experiment_planner"}
LEAKAGE_OUTPUTS = {"leakage", "leakage_result", "leakage_risk_analyst"}
LEAKAGE_CONFIRMED_PHRASES = (
    "leakage is confirmed",
    "confirmed leakage",
    "leakage confirmed",
    "leakage was confirmed",
    "data leakage is confirmed",
)


def validate_reasoning_outputs(outputs: dict) -> list[str]:
    warnings: list[str] = []
    normalized_outputs = _to_plain(outputs)
    if not isinstance(normalized_outputs, dict):
        return ["Reasoning outputs must be a dict."]

    for name, value in normalized_outputs.items():
        output_name = str(name)
        output_key = output_name.lower()
        if output_key in CONFIDENCE_OUTPUTS:
            _check_mapping_field(
                value=value,
                field="confidence",
                output_name=output_name,
                warnings=warnings,
            )
        if output_key in EVIDENCE_OUTPUTS:
            _check_evidence_ids(
                value=value,
                output_name=output_name,
                warnings=warnings,
            )
        if output_key in LEAKAGE_OUTPUTS and _contains_confirmed_leakage_claim(value):
            warnings.append("Leakage Risk Analyst output must not say leakage is confirmed.")

    return warnings


def validate_report_text(report_text: str) -> list[str]:
    warnings: list[str] = []
    found_headings = _extract_required_heading_lines(_canonicalize_report_headings(report_text))
    if found_headings != SECTION_HEADINGS:
        warnings.append(
            "Report must contain exactly the 15 required section headings in order; "
            f"found {len(found_headings)} matching headings."
        )

    lowered = report_text.lower()
    for phrase in FORBIDDEN_REPORT_PHRASES:
        if phrase in lowered:
            warnings.append(f"Report contains forbidden data-execution claim: {phrase!r}.")

    return warnings


def validate_retrieved_documents(docs: list[RetrievedDocument]) -> list[str]:
    if not docs:
        return ["retrieved_documents must not be empty."]
    return []


def _check_mapping_field(
    value: Any,
    field: str,
    output_name: str,
    warnings: list[str],
) -> None:
    normalized_value = _to_plain(value)
    if not isinstance(normalized_value, dict):
        warnings.append(f"Reasoning output {output_name!r} must be an object with {field!r}.")
        return
    if field not in normalized_value:
        warnings.append(f"Reasoning output {output_name!r} is missing {field!r}.")


def _check_evidence_ids(value: Any, output_name: str, warnings: list[str]) -> None:
    normalized_value = _to_plain(value)
    if isinstance(normalized_value, list):
        for index, item in enumerate(normalized_value):
            if not isinstance(item, dict) or "evidence_ids" not in item:
                warnings.append(
                    f"Reasoning output {output_name!r}[{index}] is missing 'evidence_ids'."
                )
        return

    if not isinstance(normalized_value, dict):
        warnings.append(f"Reasoning output {output_name!r} must be an object with 'evidence_ids'.")
        return
    if "evidence_ids" not in normalized_value:
        warnings.append(f"Reasoning output {output_name!r} is missing 'evidence_ids'.")


def _contains_confirmed_leakage_claim(value: Any) -> bool:
    lowered_values = [text.lower() for text in _iter_text_values(_to_plain(value))]
    return any(
        phrase in text
        for text in lowered_values
        for phrase in LEAKAGE_CONFIRMED_PHRASES
    )


def _iter_text_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_text_values(item)


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
