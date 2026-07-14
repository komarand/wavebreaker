from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import (
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.schemas import RetrievedDocument, ReviewResult


async def review(
    draft_sections: dict[str, Any],
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
    artifact_dir: Path | str | None = None,
) -> ReviewResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    planned_experiment_ids = sorted(_collect_experiment_ids(draft_sections))
    result = await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "Act as a critical Kaggle Grandmaster reviewer. Do not add new facts or "
            "new unsupported claims. Only revise, remove, or qualify claims using "
            "draft_sections and retrieved_documents. Identify unsupported_claims, "
            "too_generic, and unnecessary_experiments. If a key claim has no "
            "evidence_ids, supporting_source_ids, or provenance, mark it unsupported. "
            "Return revised_sections with the same high-level keys as draft_sections. "
            "Return revised_sections as structured JSON. Each value may be a string, a dict "
            "preserving the original section structure, or a list of experiment dicts. Do not "
            "stringify structured sections. Preserve provenance fields when present. Preserve "
            "confidence fields when present. Preserve supporting_source_ids when present. "
            "Flag key claims that lack provenance."
            " When experiment_id values are present, use only those exact IDs in "
            "approved_experiment_ids and rejected_experiment_ids."
            " reviewed_experiment_ids, approved_experiment_ids, and "
            "rejected_experiment_ids must contain only exact planned experiment IDs. "
            "Never place hypothesis IDs in experiment decision fields."
        ),
        user_payload={
            "draft_sections": draft_sections,
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": ReviewResult.model_json_schema(),
            "planned_experiment_ids": planned_experiment_ids,
        },
        result_model=ReviewResult,
        artifact_dir=artifact_dir,
        raw_artifact_name="review_result_raw.json",
    )
    unknown_ids = validate_evidence_ids(result, docs)
    if unknown_ids:
        raise ValueError(f"ReviewResult contains unknown evidence_ids: {unknown_ids}")
    _validate_experiment_decisions(result, draft_sections)
    return _normalize_review_result(result, draft_sections)


def _validate_experiment_decisions(result: ReviewResult, draft_sections: dict[str, Any]) -> None:
    known_ids = _collect_experiment_ids(draft_sections)
    decisions = (
        set(result.reviewed_experiment_ids)
        | set(result.approved_experiment_ids)
        | set(result.rejected_experiment_ids)
    )
    unknown = sorted(decisions - known_ids)
    if unknown:
        raise ValueError(f"ReviewResult references unknown experiment_ids: {unknown}")
    overlap = sorted(set(result.approved_experiment_ids) & set(result.rejected_experiment_ids))
    if overlap:
        raise ValueError(f"ReviewResult both approves and rejects experiment_ids: {overlap}")
    revised_ids = _collect_experiment_ids(result.revised_sections)
    unknown_revised = sorted(revised_ids - known_ids)
    if unknown_revised:
        raise ValueError(
            f"ReviewResult revised_sections adds unknown experiment_ids: {unknown_revised}"
        )


def _collect_experiment_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        own = value.get("experiment_id")
        result = {own} if isinstance(own, str) and own else set()
        for child in value.values():
            result.update(_collect_experiment_ids(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_collect_experiment_ids(child))
        return result
    return set()


def _normalize_review_result(result: ReviewResult, draft_sections: dict[str, Any]) -> ReviewResult:
    payload = result.model_dump()
    revised_sections = result.revised_sections or {}
    payload["revised_sections"] = {
        key: revised_sections.get(key, value)
        for key, value in draft_sections.items()
    }
    unsupported_claims = list(result.unsupported_claims)
    for path, claim in _find_key_claims_without_evidence(draft_sections):
        note = f"{path}: key claim lacks evidence_ids/supporting_source_ids/provenance: {claim}"
        if note not in unsupported_claims:
            unsupported_claims.append(note)
    payload["unsupported_claims"] = unsupported_claims
    return ReviewResult.model_validate(payload)


def _find_key_claims_without_evidence(value: Any, path: str = "draft_sections") -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        has_evidence = any(
            value.get(key)
            for key in ("evidence_ids", "supporting_source_ids", "provenance")
        )
        claim_text = _extract_claim_text(value)
        if claim_text is not None and not has_evidence:
            findings.append((path, claim_text))
        for key, child in value.items():
            findings.extend(_find_key_claims_without_evidence(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_key_claims_without_evidence(child, f"{path}[{index}]"))
    return findings


def _extract_claim_text(value: dict[str, Any]) -> str | None:
    for key in ("claim", "key_claim", "recommendation", "recommended_cv", "experiment"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None
