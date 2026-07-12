from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses


TERMINAL_HYPOTHESIS_STATUSES = {"confirmed", "rejected"}
TEMPORAL_PRIMARY_METHODS = {"temporal_holdout", "temporal_cv", "expanding_window"}


def validate_evidence_pack(pack: EdaEvidencePack) -> list[str]:
    """Return non-fatal quality warnings for an EDA evidence pack."""

    warnings: list[str] = []
    warnings.extend(validate_evidence_refs(pack))
    if not isinstance(pack.warnings, list):
        warnings.append("evidence_pack.warnings is not a list; warning preservation is unsafe.")
    if not isinstance(pack.limitations, list):
        warnings.append("evidence_pack.limitations is not a list; limitation preservation is unsafe.")
    return warnings


def validate_evidence_refs(pack: EdaEvidencePack) -> list[str]:
    warnings: list[str] = []
    for index, result in enumerate(pack.hypothesis_results):
        label = f"hypothesis_results[{index}] {result.hypothesis_id}"
        if result.status in TERMINAL_HYPOTHESIS_STATUSES and not result.evidence_refs:
            warnings.append(f"{label} is {result.status} but has no evidence_refs.")
        for ref in result.evidence_refs:
            if not _evidence_ref_exists(pack, ref):
                warnings.append(f"{label} has broken evidence_ref: {ref}")

    for index, action in enumerate(pack.recommended_next_actions):
        label = f"recommended_next_actions[{index}]"
        if not str(action.priority or "").strip():
            warnings.append(f"{label} has no priority.")
        if not _meaningful_text(action.action):
            warnings.append(f"{label} has empty or malformed action text.")
        if not _meaningful_text(action.why):
            warnings.append(f"{label} has empty or malformed why.")
        if not action.evidence_refs:
            warnings.append(f"{label} has no evidence_refs.")
        for ref in action.evidence_refs:
            if not _evidence_ref_exists(pack, ref):
                warnings.append(f"{label} has broken evidence_ref: {ref}")
    return warnings


def validate_hypothesis_results(
    pack: EdaEvidencePack,
    hypotheses: ResearchHypotheses,
) -> list[str]:
    warnings: list[str] = []
    result_counts: dict[str, int] = {}
    for result in pack.hypothesis_results:
        result_counts[result.hypothesis_id] = result_counts.get(result.hypothesis_id, 0) + 1

    expected_ids = [hypothesis.hypothesis_id for hypothesis in hypotheses.hypotheses]
    expected_set = set(expected_ids)
    for hypothesis_id in expected_ids:
        count = result_counts.get(hypothesis_id, 0)
        if count == 0:
            warnings.append(f"Missing hypothesis result for {hypothesis_id}.")
        elif count > 1:
            warnings.append(f"Hypothesis {hypothesis_id} has {count} results; expected exactly one.")

    for hypothesis_id in sorted(set(result_counts) - expected_set):
        warnings.append(f"Unexpected hypothesis result for {hypothesis_id}.")

    warnings.extend(validate_evidence_refs(pack))
    return warnings


def validate_no_unsupported_summary_claims(
    summary_text: str,
    pack: EdaEvidencePack,
) -> list[str]:
    warnings: list[str] = []
    summary = summary_text.lower()
    if "probably confirmed" in summary and not _has_terminal_result_with_evidence(pack):
        warnings.append(
            "eda_summary.md says 'probably confirmed' without terminal hypothesis evidence."
        )
    if "leakage found" in summary and not _has_failed_leakage_with_evidence(pack):
        warnings.append(
            "eda_summary.md says 'leakage found' but leakage_evidence has no failed check with evidence."
        )
    if "baseline proves final solution" in summary:
        warnings.append(
            "eda_summary.md overclaims baseline evidence as a final solution."
        )
    if "temporal validation is required" in summary and not _primary_validation_is_temporal(pack):
        warnings.append(
            "eda_summary.md claims temporal validation is required, but validation_evidence selected another primary policy."
        )
    return warnings


def _evidence_ref_exists(pack: EdaEvidencePack, ref: str) -> bool:
    if not ref or not ref.strip():
        return False
    parts = [part for part in ref.split(".") if part]
    if not parts:
        return False
    current: Any = pack.model_dump(mode="json")
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return False
            current = current[part]
            continue
        if isinstance(current, list):
            if part.isdigit():
                index = int(part)
                if index >= len(current):
                    return False
                current = current[index]
                continue
            match = next(
                (
                    item
                    for item in current
                    if isinstance(item, dict)
                    and (
                        item.get("risk_id") == part
                        or item.get("id") == part
                          or item.get("ablation_id") == part
                          or item.get("feature_block") == part
                          or item.get("configuration") == part
                          or item.get("interaction_id") == part
                          or item.get("claim_id") == part
                    )
                ),
                None,
            )
            if match is None:
                return False
            current = match
            continue
        return False
    return True


def _has_terminal_result_with_evidence(pack: EdaEvidencePack) -> bool:
    return any(
        result.status in TERMINAL_HYPOTHESIS_STATUSES and bool(result.evidence_refs)
        for result in pack.hypothesis_results
    )


def _has_failed_leakage_with_evidence(pack: EdaEvidencePack) -> bool:
    for item in pack.leakage_evidence:
        status = str(item.get("status", "")).lower()
        evidence = item.get("evidence") or {}
        if status == "failed" and bool(evidence):
            return True
    return False


def _primary_validation_is_temporal(pack: EdaEvidencePack) -> bool:
    validation = pack.validation_evidence or {}
    primary = validation.get("primary_validation") or {}
    method = str(primary.get("method") or "").lower()
    return method in TEMPORAL_PRIMARY_METHODS


def _meaningful_text(value: str) -> bool:
    text = str(value or "").strip()
    normalized = "".join(char for char in text if char.isalnum()).strip()
    return len(normalized) >= 3


__all__ = [
    "validate_evidence_pack",
    "validate_evidence_refs",
    "validate_hypothesis_results",
    "validate_no_unsupported_summary_claims",
]
