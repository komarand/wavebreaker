from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.schemas import HypothesisResult, ResearchHypothesis


SUPPORTED_CATEGORIES = {
    "schema",
    "metric",
    "validation",
    "leakage",
    "relationship",
    "drift",
    "baseline",
    "feature",
    "notebook",
    "data_quality",
}

CATEGORY_MODULES = {
    "schema": "schema_inferer",
    "metric": "metric_analyzer",
    "validation": "validation_analyzer",
    "leakage": "leakage_checker",
    "relationship": "relationship_analyzer",
    "drift": "drift_analyzer",
    "baseline": "baseline_runner",
    "feature": "feature_probe",
    "notebook": "notebook_static_analysis",
    "data_quality": "table_profiler",
}


def evaluate_hypotheses(
    hypotheses: list[ResearchHypothesis],
    evidence_pack_partial: dict,
    module_statuses: dict[str, str] | None = None,
) -> list[HypothesisResult]:
    """Evaluate Research Scout hypotheses against generic EDA evidence."""

    statuses = {key: value.lower() for key, value in (module_statuses or {}).items()}
    results: list[HypothesisResult] = []
    for hypothesis in hypotheses:
        skipped = _skipped_result_if_needed(hypothesis, statuses)
        if skipped is not None:
            results.append(skipped)
            continue

        category = str(hypothesis.category)
        if category == "schema":
            results.append(_evaluate_schema(hypothesis, evidence_pack_partial))
        elif category == "metric":
            results.append(_evaluate_metric(hypothesis, evidence_pack_partial))
        elif category == "validation":
            results.append(_evaluate_validation(hypothesis, evidence_pack_partial))
        elif category == "leakage":
            results.append(_evaluate_leakage(hypothesis, evidence_pack_partial))
        elif category in SUPPORTED_CATEGORIES:
            results.append(_evaluate_presence_category(hypothesis, evidence_pack_partial))
        else:
            results.append(
                _make_result(
                    hypothesis,
                    status="not_testable",
                    confidence="low",
                    finding=f"Unsupported hypothesis category '{category}'.",
                    impact="No automated strategy change; review this hypothesis manually.",
                    limitations=[f"Unsupported hypothesis category: {category}."],
                )
            )
    return results


def _evaluate_schema(
    hypothesis: ResearchHypothesis,
    evidence_pack: dict,
) -> HypothesisResult:
    schema = _as_dict(evidence_pack.get("inferred_schema"))
    if not schema:
        return _not_testable(hypothesis, "Inferred schema evidence is unavailable.")

    refs: list[str] = []
    present = []
    missing = []
    for field_name in ("train_base_table", "test_base_table", "target_column", "primary_id_column"):
        if schema.get(field_name):
            refs.append(f"inferred_schema.{field_name}")
            present.append(field_name)
        else:
            missing.append(field_name)

    if not missing:
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="high",
            finding="Schema evidence identifies train/test tables, target, and primary ID.",
            impact="Use inferred schema roles as the base for validation and leakage checks.",
            evidence_refs=refs,
        )
    if present:
        return _make_result(
            hypothesis,
            status="partially_confirmed",
            confidence="medium",
            finding=f"Schema evidence is present, but missing: {', '.join(missing)}.",
            impact="Proceed with available schema roles and manually confirm missing roles.",
            evidence_refs=refs,
            limitations=[f"Missing schema fields: {', '.join(missing)}."],
        )
    return _not_testable(hypothesis, "No usable schema roles were inferred.")


def _evaluate_metric(
    hypothesis: ResearchHypothesis,
    evidence_pack: dict,
) -> HypothesisResult:
    metric = _as_dict(evidence_pack.get("metric_evidence"))
    if not metric:
        return _not_testable(hypothesis, "Metric evidence is unavailable.")

    refs = ["metric_evidence.metric_name"]
    claim = hypothesis.claim.lower()
    probability_or_rank_claim = any(
        token in claim for token in ("probab", "rank", "score")
    )
    supports_probability_or_rank = bool(
        metric.get("requires_probabilities") or metric.get("rank_based")
    )
    if probability_or_rank_claim and supports_probability_or_rank:
        refs.extend(_refs_for_truthy_metric_fields(metric))
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="high",
            finding="Metric evidence requires probabilities/ranks rather than hard labels.",
            impact="Plan model outputs as calibrated scores or ranking scores for validation.",
            evidence_refs=_unique(refs),
        )
    if probability_or_rank_claim and not supports_probability_or_rank:
        return _make_result(
            hypothesis,
            status="rejected",
            confidence="medium",
            finding="Metric evidence does not require probability or rank-style outputs.",
            impact="Use the output type required by the resolved metric instead.",
            evidence_refs=refs,
        )
    if metric.get("metric_name"):
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="medium",
            finding=f"Metric evidence resolved metric '{metric.get('metric_name')}'.",
            impact="Use the resolved metric contract when designing validation and predictions.",
            evidence_refs=refs,
        )
    return _not_testable(hypothesis, "Metric name could not be resolved.")


def _evaluate_validation(
    hypothesis: ResearchHypothesis,
    evidence_pack: dict,
) -> HypothesisResult:
    validation = _as_dict(evidence_pack.get("validation_evidence"))
    if not validation:
        return _not_testable(hypothesis, "Validation evidence is unavailable.")

    primary = _as_dict(validation.get("primary_validation"))
    diagnostics = [_as_dict(item) for item in validation.get("diagnostic_validations", [])]
    rejected = [_as_dict(item) for item in validation.get("rejected_validations", [])]
    claim = hypothesis.claim.lower()
    temporal_claim = any(
        token in claim for token in ("temporal", "out-of-time", "time", "week")
    )
    strong_temporal_claim = any(token in claim for token in ("required", "must", "primary"))

    primary_method = str(primary.get("method", ""))
    primary_is_temporal = primary_method in {"temporal_holdout", "expanding_window"}
    diagnostic_is_temporal = any(
        item.get("method") in {"temporal_holdout", "expanding_window"}
        for item in diagnostics
    )
    temporal_rejected_as_default = any(
        "temporal" in str(item.get("method", ""))
        for item in rejected
    )

    if temporal_claim:
        if primary_is_temporal:
            return _make_result(
                hypothesis,
                status="confirmed",
                confidence="high",
                finding="Validation evidence selected temporal validation as primary.",
                impact="Use temporal validation as the primary model-selection policy.",
                evidence_refs=["validation_evidence.primary_validation"],
            )
        if diagnostic_is_temporal and temporal_rejected_as_default:
            status = "rejected" if strong_temporal_claim else "partially_confirmed"
            finding = (
                "Time evidence exists only as a diagnostic; temporal validation "
                "was not selected as primary."
            )
            impact = (
                "Keep temporal diagnostics, but follow the selected primary validation policy."
            )
            return _make_result(
                hypothesis,
                status=status,
                confidence="medium",
                finding=finding,
                impact=impact,
                evidence_refs=[
                    "validation_evidence.diagnostic_validations",
                    "validation_evidence.rejected_validations",
                ],
                limitations=(
                    ["Time column alone is insufficient for primary temporal validation."]
                    if status == "partially_confirmed"
                    else []
                ),
            )
        return _make_result(
            hypothesis,
            status="rejected",
            confidence="medium",
            finding="Validation evidence does not support temporal validation.",
            impact="Use the selected non-temporal validation policy.",
            evidence_refs=["validation_evidence.primary_validation"],
        )

    if primary:
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="high",
            finding=f"Validation evidence selected {primary_method}.",
            impact="Use the selected validation policy for model comparison.",
            evidence_refs=["validation_evidence.primary_validation"],
        )
    return _not_testable(hypothesis, "Primary validation policy is unavailable.")


def _evaluate_leakage(
    hypothesis: ResearchHypothesis,
    evidence_pack: dict,
) -> HypothesisResult:
    leakage = [_as_dict(item) for item in evidence_pack.get("leakage_evidence", [])]
    if not leakage:
        return _not_testable(hypothesis, "Leakage evidence is unavailable.")

    failed = [item for item in leakage if item.get("status") == "failed"]
    warnings = [item for item in leakage if item.get("status") == "warning"]
    passed = [item for item in leakage if item.get("status") == "passed"]
    not_testable = [item for item in leakage if item.get("status") == "not_testable"]
    refs = ["leakage_evidence"]

    if failed:
        positive_leakage_claim = _is_positive_leakage_claim(hypothesis.claim)
        return _make_result(
            hypothesis,
            status="confirmed" if positive_leakage_claim else "rejected",
            confidence="high",
            finding=f"Leakage checks found failed checks: {_check_ids(failed)}.",
            impact="Fix or exclude unsafe columns/tables before modeling.",
            evidence_refs=refs,
        )
    if warnings:
        return _make_result(
            hypothesis,
            status="partially_confirmed",
            confidence="medium",
            finding=f"Leakage checks found warnings: {_check_ids(warnings)}.",
            impact="Investigate warning-level leakage risks before feature engineering.",
            evidence_refs=refs,
            limitations=["Warning-level leakage checks are suspicious, not proof."],
        )
    if passed:
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="high",
            finding="Leakage checks passed for the available direct checks.",
            impact="Proceed with modeling while keeping not-testable checks on the review list.",
            evidence_refs=refs,
            limitations=(
                [f"Not-testable checks remain: {_check_ids(not_testable)}."]
                if not_testable
                else []
            ),
        )
    return _not_testable(hypothesis, "Leakage checks were all not testable.")


def _evaluate_presence_category(
    hypothesis: ResearchHypothesis,
    evidence_pack: dict,
) -> HypothesisResult:
    category = str(hypothesis.category)
    evidence_key = f"{category}_evidence"
    evidence = evidence_pack.get(evidence_key)
    if evidence:
        return _make_result(
            hypothesis,
            status="confirmed",
            confidence="medium",
            finding=f"{evidence_key} is present.",
            impact=f"Use {category} evidence before finalizing the modeling plan.",
            evidence_refs=[evidence_key],
        )
    return _not_testable(hypothesis, f"{evidence_key} is unavailable.")


def _skipped_result_if_needed(
    hypothesis: ResearchHypothesis,
    module_statuses: dict[str, str],
) -> HypothesisResult | None:
    related_modules = _related_modules(hypothesis)
    skipped_modules = [
        module
        for module in related_modules
        if module_statuses.get(module) in {"skipped", "disabled", "not_run"}
    ]
    if not skipped_modules:
        return None
    return _make_result(
        hypothesis,
        status="skipped",
        confidence="low",
        finding=f"Related module(s) were skipped: {', '.join(skipped_modules)}.",
        impact="Do not use this hypothesis for strategy until the skipped module runs.",
        limitations=[f"Skipped modules: {', '.join(skipped_modules)}."],
    )


def _related_modules(hypothesis: ResearchHypothesis) -> list[str]:
    modules = []
    for check_name in hypothesis.expected_eda_checks:
        module = str(check_name).split(".", maxsplit=1)[0]
        if module:
            modules.append(module)
    category_module = CATEGORY_MODULES.get(str(hypothesis.category))
    if category_module is not None:
        modules.append(category_module)
    return _unique(modules)


def _not_testable(hypothesis: ResearchHypothesis, limitation: str) -> HypothesisResult:
    return _make_result(
        hypothesis,
        status="not_testable",
        confidence="low",
        finding=limitation,
        impact="No automated strategy change; collect the missing evidence first.",
        limitations=[limitation],
    )


def _make_result(
    hypothesis: ResearchHypothesis,
    *,
    status: str,
    confidence: str,
    finding: str,
    impact: str,
    evidence_refs: list[str] | None = None,
    limitations: list[str] | None = None,
) -> HypothesisResult:
    refs = evidence_refs or []
    result_limitations = limitations or []
    if status in {"confirmed", "rejected"} and not refs:
        refs = ["evidence_pack_partial"]
    if status in {"not_testable", "skipped"} and not result_limitations:
        result_limitations = ["Required evidence was unavailable."]
    return HypothesisResult(
        hypothesis_id=hypothesis.hypothesis_id,
        category=str(hypothesis.category),
        status=status,
        confidence_after_eda=confidence,
        finding=finding,
        evidence_refs=refs,
        impact_on_strategy=impact,
        limitations=result_limitations,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


def _refs_for_truthy_metric_fields(metric: dict[str, Any]) -> list[str]:
    refs = []
    for field_name in ("requires_probabilities", "rank_based", "prediction_output_type"):
        if metric.get(field_name):
            refs.append(f"metric_evidence.{field_name}")
    return refs


def _check_ids(checks: list[dict[str, Any]]) -> str:
    return ", ".join(str(check.get("check_id")) for check in checks)


def _is_positive_leakage_claim(claim: str) -> bool:
    normalized = claim.lower()
    negative_terms = (
        "no leakage",
        "no direct leakage",
        "absent",
        "should pass",
        "safe",
    )
    if any(term in normalized for term in negative_terms):
        return False
    return any(
        term in normalized
        for term in (
            "leakage",
            "target present",
            "target in test",
            "unsafe",
            "risk",
        )
    )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["evaluate_hypotheses"]
