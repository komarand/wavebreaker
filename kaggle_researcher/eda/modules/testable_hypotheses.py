from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


VALID_SCOPES = {"feature", "preprocessing", "interaction", "validation", "robustness", "postprocessing"}
RELIABILITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class HypothesisGenerationConfig:
    max_testable_hypotheses: int = 10
    max_per_scope: int = 3
    min_reliability: str = "medium"


def build_safety_constraints(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    schema = _dict(evidence_pack.get("inferred_schema"))
    leakage = [_dict(item) for item in evidence_pack.get("leakage_evidence", [])]
    rows: list[dict[str, Any]] = []
    if schema.get("target_column") or schema.get("primary_id_column"):
        refs = [name for name, value in (("inferred_schema.target_column", schema.get("target_column")), ("inferred_schema.primary_id_column", schema.get("primary_id_column"))) if value]
        rows.append(_constraint("features", "Exclude the target and primary ID from ordinary model features.", refs, "These columns have reserved roles and may produce leakage or artifact learning."))
    if schema.get("sample_submission_table"):
        rows.append(_constraint("submission", "Do not use sample-submission predictions as model features or tuning targets.", ["inferred_schema.sample_submission_table"], "Submission artifacts are outputs, not training evidence."))
    rows.append(_constraint("leakage", "Do not use test labels or fit target-aware transforms outside training folds.", ["leakage_evidence", "validation_evidence"], "Target access and globally fitted target-aware transforms invalidate out-of-fold evidence."))
    if any(item.get("status") in {"failed", "warning"} for item in leakage):
        rows.append(_constraint("leakage", "Resolve reported leakage warnings before interpreting diagnostic model gains.", ["leakage_evidence"], "Unresolved leakage can inflate validation measurements."))
    return _stable_ids(_dedupe(rows, "rule"), "constraint_id", "safety", 8)


def build_validation_requirements(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    validation = _dict(evidence_pack.get("validation_evidence"))
    metric = _dict(evidence_pack.get("metric_evidence"))
    target = _dict(evidence_pack.get("target_diagnostics"))
    primary = _dict(validation.get("recommended_validation_candidate") or validation.get("primary_validation"))
    method = str(primary.get("method") or "").lower()
    rows: list[dict[str, Any]] = []
    if "stratified" in method:
        rows.append(_requirement("Preserve target distribution across classification folds.", "required", None, ["target_diagnostics.distribution", "validation_evidence"], "The evidence-supported validation candidate requires stratification."))
    if "group" in method:
        rows.append(_requirement("Keep members of the same group in a single validation partition.", "required", None, ["validation_evidence.primary_validation"], "Observed group structure makes row-wise splitting unsafe."))
    if any(token in method for token in ("temporal", "time", "expanding")):
        rows.append(_requirement("Respect chronology when constructing validation folds.", "required", None, ["validation_evidence.primary_validation"], "Temporal evidence makes future-to-past leakage unsafe."))
    if metric.get("requires_threshold") or target.get("threshold_diagnostics"):
        rows.append(_requirement("Select prediction thresholds only inside validation.", "required", None, ["metric_evidence.requires_threshold"], "Threshold selection on held-out or test outcomes would leak evaluation information."))
    if _dict(evidence_pack.get("baseline_ablation_evidence")):
        rows.append(_requirement("Use identical folds and metric calculations for paired ablation comparisons.", "required", None, ["baseline_ablation_evidence.fold_policy", "metric_evidence"], "Paired controls are required to attribute a delta to the tested change."))
    return _stable_ids(_dedupe(rows, "rule"), "requirement_id", "validation_requirement", 6)


def build_testable_hypotheses(*, evidence_pack: dict[str, Any], config: HypothesisGenerationConfig | None = None) -> list[dict[str, Any]]:
    """Build only unresolved hypotheses from an explicit diagnostic allowlist."""
    config = config or HypothesisGenerationConfig()
    builders: tuple[Callable[[dict[str, Any]], list[dict[str, Any]]], ...] = (
        build_ablation_hypotheses, build_interaction_hypotheses,
        build_slice_hypotheses, build_drift_hypotheses,
        build_feature_hypotheses, build_metric_hypotheses,
    )
    candidates = [item for builder in builders for item in builder(evidence_pack)]
    candidates = _semantic_dedupe(candidates)
    minimum = RELIABILITY_ORDER.get(config.min_reliability, 1)
    strong = [item for item in candidates if RELIABILITY_ORDER[item["reliability"]] <= minimum]
    weak = [item for item in candidates if item not in strong]
    ranked = sorted(strong, key=_rank_key)
    if len(ranked) < config.max_testable_hypotheses:
        ranked.extend(sorted(weak, key=_rank_key))
    selected, counts = [], {}
    for item in ranked:
        scope = item["scope"]
        if counts.get(scope, 0) >= config.max_per_scope: continue
        selected.append(item); counts[scope] = counts.get(scope, 0) + 1
        if len(selected) >= min(config.max_testable_hypotheses, 12): break
    return [{**item, "hypothesis_id": f"eda_hypothesis_{index:03d}"} for index, item in enumerate(selected, 1)]


def build_ablation_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    findings = [_dict(item) for item in _dict(evidence.get("baseline_ablation_evidence")).get("feature_block_findings", [])]
    rows = []
    for item in findings:
        block, status = str(item.get("feature_block") or item.get("configuration") or ""), str(item.get("status") or "")
        if block == "high_cardinality_categorical" and status in {"neutral", "hurt", "unstable"}:
            rows.append(_hyp("preprocessing", "Fold-fitted robust encoding may improve the high-cardinality categorical block relative to the simple diagnostic treatment.", "The high-cardinality block did not show stable benefit under the diagnostic baseline.", "The baseline tested only its bounded default encoding, not alternative fold-fitted rare-group or frequency treatments.", _refs(item, "baseline_ablation_evidence.feature_block_findings.high_cardinality_categorical"), "baseline_ablation_evidence", ["same folds", "same metric", "encoder fitted inside folds", "same feature block except for encoding"], ["paired fold delta versus baseline", "fold stability", "unseen-category robustness"], "important", _reliability(item)))
        elif block == "missingness_indicators" and status in {"neutral", "hurt", "unstable"}:
            rows.append(_hyp("feature", "Missingness indicators may have model-dependent value despite being neutral or unstable in the diagnostic baseline.", "The missingness block did not produce a stable material diagnostic gain.", "A lightweight sanity-floor model cannot settle whether another downstream model representation benefits from the indicators.", _refs(item, "baseline_ablation_evidence.feature_block_findings.missingness_indicators"), "baseline_ablation_evidence", ["same folds", "same metric", "same model within the paired comparison"], ["paired fold delta", "fold consistency", "slice impact"], "optional", _reliability(item)))
        elif item.get("finding_type") == "configuration" and (status in {"unstable", "not_better"} or item.get("materiality_vs_best_prior") == "negligible"):
            rows.append(_hyp("feature", "A simpler feature configuration may generalize as well as the full safe-feature configuration.", "The composite configuration was unstable or added negligible diagnostic value over a simpler configuration.", "The bounded diagnostic baseline cannot establish whether the additional complexity helps downstream modeling.", _refs(item, "baseline_ablation_evidence.best_ablation"), "baseline_ablation_evidence", ["same folds", "same metric", "paired simpler and composite configurations"], ["paired fold delta", "complexity-adjusted materiality", "fold stability"], "optional", _reliability(item)))
    return rows


def build_interaction_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in [_dict(value) for value in _dict(evidence.get("interaction_diagnostics")).get("interaction_hypotheses", [])]:
        if item.get("reliability") != "reliable" or item.get("materiality") not in {"material", "small"}: continue
        columns = sorted(str(value) for value in item.get("columns", []) if value)
        if not columns: continue
        rows.append(_hyp("interaction", f"The bounded interaction between {', '.join(columns)} may improve OOF performance beyond its parent features.", str(item.get("hypothesis") or "Interaction screening found a supported pairwise signal."), "The interaction diagnostic is a statistical screen and has not added the transformation to a fold-safe model comparison.", _refs(item, "interaction_diagnostics.interaction_hypotheses"), "baseline_evidence", ["same folds", "same metric", "parent features retained", "only the tested interaction added"], ["paired fold delta", "materiality classification", "fold stability"], "important" if item.get("materiality") == "material" else "optional", "high" if item.get("confidence") == "high" else "medium"))
    return sorted(rows, key=_rank_key)[:3]


def build_slice_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in [_dict(value) for value in _dict(evidence.get("slice_diagnostics")).get("slices", [])]:
        if item.get("reliability") not in {"reliable", "high"} or int(item.get("row_count") or 0) < 20: continue
        under = item.get("material_underperformance") is True or item.get("status") in {"material_underperformance", "underperforming"}
        if not under: continue
        label = str(item.get("slice_id") or "the reliable slice")
        rows.append(_hyp("robustness", f"A targeted feature or preprocessing change may reduce the fold-consistent performance gap for {label}.", "A sufficiently large reliable slice underperformed materially and consistently.", "Slice evidence localizes a gap but does not identify a causal remedy.", _refs(item, "slice_diagnostics.slices"), "baseline_evidence", ["same folds", "report global and slice metrics", "no slice-specific target leakage"], ["slice metric delta", "global metric delta", "fold consistency"], "optional", "medium"))
    return sorted(rows, key=_rank_key)[:2]


def build_drift_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    drift = _dict(evidence.get("drift_evidence")); adv = _dict(drift.get("adversarial_validation"))
    features = _dict(evidence.get("feature_diagnostics")); categorical = _dict(features.get("categorical_feature_diagnostics"))
    high_card = categorical.get("high_cardinality_candidates") or []
    auc = adv.get("auc") or adv.get("roc_auc") or adv.get("metric_value")
    if adv.get("severity") not in {"high", "critical"} and not (isinstance(auc, (int, float)) and auc >= 0.7): return []
    if not high_card: return []
    return [_hyp("robustness", "Measured train/test separation may be dominated by near-unique text or code-like columns rather than core model-ready features.", "Adversarial validation shows material separation while high-cardinality artifact-prone columns are present.", "The aggregate adversarial score does not attribute separation to a bounded feature subset.", ["drift_evidence.adversarial_validation", "feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"], "drift_evidence.adversarial_validation", ["same train/test samples", "same adversarial metric", "compare with artifact-prone columns excluded"], ["adversarial AUC delta", "remaining core-feature drift severity", "feature attribution coverage"], "important", "medium")]


def build_feature_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    features = _dict(evidence.get("feature_diagnostics")); text = _dict(features.get("text_feature_diagnostics"))
    findings = [_dict(item) for item in _dict(evidence.get("baseline_ablation_evidence")).get("feature_block_findings", [])]
    text_finding = next((item for item in findings if item.get("feature_block") == "text_code_simple"), {})
    if not (text.get("columns") or text.get("structural_feature_candidates")): return []
    if text_finding.get("status") == "helped" and text_finding.get("stability") == "stable": return []
    return [_hyp("feature", "Simple structural features derived from text or code-like columns may generalize better than raw high-cardinality tokens.", "Text or code-like columns were identified without stable material evidence for their current bounded representation.", "Raw token identity and compact structural summaries have not been isolated in a paired fold-safe comparison.", ["feature_diagnostics.text_feature_diagnostics", "baseline_ablation_evidence.feature_block_findings.text_code_simple"], "baseline_ablation_evidence", ["same folds", "same metric", "exclude raw token identity in the structural-only arm"], ["paired fold delta", "fold stability", "train/test coverage"], "optional", "medium")]


def build_metric_hypotheses(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    metric = _dict(evidence.get("metric_evidence"))
    if not metric.get("requires_threshold") or metric.get("threshold_search_needed") is not True: return []
    return [_hyp("postprocessing", "A validation-fitted decision threshold may improve the threshold-dependent metric relative to the fixed default threshold.", "The metric contract is threshold-dependent and no final threshold has been established.", "EDA establishes the fitting boundary but has not compared threshold policies on held-out predictions.", ["metric_evidence.requires_threshold", "metric_evidence.threshold_search_needed"], "baseline_evidence", ["threshold selected inside validation", "same OOF predictions", "report untuned default"], ["paired metric delta", "threshold stability across folds", "calibration sensitivity"], "important", "medium")]


def validate_testable_hypotheses_output(result: dict[str, Any], *, max_count: int = 12) -> list[str]:
    items = result.get("testable_hypotheses") or []; errors = []
    ids = [item.get("hypothesis_id") for item in items]
    if len(items) > max_count: errors.append("testable_hypotheses exceeds configured cap")
    if len(ids) != len(set(ids)): errors.append("hypothesis IDs must be unique")
    intents = set()
    forbidden = re.compile(r"(?i)exclude (the )?(target|primary id)|build (a )?baseline|run drift|sample.submission|test labels")
    for index, item in enumerate(items):
        label = f"testable_hypotheses[{index}]"
        if item.get("scope") not in VALID_SCOPES: errors.append(f"{label} has invalid scope")
        statement = str(item.get("statement") or "").strip()
        if len(statement) < 20 or forbidden.search(statement): errors.append(f"{label} is not a valid unresolved proposition")
        if not item.get("evidence_refs"): errors.append(f"{label} has no evidence_refs")
        if any(str(ref).startswith("source_claim_validation") for ref in item.get("evidence_refs", [])): errors.append(f"{label} uses source-claim-only evidence")
        if item.get("status") != "untested": errors.append(f"{label} status must be untested")
        if not item.get("required_controls"): errors.append(f"{label} has no meaningful controls")
        intent = _intent(item)
        if intent in intents: errors.append(f"{label} duplicates semantic intent")
        intents.add(intent)
    compatibility = result.get("experiment_candidates")
    if compatibility is not None and compatibility != items: errors.append("experiment_candidates must match canonical testable_hypotheses")
    if len({item.get("scope") for item in items}) > 1 and len({tuple(item.get("expected_evidence") or []) for item in items}) == 1: errors.append("different hypothesis scopes use identical expected_evidence")
    return errors


def _hyp(scope: str, statement: str, trigger: str, unresolved: str, refs: list[str], baseline: str | None, controls: list[str], expected: list[str], priority: str, reliability: str) -> dict[str, Any]:
    return {"scope": scope, "statement": statement, "trigger_finding": trigger, "why_unresolved": unresolved, "evidence_refs": sorted(set(refs)), "baseline_ref": baseline, "required_controls": controls, "expected_evidence": expected, "priority_signal": priority, "reliability": reliability, "status": "untested", "evidence_origin": "reasoning_inference"}


def _constraint(scope: str, rule: str, refs: list[str], reason: str) -> dict[str, Any]: return {"scope": scope, "rule": rule, "severity": "mandatory", "evidence_refs": refs, "reason": reason, "evidence_origin": "dataset_measurement"}
def _requirement(rule: str, status: str, condition: str | None, refs: list[str], reason: str) -> dict[str, Any]: return {"rule": rule, "status": status, "condition": condition, "evidence_refs": refs, "reason": reason, "evidence_origin": "statistical_diagnostic"}
def _dict(value: Any) -> dict[str, Any]: return value.model_dump(mode="json") if hasattr(value, "model_dump") else value if isinstance(value, dict) else {}
def _refs(item: dict[str, Any], fallback: str) -> list[str]: return sorted(set(str(value) for value in item.get("evidence_refs", []) if value) | {fallback})
def _reliability(item: dict[str, Any]) -> str: return "high" if item.get("confidence") == "high" and item.get("stability") == "stable" else "medium" if item.get("confidence") in {"high", "medium"} else "low"
def _intent(item: dict[str, Any]) -> str: return re.sub(r"[^a-z0-9]+", " ", f"{item.get('scope')} {item.get('statement')} {item.get('baseline_ref') or ''}".lower()).strip()
def _rank_key(item: dict[str, Any]) -> tuple[Any, ...]: return (0 if item["priority_signal"] == "important" else 1, RELIABILITY_ORDER[item["reliability"]], item["scope"], _intent(item))
def _semantic_dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = {}
    for item in sorted(items, key=_rank_key):
        key = _intent(item)
        if key not in output: output[key] = item
        else: output[key]["evidence_refs"] = sorted(set(output[key]["evidence_refs"] + item["evidence_refs"]))
    return list(output.values())
def _dedupe(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return list({re.sub(r"\W+", " ", str(item[field]).lower()).strip(): item for item in items}.values())
def _stable_ids(items: list[dict[str, Any]], field: str, prefix: str, cap: int) -> list[dict[str, Any]]:
    return [{field: f"{prefix}_{index:03d}", **item} for index, item in enumerate(sorted(items, key=lambda value: str(value.get("rule"))), 1)][:cap]


__all__ = ["HypothesisGenerationConfig", "build_safety_constraints", "build_validation_requirements", "build_testable_hypotheses", "validate_testable_hypotheses_output"]
