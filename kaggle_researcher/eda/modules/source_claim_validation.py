from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


VALIDATION_STATUSES = {"confirmed", "partially_supported", "contradicted", "not_testable", "unsupported", "analogous_only", "too_vague", "unsafe", "superseded"}
SAFE_USE = {"confirmed": "adopt", "partially_supported": "test_as_hypothesis", "contradicted": "reject", "unsafe": "reject", "analogous_only": "test_as_hypothesis", "not_testable": "test_as_hypothesis", "unsupported": "ignore", "too_vague": "ignore", "superseded": "ignore"}


def validate_source_claims(raw_claims: list[Any], evidence_pack: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate collected source claims against current EDA evidence.

    This is a conservative rule registry. It never fetches sources and never turns
    source authority into dataset confirmation without a current EDA reference.
    """
    claims = _dedupe_claims([_normalize_claim(item) for item in raw_claims if _claim_text(item)])
    if not claims:
        return _skipped()
    validated = [_validate(claim, evidence_pack) for claim in claims]
    conflicts = _conflicts(validated, evidence_pack)
    summaries = _source_summary(claims, validated)
    implications, experiments = _implications(validated)
    result = {
        "status": "completed",
        "claim_inventory": claims,
        "validated_claims": validated,
        "unsupported_claims": _by_status(validated, "unsupported"),
        "contradicted_claims": _by_status(validated, "contradicted"),
        "not_testable_claims": _by_status(validated, "not_testable"),
        "analogous_claims": _by_status(validated, "analogous_only"),
        "claim_conflicts": conflicts,
        "source_reliability_summary": summaries,
        "strategy_implications": implications,
        "recommended_experiments": experiments,
        "final_strategy_claims": _final_buckets(validated),
        "warnings": [],
        "limitations": ["Claim validation is deterministic evidence mapping; unstructured source prose may remain not testable."],
    }
    errors = validate_source_claim_validation(result)
    if errors:
        result["warnings"].extend(errors)
    return result


def collect_source_claims(research_hypotheses: Any, notebook_static_analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Adapt Scout hypotheses, structured findings, and notebook patterns to claims."""
    payload = _as_dict(research_hypotheses)
    claims: list[dict[str, Any]] = []
    for item in payload.get("hypotheses", []):
        source_refs = list(item.get("source_refs") or [])
        claims.append({"source_id": source_refs[0] if source_refs else "scout", "source_type": "scout_hypothesis", "source_scope": "direct_competition", "source_reliability": "medium", "claim_type": item.get("category", "unsupported"), "claim_text": item.get("claim", ""), "original_evidence_refs": source_refs, "confidence": item.get("confidence_before_eda", "low")})
    for item in payload.get("structured_findings", []):
        text = item.get("finding") or item.get("claim") or item.get("summary") or ""
        if text:
            refs = list(item.get("source_refs") or item.get("supporting_source_ids") or [])
            claims.append({"source_id": refs[0] if refs else "scout_finding", "source_type": "unknown", "source_scope": "direct_competition", "source_reliability": "medium", "claim_type": item.get("category") or item.get("finding_type") or "unsupported", "claim_text": text, "original_evidence_refs": refs, "confidence": item.get("confidence", "low")})
    notebook = _as_dict(notebook_static_analysis)
    if notebook.get("status") == "completed":
        for section, claim_type in (("cv_strategy", "validation"), ("feature_families", "feature"), ("model_families", "model"), ("postprocessing", "postprocessing")):
            for item in notebook.get(section, []):
                claims.append({"source_id": str((item.get("documents") or [{}])[0].get("source_id") or "notebook"), "source_type": "notebook", "source_scope": "direct_competition", "source_reliability": "low", "claim_type": claim_type, "claim_text": item.get("description") or item.get("pattern") or "", "original_evidence_refs": ["notebook_static_analysis"], "confidence": "low"})
    return claims


def map_claim_to_eda_evidence(claim: dict[str, Any], evidence_pack: dict[str, Any]) -> dict[str, list[str]]:
    claim_type = claim["claim_type"]
    mapping = {
        "schema": ["inferred_schema"], "metric": ["metric_evidence"], "validation": ["validation_evidence.primary_validation"],
        "leakage": ["leakage_evidence", "baseline_evidence.preprocessing_policy"], "target": ["target_diagnostics"],
        "drift": ["drift_evidence"], "missingness": ["feature_diagnostics.missingness_diagnostics", "target_diagnostics.target_by_missingness", "baseline_ablation_evidence"],
        "feature": ["feature_diagnostics", "baseline_ablation_evidence"], "interaction": ["interaction_diagnostics"],
        "relationship": ["relationship_evidence"], "model": ["baseline_evidence", "baseline_ablation_evidence"],
        "postprocessing": ["metric_evidence"], "leaderboard": ["drift_evidence", "validation_evidence"],
    }
    refs = [ref for ref in mapping.get(claim_type, []) if _path_exists(evidence_pack, ref)]
    return {"supporting": refs, "contradicting": refs}


def validate_source_claim_validation(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []; seen: set[str] = set()
    for item in result.get("validated_claims", []):
        claim_id, status = item.get("claim_id"), item.get("validation_status")
        if not claim_id or claim_id in seen: errors.append("Claims must have unique deterministic claim_id values.")
        seen.add(str(claim_id))
        if status not in VALIDATION_STATUSES: errors.append(f"Invalid claim validation status: {status}")
        if status == "confirmed" and not item.get("supporting_eda_refs"): errors.append(f"Confirmed claim {claim_id} lacks EDA evidence.")
        if status == "contradicted" and not item.get("contradicting_eda_refs"): errors.append(f"Contradicted claim {claim_id} lacks contradictory EDA evidence.")
        if status == "unsafe" and not item.get("recommended_action"): errors.append(f"Unsafe claim {claim_id} lacks mitigation.")
        if not str(item.get("claim_text") or "").strip(): errors.append("Claim text must be non-empty.")
    for bucket, items in _as_dict(result.get("final_strategy_claims")).items():
        if any(item.get("safe_strategy_use") != bucket for item in items): errors.append("Final strategy buckets do not match safe_strategy_use.")
    return sorted(set(errors))


def _normalize_claim(raw: Any) -> dict[str, Any]:
    item = _as_dict(raw); text = str(item.get("claim_text") or item.get("claim") or item.get("finding") or "").strip()
    claim_type = _claim_type(str(item.get("claim_type") or item.get("category") or ""), text)
    columns, unresolved = _entity_resolution(text, _as_dict(item.get("schema")))
    normalized = _normalize_text(text)
    return {"claim_id": "", "source_id": str(item.get("source_id") or "unknown"), "source_type": str(item.get("source_type") or "unknown"), "source_scope": _scope(item), "source_reliability": str(item.get("source_reliability") or "unknown"), "source_date": item.get("source_date"), "claim_type": claim_type, "claim_text": text, "normalized_claim": normalized, "entities": {"columns": columns, "tables": list(item.get("tables") or []), "feature_blocks": list(item.get("feature_blocks") or []), "model_families": list(item.get("model_families") or []), "metrics": _metric_entities(text), "validation_methods": _validation_entities(text)}, "entity_resolution": {"resolved_columns": columns, "unresolved_columns": unresolved, "resolved_tables": [], "unresolved_tables": [], "confidence": "high" if columns and not unresolved else "low" if unresolved else "medium"}, "expected_evidence": [], "original_evidence_refs": list(item.get("original_evidence_refs") or item.get("source_refs") or []), "confidence": str(item.get("confidence") or "low")}


def _validate(claim: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    _resolve_claim_entities(claim, evidence)
    text = claim["normalized_claim"]; ctype = claim["claim_type"]; mapped = map_claim_to_eda_evidence(claim, evidence)
    support, contradict = [], []
    status, finding = "not_testable", "Required current-dataset evidence is unavailable."
    unsafe = _unsafe_reason(text)
    if unsafe:
        status, finding, contradict = "unsafe", unsafe, ["leakage_evidence", "inferred_schema"]
    elif claim["source_scope"] in {"analogous_task", "generic_methodology"} and ctype == "model":
        status, finding = "analogous_only", "Model advice is useful for hypothesis generation but has no local model-family comparison."
    elif ctype == "metric":
        actual = str(_as_dict(evidence.get("metric_evidence")).get("metric_name") or "").lower()
        mentioned = _metric_entities(text)
        if mentioned and actual and all(value not in actual for value in mentioned): status, finding, contradict = "contradicted", f"Claimed metric conflicts with current metric `{actual}`.", ["metric_evidence.metric_name"]
        elif actual: status, finding, support = "confirmed", f"Metric contract confirms `{actual}`.", ["metric_evidence"]
    elif ctype == "schema":
        schema = _as_dict(evidence.get("inferred_schema")); target = str(schema.get("target_column") or "")
        target_match = re.search(r"target(?:\s+(?:is|column))?\s+[`'\"]?([a-zA-Z_][\w]*)", claim["claim_text"], re.I)
        if target_match and target:
            if _normalize_name(target_match.group(1)) == _normalize_name(target): status, finding, support = "confirmed", "Declared target matches inferred schema.", ["inferred_schema.target_column"]
            else: status, finding, contradict = "contradicted", "Claimed target conflicts with inferred schema.", ["inferred_schema.target_column"]
        elif schema: status, finding, support = "partially_supported", "Schema evidence is available but the claim is not specific enough for exact validation.", ["inferred_schema"]
    elif ctype == "validation":
        method = str(_as_dict(_as_dict(evidence.get("validation_evidence")).get("primary_validation")).get("method") or "").lower()
        requested = _validation_entities(text)
        if requested and method:
            if any(value in method for value in requested): status, finding, support = "confirmed", f"Current validation policy uses `{method}`.", ["validation_evidence.primary_validation"]
            elif "temporal" in requested and method in {"kfold", "stratified_kfold"}: status, finding, contradict = "contradicted", "Current EDA selected IID validation; a date column alone does not justify temporal CV.", ["validation_evidence.primary_validation"]
            else: status, finding, support = "partially_supported", "Claim is a validation alternative, not the selected primary policy.", ["validation_evidence.primary_validation"]
    elif ctype == "missingness":
        status, finding, support, contradict = _missingness_validation(text, evidence)
    elif ctype == "interaction":
        interactions = _as_dict(evidence.get("interaction_diagnostics")); hypotheses = interactions.get("interaction_hypotheses") or []
        if hypotheses: status, finding, support = "partially_supported", "Current interaction diagnostics found experiment-worthy interaction hypotheses.", ["interaction_diagnostics.interaction_hypotheses"]
    elif ctype in {"feature", "model", "preprocessing", "postprocessing", "relationship", "drift", "target", "leakage"}:
        if mapped["supporting"]:
            status, finding, support = "partially_supported", "Relevant EDA evidence exists, but the source claim needs controlled confirmation.", mapped["supporting"]
    if claim["entity_resolution"]["unresolved_columns"] and ctype == "schema": status, finding, support = "not_testable", "Claim depends on an unresolved column reference.", []
    origin = "eda_confirmed" if status == "confirmed" else "analogous_supported" if claim["source_scope"] == "analogous_task" else "generic_methodology" if claim["source_scope"] == "generic_methodology" else "direct_source_supported" if claim["source_scope"] == "direct_competition" else "unsupported"
    safe = SAFE_USE[status]
    return {"claim_id": claim["claim_id"], "validation_status": status, "confidence": "high" if status in {"confirmed", "contradicted", "unsafe"} else "medium" if status in {"partially_supported", "analogous_only"} else "low", "source_scope": claim["source_scope"], "claim_type": ctype, "claim_text": claim["claim_text"], "normalized_claim": claim["normalized_claim"], "finding": finding, "supporting_eda_refs": support, "contradicting_eda_refs": contradict, "source_evidence_refs": claim["original_evidence_refs"], "limitations": [] if status in {"confirmed", "unsafe"} else ["Source claim is not automatically accepted as a current-dataset fact."], "safe_strategy_use": safe, "recommended_action": _action(status, ctype), "recommended_experiment_id": f"exp_source_{claim['claim_id']}" if safe == "test_as_hypothesis" else None, "evidence_origin": origin}


def _missingness_validation(text: str, evidence: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    findings = _as_dict(evidence.get("baseline_ablation_evidence")).get("feature_block_findings") or []
    missing = next((_as_dict(item) for item in findings if _as_dict(item).get("feature_block") == "missingness_indicators"), {})
    target_by_missingness = _as_dict(evidence.get("target_diagnostics")).get("target_by_missingness") or []
    if "indicator" in text and "improv" in text:
        if missing.get("status") == "helped" and missing.get("materiality") == "material": return "confirmed", "Missingness-indicator ablation showed material benefit.", ["baseline_ablation_evidence.feature_block_findings"], []
        if missing.get("status") in {"neutral", "hurt"}: return "contradicted", "Missingness may be associated with target, but the controlled indicator ablation was not beneficial.", [], ["baseline_ablation_evidence.feature_block_findings"]
    if target_by_missingness: return "partially_supported", "Target diagnostics show missingness association, not necessarily model benefit.", ["target_diagnostics.target_by_missingness"], []
    return "not_testable", "Missingness evidence is unavailable.", [], []


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, tuple[str, ...], str], dict[str, Any]] = {}
    for claim in claims:
        key = (claim["claim_type"], claim["normalized_claim"], tuple(claim["entities"]["columns"]), claim["source_scope"])
        current = merged.get(key)
        if current is None: merged[key] = claim
        else: current["original_evidence_refs"] = sorted(set(current["original_evidence_refs"]) | set(claim["original_evidence_refs"])); current["source_id"] = ",".join(sorted(set(current["source_id"].split(",")) | {claim["source_id"]}))
    result = sorted(merged.values(), key=lambda item: (item["claim_type"], item["normalized_claim"], item["source_scope"], item["source_id"]))
    for index, item in enumerate(result, 1): item["claim_id"] = f"claim_{index:03d}"
    return result


def _conflicts(validated: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    validation = [item for item in validated if item["claim_type"] == "validation"]
    temporal = [item for item in validation if "temporal" in item["normalized_claim"]]; iid = [item for item in validation if any(token in item["normalized_claim"] for token in ("stratified", "kfold", "random"))]
    if not temporal or not iid: return []
    supported = next((item for item in iid if item["validation_status"] == "confirmed"), None) or next((item for item in temporal if item["validation_status"] == "confirmed"), None)
    return [{"conflict_id": "conflict_001", "claim_ids": sorted([temporal[0]["claim_id"], iid[0]["claim_id"]]), "claim_type": "validation", "conflict_intent": "primary_validation_policy", "positions": [temporal[0]["claim_text"], iid[0]["claim_text"]], "eda_resolution": f"supports_{supported['claim_id']}" if supported else "unresolved", "eda_evidence_refs": ["validation_evidence.primary_validation"], "recommended_action": "Use the current EDA validation policy for model comparison."}]


def _source_summary(claims: list[dict[str, Any]], validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in validated: by_source[result["claim_id"] and next(claim["source_id"] for claim in claims if claim["claim_id"] == result["claim_id"])].append(result)
    output = []
    for source, items in sorted(by_source.items()):
        counts = Counter(item["validation_status"] for item in items); scope = next(claim["source_scope"] for claim in claims if claim["source_id"] == source)
        output.append({"source_id": source, "source_type": next(claim["source_type"] for claim in claims if claim["source_id"] == source), "source_scope": scope, "claim_count": len(items), "confirmed_count": counts["confirmed"], "partially_supported_count": counts["partially_supported"], "contradicted_count": counts["contradicted"], "unsafe_count": counts["unsafe"], "not_testable_count": counts["not_testable"], "overall_usefulness": "high" if counts["confirmed"] else "medium" if counts["partially_supported"] else "low", "recommended_use": "strategy_support" if counts["confirmed"] else "hypothesis_generation" if counts["partially_supported"] or scope in {"analogous_task", "generic_methodology"} else "caution_only"})
    return output


def _implications(validated: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    implications, experiments = [], []
    for item in validated:
        use = item["safe_strategy_use"]
        if use == "ignore": continue
        implications.append({"implication_id": f"source_implication_{item['claim_id']}", "priority": "P0" if use == "reject" else "P1" if use == "adopt" else "P2", "action": item["recommended_action"], "origin": "validated_source_claim", "safe_strategy_use": use, "claim_ids": [item["claim_id"]], "eda_evidence_refs": [*item["supporting_eda_refs"], *item["contradicting_eda_refs"]], "source_evidence_refs": item["source_evidence_refs"], "why": item["finding"], "limitations": item["limitations"]})
        if use == "test_as_hypothesis": experiments.append({"experiment_id": item["recommended_experiment_id"], "priority": "P2", "claim_ids": [item["claim_id"]], "hypothesis": item["claim_text"], "experiment_type": _experiment_type(item["claim_type"]), "base_configuration_ref": "baseline_evidence", "changes": [], "validation_ref": "validation_evidence.primary_validation", "metric_ref": "metric_evidence", "success_criteria": {"minimum_materiality": "small", "require_fold_stability": True, "must_not_trigger_leakage_warning": True}, "risks": ["source_claim_unconfirmed"], "source_evidence_refs": item["source_evidence_refs"], "eda_evidence_refs": item["supporting_eda_refs"]})
    return implications, experiments


def _final_buckets(validated: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {key: [] for key in ("adopt", "test_as_hypothesis", "use_with_caution", "reject", "ignore")}
    for item in validated:
        buckets[item["safe_strategy_use"]].append({key: item[key] for key in ("claim_id", "claim_text", "safe_strategy_use", "evidence_origin", "supporting_eda_refs", "contradicting_eda_refs", "source_evidence_refs", "limitations")})
    return buckets


def _claim_type(value: str, text: str) -> str:
    normalized = value.lower(); lower = text.lower()
    for kind, terms in {"leakage": ("target encoding", "leak", "submission", "identifier", " id "), "metric": ("auc", "accuracy", "metric", "threshold", "probability"), "validation": ("kfold", "fold", "temporal", "time split", "validation"), "missingness": ("missing", "impute"), "interaction": ("interaction", " cross ", "ratio"), "relationship": ("join", "table"), "model": ("model", "boost", "linear", "neural", "ensemble"), "schema": ("target", "column", "schema"), "drift": ("drift", "shift")}.items():
        if kind == normalized or any(term in lower for term in terms): return kind
    return "feature" if normalized in {"feature", "baseline"} else "unsupported"
def _scope(item: dict[str, Any]) -> str: return str(item.get("source_scope")) if item.get("source_scope") in {"direct_competition", "same_dataset", "analogous_task", "generic_methodology"} else "direct_competition" if item.get("source_type") in {"scout_hypothesis", "notebook"} else "generic_methodology"
def _claim_text(item: Any) -> str: return str(_as_dict(item).get("claim_text") or _as_dict(item).get("claim") or _as_dict(item).get("finding") or "").strip()
def _normalize_text(text: str) -> str: return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_ ]", " ", text.lower())).strip()
def _normalize_name(value: str) -> str: return re.sub(r"[^a-z0-9]", "", value.lower())
def _entity_resolution(text: str, schema: dict[str, Any]) -> tuple[list[str], list[str]]: return (list(schema.get("columns") or []), []) if schema.get("columns") else ([], [])
def _resolve_claim_entities(claim: dict[str, Any], evidence: dict[str, Any]) -> None:
    schema = _as_dict(evidence.get("inferred_schema")); columns = {str(value) for value in (schema.get("global_roles", {}).get("all_columns", []) or [])}
    for table in schema.get("tables", []) or []:
        columns.update(str(item.get("name")) for item in _as_dict(table).get("columns", []) if _as_dict(item).get("name"))
    for key in ("target_column", "primary_id_column", "prediction_column"):
        if schema.get(key): columns.add(str(schema[key]))
    normalized = {_normalize_name(column): column for column in columns}; text = _normalize_text(claim["claim_text"])
    resolved = sorted({column for name, column in normalized.items() if name and name in text.replace(" ", "")})
    claimed = re.findall(r"[`'\"]([A-Za-z_][\w]*)[`'\"]", claim["claim_text"])
    unresolved = sorted({value for value in claimed if _normalize_name(value) not in normalized})
    claim["entities"]["columns"] = resolved; claim["entity_resolution"] = {"resolved_columns": resolved, "unresolved_columns": unresolved, "resolved_tables": [], "unresolved_tables": [], "confidence": "high" if resolved and not unresolved else "low" if unresolved else "medium"}
def _metric_entities(text: str) -> list[str]: return [item for item in ("auc", "accuracy", "rmse", "mae", "f1", "logloss") if item in text.lower()]
def _validation_entities(text: str) -> list[str]: return [item for item in ("temporal", "stratified", "group_kfold", "kfold") if item.replace("_", " ") in text.lower() or item in text.lower()]
def _unsafe_reason(text: str) -> str | None:
    if "target encoding" in text and any(term in text for term in ("global", "full data", "all training", "before cv")): return "Claim recommends target encoding outside validation folds; reject it and use out-of-fold encoding only."
    if any(term in text for term in ("use the identifier", "use id as feature", "submission as feature", "test labels")): return "Claim recommends leakage-prone role or label usage; reject it."
    return None
def _action(status: str, ctype: str) -> str: return {"confirmed": "Adopt the claim because current EDA directly supports it.", "unsafe": "Reject the unsafe source advice and use fold-safe, metric-compatible practice.", "contradicted": "Reject this claim because current-dataset evidence conflicts with it."}.get(status, f"Test this {ctype} claim as a controlled hypothesis before adopting it.")
def _experiment_type(ctype: str) -> str: return {"validation": "validation_comparison", "feature": "feature_ablation", "model": "model_family_comparison", "postprocessing": "postprocessing_test", "relationship": "relationship_test"}.get(ctype, "robustness_test")
def _by_status(items: list[dict[str, Any]], status: str) -> list[dict[str, Any]]: return [item for item in items if item["validation_status"] == status]
def _path_exists(payload: dict[str, Any], path: str) -> bool:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current: return False
        current = current[part]
    return bool(current) or current == 0
def _as_dict(value: Any) -> dict[str, Any]: return value.model_dump(mode="json") if hasattr(value, "model_dump") else value if isinstance(value, dict) else {}
def _skipped() -> dict[str, Any]: return {"status": "skipped", "reason": "no_source_claims", "claim_inventory": [], "validated_claims": [], "unsupported_claims": [], "contradicted_claims": [], "not_testable_claims": [], "analogous_claims": [], "claim_conflicts": [], "source_reliability_summary": [], "strategy_implications": [], "recommended_experiments": [], "final_strategy_claims": {key: [] for key in ("adopt", "test_as_hypothesis", "use_with_caution", "reject", "ignore")}, "warnings": [], "limitations": ["No collected source claims were available for validation."]}


__all__ = ["collect_source_claims", "map_claim_to_eda_evidence", "validate_source_claim_validation", "validate_source_claims"]
