from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.research_scout_schemas import (
    EdaTask,
    ResearchHypothesesPayload,
    ResearchHypothesis,
)


DEFAULT_SCOUT_MODEL = "deepseek-v4-pro"
VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}
VALID_PROVENANCE = {
    "kaggle",
    "arxiv",
    "github",
    "huggingface_papers",
    "domain_memory",
    "heuristic",
    "not_verified_on_data",
}
VALID_CATEGORIES = {
    "validation",
    "leakage",
    "metric",
    "dataset_schema",
    "relationships",
    "drift",
    "feature_engineering",
    "baseline",
    "notebook_reverse_engineering",
    "leaderboard_risk",
}
CATEGORY_PREFIX = {
    "validation": "val",
    "leakage": "leak",
    "metric": "metric",
    "dataset_schema": "schema",
    "relationships": "rel",
    "drift": "drift",
    "feature_engineering": "feat",
    "baseline": "base",
    "notebook_reverse_engineering": "nb",
    "leaderboard_risk": "lb",
}
DEFAULT_EDA_SEQUENCE = [
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "relationship_inferer",
    "validation_analyzer",
    "leakage_checker",
    "drift_analyzer",
    "metric_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_reverse_engineering",
]


SYSTEM_PROMPT = """You are Kaggle Research Scout.
Your job is not to solve the competition and not to perform EDA.
Your job is to convert retrieved evidence into testable EDA hypotheses.

Every important recommendation must become a hypothesis with:
- priority;
- category;
- claim;
- why it matters;
- how to verify on real data;
- expected evidence keys;
- success condition;
- failure condition;
- provenance;
- confidence.

Do not claim that something is true on the dataset unless it can be verified from sources.
Mark dataset-dependent claims as not_verified_on_data.
Prefer actionable checks over broad advice.
Use P0 only for checks that can invalidate the entire strategy.

Return one JSON object matching this shape:
{
  "schema_version": "1.0",
  "competition_id": "...",
  "competition_url": "...",
  "competition_desc": "...",
  "task_type": "...",
  "metric": {"name": "..."},
  "domain": "...",
  "source_summary": {},
  "source_quality_summary": {},
  "hypotheses": [],
  "eda_tasks": [],
  "scout_findings": [],
  "scout_limitations": [],
  "recommended_eda_sequence": [],
  "models_used": {"research_scout": "deepseek-v4-pro"}
}"""


async def build_research_hypotheses(
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: dict,
    retrieved_documents: list[dict],
    source_quality_summary: dict | None = None,
    domain_patterns: list[dict] | None = None,
    *,
    client: DeepSeekClient | None = None,
    model: str = DEFAULT_SCOUT_MODEL,
    return_raw: bool = False,
) -> dict | tuple[dict, dict]:
    plan = _as_dict(plan_data)
    docs = [_as_dict(document) for document in retrieved_documents]
    metric = _metric_payload(plan.get("metric"), plan.get("task_type"), competition_desc)
    source_summary = _source_summary(docs)
    base_payload = {
        "schema_version": "1.0",
        "competition_id": competition_id,
        "competition_url": competition_url,
        "competition_desc": competition_desc,
        "task_type": str(plan.get("task_type") or "unknown"),
        "metric": metric,
        "domain": plan.get("domain"),
        "source_summary": source_summary,
        "source_quality_summary": source_quality_summary,
        "hypotheses": [],
        "eda_tasks": [],
        "scout_findings": [],
        "scout_limitations": [],
        "recommended_eda_sequence": DEFAULT_EDA_SEQUENCE,
        "models_used": {"research_scout": model},
    }

    if client is None:
        raw_payload = base_payload
    else:
        raw_payload = await client.chat_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=json.dumps(
                _build_user_payload(
                    competition_id=competition_id,
                    competition_url=competition_url,
                    competition_desc=competition_desc,
                    plan_data=plan,
                    metric=metric,
                    source_summary=source_summary,
                    retrieved_documents=docs,
                    source_quality_summary=source_quality_summary,
                    domain_patterns=domain_patterns,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            timeout=180,
        )

    merged = {**base_payload, **_as_dict(raw_payload)}
    merged["models_used"] = {**base_payload["models_used"], **_as_dict(merged.get("models_used"))}
    normalized = normalize_research_hypotheses(merged)
    if return_raw:
        return normalized, _as_dict(raw_payload)
    return normalized


def normalize_research_hypotheses(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["schema_version"] = str(normalized.get("schema_version") or "1.0")
    normalized["competition_id"] = str(normalized.get("competition_id") or "unknown-competition")
    normalized["competition_desc"] = str(normalized.get("competition_desc") or "")
    normalized["task_type"] = str(normalized.get("task_type") or "unknown")
    normalized["metric"] = _metric_payload(normalized.get("metric"), normalized.get("task_type"), normalized["competition_desc"])
    normalized["source_summary"] = _as_dict(normalized.get("source_summary"))
    normalized["models_used"] = {
        **_as_dict(normalized.get("models_used")),
        "research_scout": str(_as_dict(normalized.get("models_used")).get("research_scout") or DEFAULT_SCOUT_MODEL),
    }
    normalized["scout_limitations"] = _unique_strings(
        [
            *list(normalized.get("scout_limitations") or []),
            "No real EDA was performed in Research Scout mode.",
            "Hypotheses are based on retrieved sources and heuristics only.",
            "All dataset-dependent claims must be verified by EDA Engine.",
        ]
    )
    normalized["recommended_eda_sequence"] = _unique_strings(
        list(normalized.get("recommended_eda_sequence") or []) + DEFAULT_EDA_SEQUENCE
    )

    hypotheses = [_normalize_hypothesis(item) for item in list(normalized.get("hypotheses") or []) if isinstance(item, dict)]
    tasks = [_normalize_task(item) for item in list(normalized.get("eda_tasks") or []) if isinstance(item, dict)]

    hypotheses.extend(_default_hypotheses(normalized, hypotheses))
    hypotheses = _dedupe_hypotheses(_assign_hypothesis_ids(hypotheses))

    tasks.extend(_default_tasks(normalized, hypotheses, tasks))
    tasks = _dedupe_tasks(_assign_task_ids(tasks))
    _ensure_high_priority_tasks(hypotheses, tasks)
    _ensure_task_relations(hypotheses, tasks)

    normalized["hypotheses"] = hypotheses
    normalized["eda_tasks"] = tasks
    return ResearchHypothesesPayload.model_validate(normalized).model_dump(mode="json")


def validate_research_hypotheses(payload: dict) -> None:
    try:
        model = ResearchHypothesesPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Research Scout payload failed schema validation: {exc}") from exc

    errors: list[str] = []
    hypotheses = model.hypotheses
    tasks = model.eda_tasks
    metric_name = str(model.metric.get("name") or "").lower()

    if len(hypotheses) < 8:
        errors.append("payload must contain at least 8 hypotheses")
    if not any(item.category == "validation" and item.priority == "P0" for item in hypotheses):
        errors.append("payload must contain at least one P0 validation hypothesis")
    if not any(item.category == "leakage" and item.priority == "P0" for item in hypotheses):
        errors.append("payload must contain at least one P0 leakage hypothesis")
    for module in ("schema_inferer", "validation_analyzer", "leakage_checker"):
        if not any(task.module == module for task in tasks):
            errors.append(f"payload must contain at least one {module} task")
    if "stability" in metric_name and not _has_temporal_validation(hypotheses):
        errors.append("stability metrics require a temporal validation hypothesis")

    for item in hypotheses:
        missing = []
        if not item.claim:
            missing.append("claim")
        if not item.why_it_matters:
            missing.append("why_it_matters")
        if not item.how_to_verify:
            missing.append("how_to_verify")
        if not item.provenance:
            missing.append("provenance")
        if not item.confidence:
            missing.append("confidence")
        if missing:
            errors.append(f"hypothesis {item.id} missing required fields: {', '.join(missing)}")
    for task in tasks:
        missing = []
        if not task.module:
            missing.append("module")
        if not task.question:
            missing.append("question")
        if not task.expected_outputs:
            missing.append("expected_outputs")
        if missing:
            errors.append(f"EDA task {task.id} missing required fields: {', '.join(missing)}")

    if errors:
        raise ValueError("Research Scout validation failed:\n- " + "\n- ".join(errors))


def build_research_scout_summary(payload: dict) -> str:
    model = ResearchHypothesesPayload.model_validate(payload)
    by_category: dict[str, list[ResearchHypothesis]] = {}
    for hypothesis in model.hypotheses:
        by_category.setdefault(hypothesis.category, []).append(hypothesis)

    task_by_hypothesis: dict[str, list[EdaTask]] = {}
    for task in model.eda_tasks:
        for hypothesis_id in task.related_hypothesis_ids:
            task_by_hypothesis.setdefault(hypothesis_id, []).append(task)

    lines = [
        f"# Research Scout Summary - {model.competition_id}",
        "",
        "## Executive summary",
        _executive_summary(model),
        "",
        "## P0 EDA checks",
    ]
    p0_hypotheses = [item for item in model.hypotheses if item.priority == "P0"]
    lines.extend(_hypothesis_lines(p0_hypotheses, task_by_hypothesis))

    sections = [
        ("Validation hypotheses", "validation"),
        ("Leakage hypotheses", "leakage"),
        ("Dataset schema hypotheses", "dataset_schema"),
        ("Drift hypotheses", "drift"),
        ("Feature engineering hypotheses", "feature_engineering"),
        ("Baseline hypotheses", "baseline"),
        ("Notebook reverse engineering tasks", "notebook_reverse_engineering"),
    ]
    for title, category in sections:
        lines.extend(["", f"## {title}"])
        lines.extend(_hypothesis_lines(by_category.get(category, []), task_by_hypothesis))

    lines.extend(["", "## Limitations"])
    for limitation in model.scout_limitations:
        lines.append(f"- {limitation}")
    return "\n".join(lines).rstrip() + "\n"


def split_eda_task_plan(payload: dict) -> dict:
    model = ResearchHypothesesPayload.model_validate(payload)
    return {
        "schema_version": model.schema_version,
        "competition_id": model.competition_id,
        "competition_url": model.competition_url,
        "task_type": model.task_type,
        "metric": model.metric,
        "eda_tasks": [task.model_dump(mode="json") for task in model.eda_tasks],
        "hypothesis_index": {
            hypothesis.id: {
                "category": hypothesis.category,
                "priority": hypothesis.priority,
                "claim": hypothesis.claim,
            }
            for hypothesis in model.hypotheses
        },
        "recommended_sequence": model.recommended_eda_sequence,
        "blocking_tasks": [
            task.id for task in model.eda_tasks if task.blocking
        ],
    }


def _build_user_payload(
    *,
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: dict[str, Any],
    metric: dict[str, Any],
    source_summary: dict[str, int],
    retrieved_documents: list[dict[str, Any]],
    source_quality_summary: dict | None,
    domain_patterns: list[dict] | None,
) -> dict[str, Any]:
    return {
        "competition_id": competition_id,
        "competition_url": competition_url,
        "competition_desc": competition_desc,
        "plan_data": plan_data,
        "metric": metric,
        "source_summary": source_summary,
        "source_quality_summary": source_quality_summary,
        "top_retrieved_documents": [_format_doc_for_prompt(doc) for doc in retrieved_documents[:20]],
        "domain_patterns": domain_patterns or [],
        "required_coverage": [
            "validation",
            "leakage",
            "metric",
            "dataset_schema",
            "relationships",
            "drift",
            "feature_engineering",
            "baseline",
            "notebook_reverse_engineering",
        ],
    }


def _format_doc_for_prompt(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(doc.get("metadata"))
    return {
        "id": doc.get("id"),
        "source": doc.get("source"),
        "title": doc.get("title"),
        "url": str(doc.get("url") or ""),
        "content_summary": str(doc.get("summary") or doc.get("content") or "")[:1600],
        "rrf_score": doc.get("rrf_score"),
        "source_quality": {
            key: metadata.get(key)
            for key in ("quality_score", "final_score", "specificity", "evidence_type", "quality_notes")
            if key in metadata
        },
    }


def _normalize_hypothesis(item: dict[str, Any]) -> dict[str, Any]:
    category = str(item.get("category") or "feature_engineering")
    if category not in VALID_CATEGORIES:
        category = "feature_engineering"
    priority = str(item.get("priority") or "P2")
    if priority not in VALID_PRIORITIES:
        priority = "P2"
    provenance = [str(value) for value in list(item.get("provenance") or []) if str(value) in VALID_PROVENANCE]
    if not provenance:
        provenance.append("heuristic")
    if "not_verified_on_data" not in provenance:
        provenance.append("not_verified_on_data")
    status = str(item.get("status") or "needs_eda")
    if status not in {"untested", "source_supported", "heuristic", "needs_eda"}:
        status = "needs_eda"
    confidence = str(item.get("confidence") or "medium")
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "id": str(item.get("id") or ""),
        "category": category,
        "priority": priority,
        "claim": str(item.get("claim") or "Verify this competition-specific assumption with EDA."),
        "why_it_matters": str(item.get("why_it_matters") or "Unverified assumptions can invalidate validation, features, or leaderboard strategy."),
        "how_to_verify": _string_list(item.get("how_to_verify")) or ["Check the relevant evidence on the real train/test files."],
        "expected_evidence_keys": _string_list(item.get("expected_evidence_keys")),
        "failure_condition": item.get("failure_condition"),
        "success_condition": item.get("success_condition"),
        "provenance": _unique_strings(provenance),
        "supporting_source_ids": _string_list(item.get("supporting_source_ids")),
        "confidence": confidence,
        "status": status,
    }


def _normalize_task(item: dict[str, Any]) -> dict[str, Any]:
    module = str(item.get("module") or "table_profiler")
    valid_modules = set(EdaTask.model_fields["module"].annotation.__args__)  # type: ignore[attr-defined]
    if module not in valid_modules:
        module = "table_profiler"
    priority = str(item.get("priority") or "P2")
    if priority not in VALID_PRIORITIES:
        priority = "P2"
    return {
        "id": str(item.get("id") or ""),
        "priority": priority,
        "module": module,
        "question": str(item.get("question") or "What does the dataset show for this hypothesis?"),
        "rationale": str(item.get("rationale") or "This check provides evidence for Research Scout hypotheses."),
        "required_inputs": _string_list(item.get("required_inputs")),
        "expected_outputs": _string_list(item.get("expected_outputs")) or [f"{module}.result"],
        "related_hypothesis_ids": _string_list(item.get("related_hypothesis_ids")),
        "blocking": bool(item.get("blocking", False)),
    }


def _default_hypotheses(payload: dict[str, Any], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric = payload.get("metric") or {}
    metric_name = str(metric.get("name") or "").lower()
    task_type = str(payload.get("task_type") or "").lower()
    domain = str(payload.get("domain") or "").lower()
    temporal = _is_temporal_or_stability(metric_name, payload.get("competition_desc"), domain)
    multi_table = _looks_multi_table(payload.get("competition_desc"), domain)
    existing_categories = {item.get("category") for item in existing}

    defaults = [
        _hypothesis(
            "validation",
            "P0",
            "Identify a reliable temporal or ordered split column before trusting local validation." if temporal else "Choose a validation split that matches the competition holdout mechanism before optimizing models.",
            "A mismatched split can make every feature and model decision look better locally than it will score on the leaderboard.",
            [
                "Inspect train/test files for date, period, week, month, fold, or group columns.",
                "Profile target rate and row counts by candidate split columns.",
                "Compare validation distributions against test-like rows when test metadata is available.",
            ],
            ["inferred_schema.global_roles.candidate_time_columns", "validation_evidence.recommended_validation"],
            "No reliable split or grouping signal exists and random CV is consistent with the competition design.",
            "A temporal, grouped, or otherwise competition-aligned split signal is identified.",
            ["heuristic", "not_verified_on_data"],
            "high" if temporal else "medium",
        ),
        _hypothesis(
            "leakage",
            "P0",
            "Check for train/test ID overlap, target-like columns, and future-information leakage before feature engineering.",
            "Leakage can dominate local validation and produce features that cannot be used safely for the hidden test set.",
            [
                "Compare train and test identifiers for overlap and duplicate entities.",
                "Search column names and metadata for target-like or post-outcome fields.",
                "If dates exist, verify secondary rows do not occur after the prediction timestamp.",
            ],
            ["leakage_evidence.id_overlap", "leakage_evidence.target_like_columns", "leakage_evidence.future_rows"],
            "No overlapping IDs, target-like columns, or lookahead rows are found.",
            "Any unsafe overlap, target proxy, or future row is found.",
            ["heuristic", "not_verified_on_data"],
            "high",
        ),
        _hypothesis(
            "metric",
            "P1",
            f"Implement a local metric check for {metric_name or 'the competition metric'} and confirm whether it is rank-based and probability-based.",
            "A wrong metric implementation can select the wrong model, threshold, calibration, or validation split.",
            [
                "Read the evaluation definition and sample submission requirements.",
                "Confirm whether higher is better and whether raw probabilities are required.",
                "Compute the metric on a small known example before running baselines.",
            ],
            ["metric_evidence.local_metric", "metric_evidence.requires_probabilities", "metric_evidence.rank_based"],
            "The metric cannot be reproduced from available target and prediction columns.",
            "A deterministic local metric implementation matches the competition definition.",
            ["heuristic", "not_verified_on_data"],
            "medium",
        ),
        _hypothesis(
            "dataset_schema",
            "P1",
            "Infer target, ID, group, time, train/test, and sample-submission roles before modeling.",
            "Schema mistakes cause invalid joins, invalid metrics, and accidental leakage.",
            [
                "Inventory all files and table names.",
                "Infer column roles from names, dtypes, cardinality, uniqueness, and train/test presence.",
                "Validate target and submission columns against the sample submission.",
            ],
            ["inferred_schema.global_roles", "file_inventory.files", "sample_submission.schema"],
            "Required target, ID, or submission roles remain ambiguous.",
            "Core roles are assigned with clear confidence and evidence.",
            ["heuristic", "not_verified_on_data"],
            "high",
        ),
        _hypothesis(
            "relationships",
            "P1" if multi_table else "P2",
            "Infer base and secondary table relationships before creating aggregations.",
            "Incorrect joins can multiply rows, drop entities, or leak future information.",
            [
                "Detect primary keys and foreign keys by name, uniqueness, and overlap.",
                "Measure one-to-many coverage and orphan rates.",
                "For dated secondary rows, verify cutoff feasibility relative to the base prediction time.",
            ],
            ["relationship_evidence.base_table", "relationship_evidence.join_coverage", "relationship_evidence.cutoff_feasibility"],
            "No stable relationship graph can be inferred.",
            "Base table, secondary tables, join keys, and row cardinalities are mapped.",
            ["heuristic", "not_verified_on_data"],
            "medium",
        ),
        _hypothesis(
            "drift",
            "P1",
            "Measure train/test drift across missingness, numeric distributions, categorical levels, and periods.",
            "Distribution shift changes which validation and features are trustworthy.",
            [
                "Run adversarial validation using schema-safe features.",
                "Compute missingness drift, numeric PSI, and categorical distribution shift.",
                "If time exists, measure drift by period.",
            ],
            ["drift_evidence.adversarial_auc", "drift_evidence.missingness", "drift_evidence.numeric_psi", "drift_evidence.categorical_shift"],
            "Train and test distributions are stable across major feature families.",
            "Material train/test or period drift is detected.",
            ["heuristic", "not_verified_on_data"],
            "medium",
        ),
        _hypothesis(
            "feature_engineering",
            "P2",
            "Prioritize leakage-safe aggregations, date features, missingness indicators, and high-cardinality encodings.",
            "Tabular competitions often depend on compact, validation-safe feature families before complex model tuning.",
            [
                "Generate features only after validation and leakage checks pass.",
                "Track per-family feature counts, missingness, and validation lift.",
                "Use target encoding or WoE only inside validation folds.",
            ],
            ["feature_evidence.family_lift", "feature_evidence.missingness_indicators", "feature_evidence.encoding_safety"],
            "Feature families add no validation lift or violate leakage constraints.",
            "At least one safe feature family improves aligned validation.",
            ["heuristic", "not_verified_on_data"],
            "medium",
        ),
        _hypothesis(
            "baseline",
            "P1",
            "Build base-table and simple-aggregation baselines with per-period metric tracking before advanced modeling.",
            "Baselines expose schema, metric, validation, and drift problems early.",
            [
                "Train a base-table-only baseline if the task type supports it.",
                "Add simple secondary-table aggregations after relationship checks.",
                "Track global and per-period metric plus feature importance artifacts.",
            ],
            ["baseline_evidence.base_table_metric", "baseline_evidence.simple_aggregation_metric", "baseline_evidence.per_period_metric"],
            "Baseline cannot be trained with validated schema and metric inputs.",
            "Baseline artifacts are reproducible and aligned with validation policy.",
            ["heuristic", "not_verified_on_data"],
            "medium",
        ),
        _hypothesis(
            "notebook_reverse_engineering",
            "P2",
            "Statically inspect top competition-specific notebooks for CV strategy, feature families, model families, metric code, and postprocessing.",
            "Notebook patterns can reveal competition-specific tricks without executing untrusted code.",
            [
                "Rank notebooks by relevance and source quality.",
                "Parse code and markdown statically for validation, features, models, metrics, and postprocessing.",
                "Convert recurring patterns into EDA or experiment hypotheses.",
            ],
            ["notebook_static_analysis.cv_strategy", "notebook_static_analysis.feature_families", "notebook_static_analysis.metric_code"],
            "No relevant competition-specific notebook content is available.",
            "Reusable static patterns are extracted without notebook execution.",
            ["kaggle", "heuristic", "not_verified_on_data"],
            "medium",
        ),
    ]
    if temporal:
        defaults.append(
            _hypothesis(
                "validation",
                "P0",
                "Compare random CV against strict out-of-time or rolling validation for stability-sensitive scoring.",
                "Stability metrics punish period-specific failure modes that random CV can hide.",
                [
                    "Create candidate out-of-time folds from time-like columns.",
                    "Measure target rate, row count, and metric by period.",
                    "Compare random CV estimates with out-of-time estimates after baseline is enabled.",
                ],
                ["validation_evidence.random_vs_oot", "validation_evidence.target_by_period", "baseline_evidence.per_period_metric"],
                "Temporal folds are impossible or equivalent to random CV under the metric.",
                "Out-of-time validation reveals different risk or is required by the metric.",
                ["heuristic", "not_verified_on_data"],
                "high",
            )
        )
    if "metric" not in existing_categories and task_type in {"classification", "binary_classification"}:
        defaults.append(
            _hypothesis(
                "metric",
                "P2",
                "Avoid threshold search unless the official metric consumes hard class labels.",
                "Rank and probability metrics usually reward calibrated ordering, not tuned class thresholds.",
                [
                    "Inspect the metric definition for probability versus label inputs.",
                    "Check the sample submission format.",
                ],
                ["metric_evidence.requires_probabilities", "sample_submission.prediction_columns"],
                "Metric requires labels or a thresholded decision.",
                "Metric consumes probabilities or ranks.",
                ["heuristic", "not_verified_on_data"],
                "medium",
            )
        )
    return defaults


def _default_tasks(
    payload: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_category = _first_hypothesis_id_by_category(hypotheses)
    return [
        _task(
            "eda_file_inventory",
            "P0",
            "file_inventory",
            "Which files, tables, and submission templates are available?",
            "Every downstream EDA module needs a stable file and table inventory.",
            [],
            ["file_inventory.files", "file_inventory.table_roles"],
            [by_category.get("dataset_schema", "")],
            True,
        ),
        _task(
            "eda_schema_001",
            "P0",
            "schema_inferer",
            "Which columns are target, ID, group, time, and submission fields?",
            "Schema roles gate validation, leakage checks, metrics, joins, and baselines.",
            ["file_inventory.files"],
            ["inferred_schema.global_roles", "sample_submission.schema"],
            [by_category.get("dataset_schema", "")],
            True,
        ),
        _task(
            "eda_val_001",
            "P0",
            "validation_analyzer",
            "What validation split best matches the competition holdout and metric?",
            "Validation quality is the main control point for strategy reliability.",
            ["inferred_schema.global_roles", "train-like base table", "test-like base table if available"],
            ["validation_evidence.recommended_validation", "validation_evidence.target_by_period", "validation_evidence.train_test_time_relation"],
            _hypothesis_ids_by_category(hypotheses, "validation"),
            True,
        ),
        _task(
            "eda_leak_001",
            "P0",
            "leakage_checker",
            "Are there ID overlaps, target-like columns, future rows, or group leakage risks?",
            "Leakage can invalidate both local validation and leaderboard submissions.",
            ["inferred_schema.global_roles", "relationship_evidence if available"],
            ["leakage_evidence.id_overlap", "leakage_evidence.target_like_columns", "leakage_evidence.future_rows", "leakage_evidence.group_leakage"],
            _hypothesis_ids_by_category(hypotheses, "leakage"),
            True,
        ),
        _task(
            "eda_metric_001",
            "P1",
            "metric_analyzer",
            "Can the local metric be computed and does it require probabilities, ranks, time, or groups?",
            "Metric semantics determine validation reporting and model selection.",
            ["target column", "prediction column", "sample submission"],
            ["metric_evidence.local_metric", "metric_evidence.rank_based", "metric_evidence.requires_probabilities"],
            _hypothesis_ids_by_category(hypotheses, "metric"),
            False,
        ),
        _task(
            "eda_rel_001",
            "P1",
            "relationship_inferer",
            "Which tables join to the base table and what one-to-many coverage do they have?",
            "Relationship evidence prevents row multiplication and unsafe secondary-table features.",
            ["file_inventory.files", "inferred_schema.global_roles"],
            ["relationship_evidence.base_table", "relationship_evidence.join_coverage", "relationship_evidence.cutoff_feasibility"],
            _hypothesis_ids_by_category(hypotheses, "relationships"),
            False,
        ),
        _task(
            "eda_drift_001",
            "P1",
            "drift_analyzer",
            "How different are train and test distributions and period distributions?",
            "Drift evidence informs validation, feature selection, and leaderboard risk.",
            ["train-like table", "test-like table", "candidate time columns"],
            ["drift_evidence.adversarial_auc", "drift_evidence.missingness", "drift_evidence.numeric_psi", "drift_evidence.categorical_shift"],
            _hypothesis_ids_by_category(hypotheses, "drift"),
            False,
        ),
        _task(
            "eda_base_001",
            "P1",
            "baseline_runner",
            "What do base-table and simple-aggregation baselines reveal about metric, drift, and validation?",
            "Baseline artifacts make Scout hypotheses measurable before deeper modeling work.",
            ["validated schema", "validated metric", "recommended validation"],
            ["baseline_evidence.base_table_metric", "baseline_evidence.per_period_metric", "baseline_evidence.feature_importance"],
            _hypothesis_ids_by_category(hypotheses, "baseline"),
            False,
        ),
        _task(
            "eda_feat_001",
            "P2",
            "feature_probe",
            "Which leakage-safe feature families deserve experiments after validation passes?",
            "Feature probes prioritize engineering work with evidence.",
            ["validated schema", "relationship_evidence", "leakage_evidence"],
            ["feature_evidence.family_lift", "feature_evidence.encoding_safety"],
            _hypothesis_ids_by_category(hypotheses, "feature_engineering"),
            False,
        ),
        _task(
            "eda_nb_001",
            "P2",
            "notebook_reverse_engineering",
            "What static patterns appear in top competition-specific notebooks?",
            "Static notebook analysis can extract useful ideas without executing notebooks.",
            ["retrieved Kaggle notebook sources"],
            ["notebook_static_analysis.cv_strategy", "notebook_static_analysis.feature_families", "notebook_static_analysis.metric_code"],
            _hypothesis_ids_by_category(hypotheses, "notebook_reverse_engineering"),
            False,
        ),
    ]


def _ensure_high_priority_tasks(hypotheses: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    related = {hypothesis_id for task in tasks for hypothesis_id in task.get("related_hypothesis_ids", [])}
    module_by_category = {
        "validation": "validation_analyzer",
        "leakage": "leakage_checker",
        "metric": "metric_analyzer",
        "dataset_schema": "schema_inferer",
        "relationships": "relationship_inferer",
        "drift": "drift_analyzer",
        "feature_engineering": "feature_probe",
        "baseline": "baseline_runner",
        "notebook_reverse_engineering": "notebook_reverse_engineering",
        "leaderboard_risk": "drift_analyzer",
    }
    for hypothesis in hypotheses:
        if hypothesis["priority"] not in {"P0", "P1"} or hypothesis["id"] in related:
            continue
        module = module_by_category.get(hypothesis["category"], "table_profiler")
        tasks.append(
            _task(
                "",
                hypothesis["priority"],
                module,
                f"How should EDA verify: {hypothesis['claim']}",
                hypothesis["why_it_matters"],
                [],
                hypothesis.get("expected_evidence_keys") or [f"{module}.result"],
                [hypothesis["id"]],
                hypothesis["priority"] == "P0",
            )
        )


def _ensure_task_relations(hypotheses: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    by_category = _first_hypothesis_id_by_category(hypotheses)
    category_by_module = {
        "schema_inferer": "dataset_schema",
        "file_inventory": "dataset_schema",
        "validation_analyzer": "validation",
        "leakage_checker": "leakage",
        "metric_analyzer": "metric",
        "relationship_inferer": "relationships",
        "drift_analyzer": "drift",
        "baseline_runner": "baseline",
        "feature_probe": "feature_engineering",
        "notebook_reverse_engineering": "notebook_reverse_engineering",
    }
    known = {hypothesis["id"] for hypothesis in hypotheses}
    for task in tasks:
        task["related_hypothesis_ids"] = [
            hypothesis_id for hypothesis_id in task.get("related_hypothesis_ids", []) if hypothesis_id in known
        ]
        if not task["related_hypothesis_ids"]:
            category = category_by_module.get(task["module"])
            hypothesis_id = by_category.get(category or "")
            if hypothesis_id:
                task["related_hypothesis_ids"] = [hypothesis_id]


def _assign_hypothesis_ids(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    counters: Counter[str] = Counter()
    for hypothesis in hypotheses:
        category = hypothesis["category"]
        prefix = CATEGORY_PREFIX.get(category, "hyp")
        raw_id = str(hypothesis.get("id") or "")
        if not raw_id or raw_id in used:
            counters[prefix] += 1
            raw_id = f"{prefix}_{counters[prefix]:03d}"
            while raw_id in used:
                counters[prefix] += 1
                raw_id = f"{prefix}_{counters[prefix]:03d}"
        else:
            match = re.match(rf"^{re.escape(prefix)}_(\d+)$", raw_id)
            if match:
                counters[prefix] = max(counters[prefix], int(match.group(1)))
        hypothesis["id"] = raw_id
        used.add(raw_id)
    return hypotheses


def _assign_task_ids(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    counters: Counter[str] = Counter()
    for task in tasks:
        module_prefix = str(task.get("module", "task")).replace("_analyzer", "").replace("_inferer", "")
        raw_id = str(task.get("id") or "")
        if not raw_id or raw_id in used:
            counters[module_prefix] += 1
            raw_id = f"eda_{module_prefix}_{counters[module_prefix]:03d}"
            while raw_id in used:
                counters[module_prefix] += 1
                raw_id = f"eda_{module_prefix}_{counters[module_prefix]:03d}"
        task["id"] = raw_id
        used.add(raw_id)
    return tasks


def _dedupe_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        key = hypothesis["id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hypothesis)
    return deduped


def _dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for task in tasks:
        key = task["id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped


def _metric_payload(metric: Any, task_type: Any, description: str | None = None) -> dict[str, Any]:
    if isinstance(metric, dict):
        name = str(metric.get("name") or metric.get("metric") or "unknown")
        payload = dict(metric)
    else:
        name = str(metric or "unknown")
        payload = {}
    normalized = name.lower()
    text = f"{normalized} {description or ''}".lower()
    rank_based = any(token in text for token in ("gini", "auc", "ndcg", "map@", "rank"))
    requires_probabilities = any(token in text for token in ("gini", "auc", "logloss", "log_loss", "probab"))
    requires_time_or_groups = any(token in text for token in ("stability", "time", "week", "period", "group"))
    higher_is_better = not any(token in normalized for token in ("loss", "rmse", "mae", "error"))
    return {
        "name": name,
        "higher_is_better": bool(payload.get("higher_is_better", higher_is_better)),
        "rank_based": bool(payload.get("rank_based", rank_based)),
        "requires_probabilities": bool(payload.get("requires_probabilities", requires_probabilities)),
        "requires_time_or_groups": bool(payload.get("requires_time_or_groups", requires_time_or_groups)),
    }


def _source_summary(docs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(doc.get("source") or "unknown") for doc in docs)
    return {
        "kaggle": counts.get("kaggle", 0),
        "arxiv": counts.get("arxiv", 0),
        "github": counts.get("github", 0),
        "huggingface_papers": counts.get("huggingface_papers", 0),
    }


def _has_temporal_validation(hypotheses: list[ResearchHypothesis]) -> bool:
    terms = ("time", "temporal", "period", "week", "month", "out-of-time", "chronological", "rolling", "stability")
    return any(
        hypothesis.category == "validation"
        and any(term in hypothesis.claim.lower() for term in terms)
        for hypothesis in hypotheses
    )


def _is_temporal_or_stability(metric_name: str, description: Any, domain: str) -> bool:
    text = f"{metric_name} {description or ''} {domain}".lower()
    return any(term in text for term in ("stability", "temporal", "time", "week_num", "date", "period"))


def _looks_multi_table(description: Any, domain: str) -> bool:
    text = f"{description or ''} {domain}".lower()
    return any(term in text for term in ("multi-table", "multiple tables", "secondary table", "relational", "home credit", "tabular_credit"))


def _hypothesis(
    category: str,
    priority: str,
    claim: str,
    why_it_matters: str,
    how_to_verify: list[str],
    expected_evidence_keys: list[str],
    failure_condition: str,
    success_condition: str,
    provenance: list[str],
    confidence: str,
) -> dict[str, Any]:
    return {
        "id": "",
        "category": category,
        "priority": priority,
        "claim": claim,
        "why_it_matters": why_it_matters,
        "how_to_verify": how_to_verify,
        "expected_evidence_keys": expected_evidence_keys,
        "failure_condition": failure_condition,
        "success_condition": success_condition,
        "provenance": provenance,
        "supporting_source_ids": [],
        "confidence": confidence,
        "status": "needs_eda",
    }


def _task(
    task_id: str,
    priority: str,
    module: str,
    question: str,
    rationale: str,
    required_inputs: list[str],
    expected_outputs: list[str],
    related_hypothesis_ids: list[str],
    blocking: bool,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "priority": priority,
        "module": module,
        "question": question,
        "rationale": rationale,
        "required_inputs": required_inputs,
        "expected_outputs": expected_outputs,
        "related_hypothesis_ids": [item for item in related_hypothesis_ids if item],
        "blocking": blocking,
    }


def _hypothesis_ids_by_category(hypotheses: list[dict[str, Any]], category: str) -> list[str]:
    return [item["id"] for item in hypotheses if item.get("category") == category]


def _first_hypothesis_id_by_category(hypotheses: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for hypothesis in hypotheses:
        result.setdefault(str(hypothesis.get("category")), str(hypothesis.get("id")))
    return result


def _executive_summary(model: ResearchHypothesesPayload) -> str:
    p0 = [hypothesis.claim for hypothesis in model.hypotheses if hypothesis.priority == "P0"]
    if p0:
        return "Scout recommends verifying first: " + "; ".join(p0[:3]) + "."
    return "Scout recommends starting with schema, validation, and leakage checks before feature work."


def _hypothesis_lines(
    hypotheses: list[ResearchHypothesis],
    task_by_hypothesis: dict[str, list[EdaTask]],
) -> list[str]:
    if not hypotheses:
        return ["- No hypotheses in this category."]
    lines: list[str] = []
    for hypothesis in hypotheses:
        task_ids = [task.id for task in task_by_hypothesis.get(hypothesis.id, [])]
        suffix = f" Related tasks: {', '.join(task_ids)}." if task_ids else ""
        lines.append(f"- [{hypothesis.priority}] {hypothesis.id}: {hypothesis.claim}{suffix}")
    return lines


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique_strings(values: list[Any]) -> list[str]:
    return [item for item in dict.fromkeys(str(value) for value in values if str(value))]
