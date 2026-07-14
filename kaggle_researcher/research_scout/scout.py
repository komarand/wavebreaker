from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.research_hypotheses import (
    ALLOWED_HYPOTHESIS_CATEGORIES,
    ResearchHypotheses,
)
from kaggle_researcher.contracts.eda_task_plan import EdaTaskPlan
from kaggle_researcher.research_scout.prompts import (
    RESEARCH_SCOUT_OUTPUT_INSTRUCTIONS,
    RESEARCH_SCOUT_SYSTEM_PROMPT,
)
from kaggle_researcher.research_scout.schemas import (
    EdaTaskPlanDraft,
    ResearchScoutOutput,
    ScoutEdaTask,
    ScoutHypothesis,
    ScoutLimitation,
    ScoutStructuredFinding,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument


DEFAULT_MODEL = "deepseek-v4-pro"
CORE_MODULE_SEQUENCE = [
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
]
CORE_BLOCKING_MODULES = [
    "file_inventory", "schema_inferer", "validation_analyzer", "leakage_checker"
]


async def run_research_scout(
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    client: DeepSeekClient,
    model: str = DEFAULT_MODEL,
) -> ResearchScoutOutput:
    """Generate generic EDA hypotheses from source evidence, with deterministic fallback."""

    raw: Any = {}
    try:
        raw = await client.chat_json(
            model=model,
            system_prompt=RESEARCH_SCOUT_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(
                competition_id=competition_id,
                competition_url=competition_url,
                competition_desc=competition_desc,
                plan_data=plan_data,
                retrieved_documents=retrieved_documents,
            ),
        )
        output = _output_from_payload(
            raw,
            competition_id=competition_id,
            competition_url=competition_url,
            competition_desc=competition_desc,
            plan_data=plan_data,
            model=model,
        )
        _validate_against_eda_schemas(output)
    except (ValidationError, ValueError) as exc:
        try:
            repaired = await _repair_scout_payload_once(
                client=client,
                model=model,
                invalid_payload=raw,
                validation_error=exc,
            )
            output = _output_from_payload(
                repaired,
                competition_id=competition_id,
                competition_url=competition_url,
                competition_desc=competition_desc,
                plan_data=plan_data,
                model=model,
            )
            _validate_against_eda_schemas(output)
        except Exception as repair_exc:
            output = _fallback_output(
                competition_id=competition_id,
                competition_url=competition_url,
                competition_desc=competition_desc,
                plan_data=plan_data,
                retrieved_documents=retrieved_documents,
                model=model,
                reason=str(repair_exc),
            )
    except Exception as exc:
        output = _fallback_output(
            competition_id=competition_id,
            competition_url=competition_url,
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            model=model,
            reason=str(exc),
        )

    return output


async def _repair_scout_payload_once(
    *,
    client: DeepSeekClient,
    model: str,
    invalid_payload: Any,
    validation_error: Exception,
) -> dict[str, Any]:
    response = await client.chat_json(
        model=model,
        system_prompt=(
            "Correct the existing Research Scout payload to the canonical research-to-EDA "
            "artifact schema. Do not add hypotheses or invent hypothesis IDs. Preserve task "
            "intent. Use hypothesis_id and task_id, never id. Every hypothesis_index value "
            "must be an array of task IDs. Return JSON only."
        ),
        user_prompt=json.dumps({
            "validation_errors": [str(validation_error)[:2000]],
            "allowed_categories": ALLOWED_HYPOTHESIS_CATEGORIES,
            "canonical_task_schema": EdaTaskPlan.model_json_schema(),
            "invalid_payload": invalid_payload,
        }, ensure_ascii=False),
    )
    if not isinstance(response, dict):
        raise ValueError("Scout repair response must be a JSON object.")
    return response


def _build_user_prompt(
    *,
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
) -> str:
    payload = {
        "competition_id": competition_id,
        "competition_url": competition_url,
        "competition_desc": competition_desc,
        "plan_data": plan_data.model_dump(mode="json"),
        "retrieved_documents": [
            _document_payload(document) for document in retrieved_documents[:12]
        ],
        "instructions": RESEARCH_SCOUT_OUTPUT_INSTRUCTIONS,
        "required_core_hypotheses": ["schema_001", "metric_001", "val_001", "leak_001"],
        "canonical_hypothesis_contract": {
            "required_fields": [
                "hypothesis_id", "category", "claim", "confidence_before_eda",
            ],
            "allowed_categories": ALLOWED_HYPOTHESIS_CATEGORIES,
        },
        "canonical_task_plan_contract": {
            "task_fields": list(EdaTaskPlan.model_json_schema()["$defs"]["EdaTask"]["properties"]),
            "hypothesis_index_value_type": "array of task_id strings",
        },
        "generic_rules": [
            "Separate source facts from dataset-dependent hypotheses.",
            "Do not say EDA has already run.",
            "Every hypothesis must include expected_eda_checks.",
            "P0 blocking checks must cover schema, metric, validation, and leakage.",
            "Do not force temporal validation for ordinary tabular classification or regression.",
            "Only create temporal validation hypotheses when metric/source/description supports them.",
            "Only create group validation hypotheses when group/entity/query risk is plausible.",
            "Use hypothesis_id, never id, and always provide confidence_before_eda.",
            "Use relationship, feature, schema, and notebook; do not invent category aliases.",
            "Use task_id, never id; every hypothesis_index value must be a JSON array of task IDs.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _output_from_payload(
    payload: dict[str, Any],
    *,
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    model: str,
) -> ResearchScoutOutput:
    normalised = dict(payload)
    raw_hypotheses = normalised.get("hypotheses") or normalised.get("research_hypotheses") or []
    raw_tasks = _raw_tasks(normalised)
    hypotheses = _normalise_hypotheses(raw_hypotheses, plan_data=plan_data)
    hypotheses = _ensure_core_hypotheses(hypotheses, plan_data=plan_data)
    tasks = _normalise_tasks(raw_tasks, hypotheses=hypotheses)
    tasks = _ensure_core_tasks(tasks, hypotheses=hypotheses)
    task_plan = _task_plan_from_payload(
        normalised,
        competition_id=competition_id,
        plan_data=plan_data,
        tasks=tasks,
    )
    structured_findings = _normalise_findings(normalised.get("structured_findings", []))
    limitations = _normalise_limitations(normalised.get("scout_limitations", []))
    summary = normalised.get("summary") or normalised.get("research_scout_summary")

    return ResearchScoutOutput(
        competition_id=competition_id,
        competition_url=competition_url,
        task_type=_task_type(plan_data),
        metric={"name": _metric_name(plan_data)},
        dataset=normalised.get("dataset") or {},
        hypotheses=hypotheses,
        eda_task_plan=task_plan,
        structured_findings=structured_findings,
        scout_limitations=limitations,
        models_used={**normalised.get("models_used", {}), "research_scout": model},
        summary=summary,
    )


def _fallback_output(
    *,
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    model: str,
    reason: str,
) -> ResearchScoutOutput:
    hypotheses = _ensure_core_hypotheses([], plan_data=plan_data)
    tasks = _ensure_core_tasks([], hypotheses=hypotheses)
    task_plan = EdaTaskPlanDraft(
        competition_id=competition_id,
        task_type=_task_type(plan_data),
        metric={"name": _metric_name(plan_data)},
        dataset={},
        eda_tasks=tasks,
        hypothesis_index=_hypothesis_index(tasks),
        recommended_module_sequence=list(CORE_MODULE_SEQUENCE),
        recommended_human_checklist=[
            "Confirm the official metric and submission format from competition documentation.",
            "Inspect train/test schema before choosing feature engineering assumptions.",
        ],
        blocking_tasks=list(CORE_BLOCKING_MODULES),
    )
    limitations = [
        ScoutLimitation(
            limitation_id="lim_001",
            description=f"LLM Research Scout failed; deterministic fallback was used: {reason}",
            severity="medium",
            affected_outputs=["research_hypotheses", "eda_task_plan"],
        )
    ]
    findings = [
        ScoutStructuredFinding(
            finding_id="finding_001",
            category="schema",
            finding="Fallback generated generic EDA hypotheses without claiming dataset facts.",
            evidence_refs=["schema_001"],
            source_refs=[document.id for document in retrieved_documents[:3]],
            confidence="low",
        )
    ]
    return ResearchScoutOutput(
        competition_id=competition_id,
        competition_url=competition_url,
        task_type=_task_type(plan_data),
        metric={"name": _metric_name(plan_data)},
        dataset={},
        hypotheses=hypotheses,
        eda_task_plan=task_plan,
        structured_findings=findings,
        scout_limitations=limitations,
        models_used={"research_scout": model, "fallback": True},
    )


def _ensure_core_hypotheses(
    hypotheses: list[ScoutHypothesis],
    *,
    plan_data: PlanData,
) -> list[ScoutHypothesis]:
    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    for hypothesis in _core_hypotheses(plan_data):
        by_id.setdefault(hypothesis.hypothesis_id, hypothesis)
    return sorted(by_id.values(), key=_hypothesis_sort_key)


def _core_hypotheses(plan_data: PlanData) -> list[ScoutHypothesis]:
    metric_name = _metric_name(plan_data)
    task_type = _task_type(plan_data)
    normalized_metric = metric_name.strip().lower().replace("-", "_").replace(" ", "_")
    validation_checks = ["validation_analyzer.primary_policy"]
    leakage_checks = ["leakage_checker.basic"]
    if normalized_metric == "gini_stability":
        validation_checks.append("validation_analyzer.temporal_cv_feasibility")
    if task_type == "ranking":
        validation_checks.append("validation_analyzer.ranking_validation")
        leakage_checks.append("leakage_checker.ranking_query_overlap")
    return [
        ScoutHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Infer generic train/test/base table roles, target, id, and submission columns.",
            rationale="Schema roles are required before validation, leakage checks, and modeling.",
            expected_eda_checks=["schema_inferer.roles", "file_inventory.roles"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=[],
        ),
        ScoutHypothesis(
            hypothesis_id="metric_001",
            category="metric",
            claim=f"Resolve metric '{metric_name}' and required prediction semantics.",
            rationale="The metric determines output type, validation diagnostics, and scoring.",
            expected_eda_checks=["metric_analyzer.registry"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=[],
        ),
        ScoutHypothesis(
            hypothesis_id="val_001",
            category="validation",
            claim=f"Select a primary validation policy for {task_type} from metric and data evidence.",
            rationale="Validation must fit task structure without assuming temporal splits by default.",
            expected_eda_checks=validation_checks,
            priority="P0",
            confidence_before_eda="medium",
            source_refs=[],
        ),
        ScoutHypothesis(
            hypothesis_id="leak_001",
            category="leakage",
            claim="Check generic leakage risks before using any features for modeling.",
            rationale="Leakage can invalidate local validation and strategy decisions.",
            expected_eda_checks=leakage_checks,
            priority="P0",
            confidence_before_eda="medium",
            source_refs=[],
        ),
        ScoutHypothesis(
            hypothesis_id="drift_001",
            category="drift",
            claim="Measure train/test drift as diagnostic evidence when shared columns exist.",
            rationale="Drift can inform risk without retroactively forcing temporal validation.",
            expected_eda_checks=["drift_analyzer.generic"],
            priority="P1",
            confidence_before_eda="low",
            source_refs=[],
        ),
    ]


def _normalise_hypotheses(
    raw_hypotheses: list[Any],
    *,
    plan_data: PlanData,
) -> list[ScoutHypothesis]:
    hypotheses: list[ScoutHypothesis] = []
    for index, item in enumerate(raw_hypotheses, start=1):
        if not isinstance(item, dict):
            continue
        category = _normalise_category(item.get("category"))
        hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or _generated_id(category, index))
        data = {
            "hypothesis_id": hypothesis_id,
            "category": category,
            "claim": item.get("claim") or _default_claim(category, plan_data),
            "rationale": item.get("rationale") or item.get("why_it_matters") or "This must be verified by EDA before strategy decisions.",
            "expected_eda_checks": item.get("expected_eda_checks") or item.get("how_to_verify") or [_default_check(category)],
            "priority": item.get("priority") or ("P0" if category in {"schema", "metric", "validation", "leakage"} else "P1"),
            "confidence_before_eda": item.get("confidence_before_eda") or item.get("confidence") or "medium",
            "source_refs": item.get("source_refs") or item.get("supporting_source_ids") or [],
            "status": item.get("status") or "needs_eda",
        }
        try:
            hypotheses.append(ScoutHypothesis(**data))
        except ValidationError:
            fallback = _core_hypothesis_for_category(category, plan_data)
            if fallback is not None:
                hypotheses.append(fallback)
    return _dedupe_hypotheses(hypotheses)


def _normalise_tasks(
    raw_tasks: list[Any],
    *,
    hypotheses: list[ScoutHypothesis],
) -> list[ScoutEdaTask]:
    hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses}
    tasks: list[ScoutEdaTask] = []
    for index, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            continue
        module = _normalise_module(item.get("module"))
        related = [
            str(value) for value in item.get("related_hypothesis_ids", []) if str(value) in hypothesis_ids
        ]
        try:
            tasks.append(
                ScoutEdaTask(
                    task_id=str(item.get("task_id") or item.get("id") or f"{module}_{index:03d}"),
                    module=module,
                    priority=item.get("priority") or "P0",
                    blocking=bool(item.get("blocking", False)),
                    related_hypothesis_ids=related,
                    params=item.get("params") or {},
                )
            )
        except ValidationError:
            continue
    return _dedupe_tasks(tasks)


def _ensure_core_tasks(
    tasks: list[ScoutEdaTask],
    *,
    hypotheses: list[ScoutHypothesis],
) -> list[ScoutEdaTask]:
    by_id = {task.task_id: task for task in tasks}
    available = {hypothesis.hypothesis_id for hypothesis in hypotheses}
    for task in [
        ScoutEdaTask(task_id="file_inventory_001", module="file_inventory", priority="P0", blocking=True, related_hypothesis_ids=["schema_001"]),
        ScoutEdaTask(task_id="schema_001", module="schema_inferer", priority="P0", blocking=True, related_hypothesis_ids=["schema_001"]),
        ScoutEdaTask(task_id="metric_001", module="metric_analyzer", priority="P0", blocking=False, related_hypothesis_ids=["metric_001"]),
        ScoutEdaTask(task_id="validation_001", module="validation_analyzer", priority="P0", blocking=True, related_hypothesis_ids=["val_001"]),
        ScoutEdaTask(task_id="leakage_001", module="leakage_checker", priority="P0", blocking=True, related_hypothesis_ids=["leak_001"]),
        ScoutEdaTask(task_id="drift_001", module="drift_analyzer", priority="P1", blocking=False, related_hypothesis_ids=["drift_001"]),
    ]:
        task.related_hypothesis_ids = [
            hypothesis_id for hypothesis_id in task.related_hypothesis_ids if hypothesis_id in available
        ]
        by_id.setdefault(task.task_id, task)
    return list(by_id.values())


def _task_plan_from_payload(
    payload: dict[str, Any],
    *,
    competition_id: str,
    plan_data: PlanData,
    tasks: list[ScoutEdaTask],
) -> EdaTaskPlanDraft:
    raw_plan = payload.get("eda_task_plan") or payload.get("task_plan") or {}
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    sequence = raw_plan.get("recommended_module_sequence") or payload.get("recommended_module_sequence") or CORE_MODULE_SEQUENCE
    blocking = raw_plan.get("blocking_tasks") or CORE_BLOCKING_MODULES
    return EdaTaskPlanDraft(
        competition_id=competition_id,
        task_type=raw_plan.get("task_type") or _task_type(plan_data),
        metric=raw_plan.get("metric") or {"name": _metric_name(plan_data)},
        dataset=raw_plan.get("dataset") or {},
        eda_tasks=tasks,
        hypothesis_index=raw_plan.get("hypothesis_index") or _hypothesis_index(tasks),
        recommended_module_sequence=_normalise_module_sequence(sequence),
        recommended_human_checklist=raw_plan.get("recommended_human_checklist") or [
            "Confirm official metric, target column, and submission format before modeling."
        ],
        blocking_tasks=_normalise_blocking_tasks(blocking),
    )


def _raw_tasks(payload: dict[str, Any]) -> list[Any]:
    raw_plan = payload.get("eda_task_plan") or payload.get("task_plan") or {}
    if isinstance(raw_plan, dict) and raw_plan.get("eda_tasks"):
        return list(raw_plan["eda_tasks"])
    return list(payload.get("eda_tasks") or [])


def _normalise_findings(raw_findings: Any) -> list[ScoutStructuredFinding]:
    findings: list[ScoutStructuredFinding] = []
    if not isinstance(raw_findings, list):
        return findings
    for index, item in enumerate(raw_findings, start=1):
        if isinstance(item, str):
            item = {"finding": item}
        if not isinstance(item, dict):
            continue
        try:
            findings.append(
                ScoutStructuredFinding(
                    finding_id=str(item.get("finding_id") or item.get("id") or f"finding_{index:03d}"),
                    category=_normalise_category(item.get("category")),
                    finding=item.get("finding") or item.get("claim") or "Source-backed scout finding.",
                    evidence_refs=item.get("evidence_refs") or [],
                    source_refs=item.get("source_refs") or item.get("supporting_source_ids") or [],
                    confidence=item.get("confidence") or "medium",
                )
            )
        except ValidationError:
            continue
    return findings


def _normalise_limitations(raw_limitations: Any) -> list[ScoutLimitation]:
    limitations: list[ScoutLimitation] = []
    if not isinstance(raw_limitations, list):
        return limitations
    for index, item in enumerate(raw_limitations, start=1):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, dict):
            continue
        try:
            limitations.append(
                ScoutLimitation(
                    limitation_id=str(item.get("limitation_id") or item.get("id") or f"lim_{index:03d}"),
                    description=item.get("description") or item.get("limitation") or "Scout limitation.",
                    severity=item.get("severity") or "medium",
                    affected_outputs=item.get("affected_outputs") or [],
                )
            )
        except ValidationError:
            continue
    return limitations


def _validate_against_eda_schemas(output: ResearchScoutOutput) -> None:
    ResearchHypotheses(**output.to_research_hypotheses_payload())
    EdaTaskPlan(**output.to_eda_task_plan_payload())


def _document_payload(document: RetrievedDocument) -> dict[str, Any]:
    return {
        "id": document.id,
        "source": document.source,
        "title": document.title,
        "content": document.content[:2500],
        "score": document.score,
        "rrf_score": document.rrf_score,
        "metadata": document.metadata,
    }


def _hypothesis_index(tasks: list[ScoutEdaTask]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for task in tasks:
        for hypothesis_id in task.related_hypothesis_ids:
            index.setdefault(hypothesis_id, []).append(task.task_id)
    return index


def _normalise_module_sequence(values: Any) -> list[str]:
    modules = []
    for value in values if isinstance(values, list) else CORE_MODULE_SEQUENCE:
        module = _normalise_module(value)
        if module not in modules:
            modules.append(module)
    return modules


def _normalise_blocking_tasks(values: Any) -> list[str]:
    modules = []
    for value in values if isinstance(values, list) else CORE_BLOCKING_MODULES:
        module = _normalise_module(value)
        if module not in modules:
            modules.append(module)
    return modules


def _normalise_category(value: Any) -> str:
    normalized = str(value or "schema").strip().lower()
    aliases = {
        "dataset_schema": "schema",
        "relationships": "relationship",
        "feature_engineering": "feature",
        "notebook_reverse_engineering": "notebook",
        "leaderboard_risk": "leaderboard",
    }
    return aliases.get(normalized, normalized)


def _normalise_module(value: Any) -> str:
    normalized = str(value or "schema_inferer").strip().lower()
    aliases = {
        "notebook_reverse_engineering": "notebook_static_analysis",
        "relationship_analyzer": "relationship_inferer",
    }
    return aliases.get(normalized, normalized)


def _generated_id(category: str, index: int) -> str:
    prefixes = {
        "schema": "schema",
        "metric": "metric",
        "validation": "val",
        "leakage": "leak",
        "relationship": "rel",
        "drift": "drift",
        "baseline": "base",
        "feature": "feat",
        "notebook": "nb",
        "leaderboard": "lb",
        "data_quality": "dq",
    }
    return f"{prefixes.get(category, 'schema')}_{index:03d}"


def _default_claim(category: str, plan_data: PlanData) -> str:
    return f"Validate {category} assumptions for {_task_type(plan_data)} before strategy decisions."


def _default_check(category: str) -> str:
    checks = {
        "schema": "schema_inferer.roles",
        "metric": "metric_analyzer.registry",
        "validation": "validation_analyzer.primary_policy",
        "leakage": "leakage_checker.basic",
        "relationship": "relationship_inferer.generic",
        "drift": "drift_analyzer.generic",
        "baseline": "baseline_runner.honest_baseline",
        "feature": "feature_probe.families",
        "notebook": "notebook_static_analysis.patterns",
        "leaderboard": "notebook_static_analysis.leaderboard_risk",
        "data_quality": "table_profiler.quality",
    }
    return checks.get(category, "schema_inferer.roles")


def _core_hypothesis_for_category(
    category: str,
    plan_data: PlanData,
) -> ScoutHypothesis | None:
    for hypothesis in _core_hypotheses(plan_data):
        if hypothesis.category == category:
            return hypothesis
    return None


def _dedupe_hypotheses(hypotheses: list[ScoutHypothesis]) -> list[ScoutHypothesis]:
    seen = set()
    result = []
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in seen:
            continue
        seen.add(hypothesis.hypothesis_id)
        result.append(hypothesis)
    return result


def _dedupe_tasks(tasks: list[ScoutEdaTask]) -> list[ScoutEdaTask]:
    seen = set()
    result = []
    for task in tasks:
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        result.append(task)
    return result


def _hypothesis_sort_key(hypothesis: ScoutHypothesis) -> tuple[int, str]:
    order = {"schema_001": 0, "metric_001": 1, "val_001": 2, "leak_001": 3, "drift_001": 4}
    return (order.get(hypothesis.hypothesis_id, 100), hypothesis.hypothesis_id)


def _task_type(plan_data: PlanData) -> str:
    return str(plan_data.task_type or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


def _metric_name(plan_data: PlanData) -> str:
    return str(plan_data.metric or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


__all__ = ["run_research_scout"]
