from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.research_scout_schemas import (
    EdaTask,
    ResearchHypothesesPayload,
    ResearchHypothesis,
    ScoutFinding,
    VerificationStep,
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
ALLOWED_EDA_MODULES = {
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
}
MODULE_SHORT_NAMES = {
    "file_inventory": "file",
    "schema_inferer": "schema",
    "table_profiler": "profile",
    "relationship_inferer": "rel",
    "validation_analyzer": "val",
    "leakage_checker": "leak",
    "drift_analyzer": "drift",
    "metric_analyzer": "metric",
    "baseline_runner": "base",
    "feature_probe": "feat",
    "notebook_reverse_engineering": "nb",
}
GLOBAL_TASK_MODULES = {"file_inventory", "table_profiler"}
GENERIC_TABLE_PROFILE_PHRASE = "what does the dataset show for this hypothesis?"
TEMPORAL_VALIDATION_CLAIM = (
    "Primary validation should be strict out-of-time holdout on the latest periods plus "
    "rolling or expanding temporal CV. StratifiedGroupKFold with groups=WEEK_NUM may be "
    "used only as a secondary diagnostic; it does not guarantee chronological "
    "train-before-validation order."
)
TEMPORAL_VALIDATION_TASK_QUESTION = (
    "Which strict temporal or out-of-time validation split best matches the competition "
    "holdout and stability metric?"
)
TEMPORAL_VALIDATION_OUTPUTS = [
    "validation_evidence.temporal_folds",
    "validation_evidence.oot_holdout",
    "validation_evidence.stratified_group_kfold_diagnostic",
]
GENERIC_VERIFICATION_PHRASES = [
    "check the relevant evidence",
    "check the dataset",
    "inspect the data",
    "analyze the data",
    "look at the data",
    "verify on real train/test files",
    "perform eda",
    "run eda",
]
CONCRETE_VERIFICATION_TERMS = set(ALLOWED_EDA_MODULES) | {
    "column",
    "columns",
    "statistic",
    "rate",
    "auc",
    "gini",
    "week_num",
    "output",
    "threshold",
    "artifact",
    "missingness",
    "dtype",
    "fold",
    "psi",
}
STRATIFIED_GROUPKFOLD_CAVEAT = (
    "This is observed in sources, but it is not sufficient as primary temporal validation "
    "because it does not guarantee chronological train-before-validation order. Use it only "
    "as a secondary diagnostic; primary validation should be out-of-time plus rolling or "
    "expanding temporal CV."
)
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
DEFAULT_MODULE_SEQUENCE = [
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
    "relationship_inferer",
    "drift_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_reverse_engineering",
]
DEFAULT_EDA_SEQUENCE = DEFAULT_MODULE_SEQUENCE
DEFAULT_HUMAN_CHECKLIST = [
    "Implement and test the official gini_stability metric.",
    "Identify WEEK_NUM/date_decision and build latest-period holdout.",
    "Compare random/grouped CV with strict out-of-time validation.",
    "Compute overall and weekly default rates.",
    "Check column suffix consistency.",
    "Compare train and test file lists.",
    "Plot application counts over time.",
    "Compare CSV and Parquet versions if both exist.",
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
For every hypothesis, provide verification_steps. Do not use vague verification text such as
"check the dataset", "inspect the data", "run EDA", or "check the relevant evidence".
Every verification step must name an EDA module, operation, inputs, outputs, success criteria,
and failure criteria.
Use structured_findings instead of raw mixed findings. Finding types are observed_in_sources,
recommendation, warning, caveat, and limitation.
If a claim is about target drift, train/test shift, adversarial validation, PSI, or time
distributions, classify it as drift. If it is about metric formula, Gini, weekly Gini,
residual std, slope penalty, prediction ties, rank/probability semantics, or thresholds,
classify it as metric. If it is about missingness, dtype, suffixes, file lists, target/id/time
roles, or sample submission, classify it as dataset_schema.
If a claim mentions StratifiedGroupKFold or GroupKFold with WEEK_NUM, state clearly that it is
observed in sources but not sufficient as primary temporal validation.

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
  "structured_findings": [],
  "scout_findings": [],
  "scout_limitations": [],
  "recommended_module_sequence": [],
  "recommended_human_checklist": [],
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
    normalized = attach_supporting_source_ids(normalized, docs)
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
    normalized["structured_findings"] = normalize_structured_findings(normalized)
    normalized["scout_findings"] = _legacy_findings_from_structured(normalized["structured_findings"])
    normalized["category_corrections"] = list(normalized.get("category_corrections") or [])
    sequence_payload = split_recommended_sequences(normalized)
    normalized.update(sequence_payload)

    hypotheses = [_normalize_hypothesis(item) for item in list(normalized.get("hypotheses") or []) if isinstance(item, dict)]
    tasks = [_normalize_task(item) for item in list(normalized.get("eda_tasks") or []) if isinstance(item, dict)]

    hypotheses.extend(_default_hypotheses(normalized, hypotheses))
    hypotheses = _dedupe_hypotheses(_assign_hypothesis_ids(hypotheses))
    correction_payload = correct_hypothesis_categories(
        {**normalized, "hypotheses": hypotheses, "eda_tasks": tasks}
    )
    hypotheses = correction_payload["hypotheses"]
    tasks = correction_payload["eda_tasks"]
    normalized["category_corrections"] = correction_payload.get("category_corrections", [])

    tasks.extend(_default_tasks(normalized, hypotheses, tasks))
    tasks = cleanup_generic_eda_tasks(tasks, {hypothesis["id"] for hypothesis in hypotheses})
    _ensure_high_priority_tasks(hypotheses, tasks)
    _ensure_task_relations(hypotheses, tasks)
    tasks = ensure_task_ids(_dedupe_tasks(tasks))

    normalized["hypotheses"] = hypotheses
    normalized["eda_tasks"] = tasks
    normalized = enforce_scout_validation_policy(normalized)
    normalized = relink_eda_tasks(normalized)
    normalized = expand_verification_steps(normalized)
    normalized = enforce_stratified_groupkfold_caveat(normalized)
    normalized = _ensure_source_attachment_limitations(normalized)
    normalized["structured_findings"] = normalize_structured_findings(normalized)
    normalized["scout_findings"] = _legacy_findings_from_structured(normalized["structured_findings"])
    normalized["eda_tasks"] = ensure_task_ids(cleanup_generic_eda_tasks(normalized["eda_tasks"], {hypothesis["id"] for hypothesis in normalized["hypotheses"]}))
    return ResearchHypothesesPayload.model_validate(normalized).model_dump(mode="json")


def ensure_task_ids(tasks: list[dict], existing_ids: set[str] | None = None) -> list[dict]:
    used = set(existing_ids or set())
    counters: Counter[str] = Counter()
    normalized_tasks: list[dict] = []
    for task in tasks:
        item = dict(task)
        module = str(item.get("module") or "table_profiler").strip()
        short = MODULE_SHORT_NAMES.get(module, _slug(module or "task"))
        related_ids = _string_list(item.get("related_hypothesis_ids"))
        raw_id = _normalize_identifier(item.get("id"))
        if not raw_id:
            if related_ids:
                raw_id = f"eda_{short}_{_normalize_identifier(related_ids[0])}"
            else:
                counters[short] += 1
                raw_id = f"eda_{short}_{counters[short]:03d}"
        elif raw_id.startswith("eda_table_profiler"):
            raw_id = raw_id.replace("eda_table_profiler", "eda_profile", 1)

        candidate = raw_id
        suffix = 2
        while candidate in used:
            candidate = f"{raw_id}_{suffix}"
            suffix += 1
        item["id"] = candidate
        used.add(candidate)
        normalized_tasks.append(item)
    return normalized_tasks


def cleanup_generic_eda_tasks(
    tasks: list[dict],
    hypothesis_ids: set[str] | None = None,
) -> list[dict]:
    cleaned: list[dict] = []
    removed_generic = False
    useful_profile_exists = False
    global_profile_exists = False
    known_ids = hypothesis_ids or set()

    for task in tasks:
        item = dict(task)
        module = str(item.get("module") or "")
        question = str(item.get("question") or "")
        related_ids = _string_list(item.get("related_hypothesis_ids"))
        is_profile = module == "table_profiler"
        is_generic = is_profile and GENERIC_TABLE_PROFILE_PHRASE in question.strip().lower()
        is_global_profile = is_profile and not related_ids and "row counts" in question.lower()
        if is_profile and related_ids and not is_generic:
            useful_profile_exists = True
        if is_global_profile:
            if global_profile_exists:
                continue
            global_profile_exists = True
        if is_generic and not related_ids:
            removed_generic = True
            continue
        cleaned.append(item)

    if removed_generic and not useful_profile_exists and not global_profile_exists:
        schema_ids = [item for item in ("schema_001", "schema_004") if not known_ids or item in known_ids]
        if not schema_ids and known_ids:
            schema_ids = sorted(item for item in known_ids if item.startswith("schema_"))[:2]
        if not schema_ids:
            schema_ids = ["schema_001"]
        cleaned.append(
            {
                "id": "eda_profile_global",
                "priority": "P0",
                "module": "table_profiler",
                "question": (
                    "What are the row counts, column types, missingness, cardinality, "
                    "and basic distributions for each train/test table?"
                ),
                "rationale": (
                    "Global table profiling is required before validation, leakage checks, "
                    "drift analysis, and baseline modeling."
                ),
                "required_inputs": ["file_inventory.files", "inferred_schema.tables"],
                "expected_outputs": [
                    "table_profiles",
                    "table_profiles.missingness",
                    "table_profiles.cardinality",
                    "table_profiles.dtypes",
                ],
                "related_hypothesis_ids": schema_ids,
                "blocking": True,
            }
        )
    return cleaned


def enforce_scout_validation_policy(payload: dict) -> dict:
    normalized = dict(payload)
    if not _is_temporal_scout_payload(normalized):
        return normalized

    hypotheses = [dict(item) for item in list(normalized.get("hypotheses") or [])]
    tasks = [dict(item) for item in list(normalized.get("eda_tasks") or [])]
    validation_hypotheses = [
        item for item in hypotheses if item.get("category") == "validation" and item.get("priority") == "P0"
    ]
    target = validation_hypotheses[0] if validation_hypotheses else None
    if target is None:
        target = _temporal_validation_hypothesis()
        hypotheses.insert(0, target)
    elif _unsafe_or_missing_temporal_policy(target):
        target.update(_temporal_validation_hypothesis(hypothesis_id=str(target.get("id") or "val_001")))

    target_id = str(target.get("id") or "val_001")
    for task in tasks:
        if task.get("module") == "validation_analyzer" and target_id in _string_list(task.get("related_hypothesis_ids")):
            _rewrite_temporal_validation_task(task)
    if not any(
        task.get("module") == "validation_analyzer"
        and target_id in _string_list(task.get("related_hypothesis_ids"))
        for task in tasks
    ):
        tasks.append(
            _task(
                "eda_val_001",
                "P0",
                "validation_analyzer",
                TEMPORAL_VALIDATION_TASK_QUESTION,
                "Temporal validation is required to avoid training on future periods.",
                ["inferred_schema.global_roles.candidate_time_columns"],
                TEMPORAL_VALIDATION_OUTPUTS,
                [target_id],
                True,
            )
        )

    normalized["hypotheses"] = hypotheses
    normalized["eda_tasks"] = tasks
    normalized["scout_limitations"] = _unique_strings(
        [
            *list(normalized.get("scout_limitations") or []),
            "Validation policy enforced: strict temporal validation is primary for stability competitions.",
        ]
    )
    return normalized


def attach_supporting_source_ids(
    payload: dict,
    retrieved_documents: list[dict],
    max_sources_per_hypothesis: int = 3,
) -> dict:
    normalized = dict(payload)
    docs = [_normalize_retrieved_doc(doc, index) for index, doc in enumerate(retrieved_documents)]
    if not docs:
        return normalized

    hypotheses = [dict(item) for item in list(normalized.get("hypotheses") or [])]
    for hypothesis in hypotheses:
        matches = _rank_source_matches(hypothesis, docs)
        source_ids = [doc["id"] for _, doc in matches[:max_sources_per_hypothesis]]
        if source_ids:
            hypothesis["supporting_source_ids"] = _unique_strings(
                [*list(hypothesis.get("supporting_source_ids") or []), *source_ids]
            )[:max_sources_per_hypothesis]
            provenance = list(hypothesis.get("provenance") or [])
            for _, doc in matches[:max_sources_per_hypothesis]:
                source = str(doc.get("source") or "")
                if source in VALID_PROVENANCE:
                    provenance.append(source)
            hypothesis["provenance"] = _unique_strings(provenance)
        else:
            hypothesis["provenance"] = _unique_strings([*list(hypothesis.get("provenance") or []), "heuristic"])

    source_ratio = (
        sum(1 for item in hypotheses if item.get("supporting_source_ids")) / len(hypotheses)
        if hypotheses
        else 1.0
    )
    limitations = list(normalized.get("scout_limitations") or [])
    p0_without_sources = [
        item["id"]
        for item in hypotheses
        if item.get("priority") == "P0" and not item.get("supporting_source_ids")
    ]
    if source_ratio < 0.5:
        limitations.append("Fewer than 50% of hypotheses have supporting_source_ids from retrieved documents.")
    if p0_without_sources:
        limitations.append("P0 hypotheses without supporting sources: " + ", ".join(p0_without_sources))
    normalized["hypotheses"] = hypotheses
    normalized["scout_limitations"] = _unique_strings(limitations)
    return normalized


def split_recommended_sequences(payload: dict) -> dict[str, list[str]]:
    existing_modules = _string_list(payload.get("recommended_module_sequence"))
    human = _string_list(payload.get("recommended_human_checklist"))
    legacy = _string_list(payload.get("recommended_eda_sequence"))

    modules = [item for item in existing_modules if item in ALLOWED_EDA_MODULES]
    for item in legacy:
        stripped = item.strip()
        if stripped in ALLOWED_EDA_MODULES:
            modules.append(stripped)
        elif stripped:
            human.append(_normalize_checklist_item(stripped))
    modules = [item for item in DEFAULT_MODULE_SEQUENCE if item in set(modules or DEFAULT_MODULE_SEQUENCE)]
    if not modules:
        modules = DEFAULT_MODULE_SEQUENCE.copy()
    if not human:
        human = DEFAULT_HUMAN_CHECKLIST.copy()
    return {
        "recommended_module_sequence": _unique_strings(modules),
        "recommended_human_checklist": _unique_strings(human),
        "recommended_eda_sequence": _unique_strings(modules),
    }


def is_generic_verification_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9_.]+", " ", text.lower()).strip()
    if not any(phrase in normalized for phrase in GENERIC_VERIFICATION_PHRASES):
        return False
    return not any(term in normalized for term in CONCRETE_VERIFICATION_TERMS)


def expand_verification_steps(payload: dict) -> dict:
    normalized = dict(payload)
    tasks = [dict(task) for task in list(normalized.get("eda_tasks") or [])]
    task_ids_by_module: dict[str, list[str]] = {}
    for task in tasks:
        task_ids_by_module.setdefault(str(task.get("module")), []).append(str(task.get("id")))

    hypotheses: list[dict[str, Any]] = []
    for hypothesis in list(normalized.get("hypotheses") or []):
        item = dict(hypothesis)
        existing_steps = [
            _normalize_verification_step(step)
            for step in list(item.get("verification_steps") or [])
            if isinstance(step, dict)
        ]
        steps = existing_steps.copy()
        if item.get("priority") in {"P0", "P1"} or not steps:
            for step in _verification_templates_for_hypothesis(item, task_ids_by_module):
                if not any(existing["id"] == step["id"] for existing in steps):
                    steps.append(step)
        if item.get("priority") in {"P0", "P1"} and not steps:
            steps.append(_generic_structured_step(item, task_ids_by_module))
        item["verification_steps"] = ensure_verification_step_ids(steps)
        if not item.get("how_to_verify") or any(is_generic_verification_text(text) for text in _string_list(item.get("how_to_verify"))):
            item["how_to_verify"] = _how_to_verify_from_steps(item["verification_steps"])
        hypotheses.append(item)
    normalized["hypotheses"] = hypotheses
    return normalized


def correct_hypothesis_categories(payload: dict) -> dict:
    normalized = dict(payload)
    hypotheses = [dict(item) for item in list(normalized.get("hypotheses") or [])]
    tasks = [dict(item) for item in list(normalized.get("eda_tasks") or [])]
    corrections = list(normalized.get("category_corrections") or [])
    used_ids = {str(item.get("id")) for item in hypotheses if item.get("id")}
    id_map: dict[str, str] = {}

    for hypothesis in hypotheses:
        old_category = str(hypothesis.get("category") or "")
        new_category, reason = _corrected_category_for_hypothesis(hypothesis)
        if new_category == old_category:
            continue
        old_id = str(hypothesis.get("id") or "")
        new_id = _reassign_hypothesis_id(old_id, new_category, used_ids)
        used_ids.discard(old_id)
        used_ids.add(new_id)
        hypothesis["category"] = new_category
        hypothesis["id"] = new_id
        id_map[old_id] = new_id
        corrections.append(
            {
                "old_id": old_id,
                "new_id": new_id,
                "old_category": old_category,
                "new_category": new_category,
                "reason": reason,
            }
        )

    if id_map:
        for task in tasks:
            task["related_hypothesis_ids"] = [
                id_map.get(hypothesis_id, hypothesis_id)
                for hypothesis_id in _string_list(task.get("related_hypothesis_ids"))
            ]
        for hypothesis in hypotheses:
            for step in list(hypothesis.get("verification_steps") or []):
                if isinstance(step, dict):
                    step["related_hypothesis_ids"] = [
                        id_map.get(hypothesis_id, hypothesis_id)
                        for hypothesis_id in _string_list(step.get("related_hypothesis_ids"))
                    ]

    normalized["hypotheses"] = hypotheses
    normalized["eda_tasks"] = tasks
    normalized["category_corrections"] = corrections
    return normalized


def enforce_stratified_groupkfold_caveat(payload: dict) -> dict:
    normalized = enforce_scout_validation_policy(payload)
    if not _is_temporal_scout_payload(normalized):
        return normalized

    references_group_kfold = _payload_mentions_groupkfold(normalized)
    findings = normalize_structured_findings(normalized)
    for finding in findings:
        text = f"{finding.get('claim', '')} {finding.get('caveat', '')}".lower()
        if _mentions_groupkfold(text):
            if finding.get("finding_type") == "recommendation" and "not sufficient as primary" not in text:
                finding["finding_type"] = "observed_in_sources"
            finding["caveat"] = STRATIFIED_GROUPKFOLD_CAVEAT
            finding["confidence"] = "high"

    if references_group_kfold and not any(finding.get("id") == "caveat_stratified_groupkfold_temporal" for finding in findings):
        findings.append(
            {
                "id": "caveat_stratified_groupkfold_temporal",
                "finding_type": "caveat",
                "claim": "StratifiedGroupKFold with WEEK_NUM can keep weeks disjoint but does not enforce chronological order.",
                "implication": "It must not be used as the sole primary validation strategy for stability competitions.",
                "caveat": "Primary validation should remain strict out-of-time holdout plus rolling or expanding temporal CV.",
                "provenance": ["heuristic", "not_verified_on_data"],
                "supporting_source_ids": [],
                "related_hypothesis_ids": [
                    item["id"]
                    for item in list(normalized.get("hypotheses") or [])
                    if isinstance(item, dict) and item.get("category") == "validation"
                ][:1],
                "confidence": "high",
            }
        )
    normalized["structured_findings"] = findings
    normalized["scout_findings"] = _legacy_findings_from_structured(findings)
    return normalized


def relink_eda_tasks(payload: dict) -> dict:
    normalized = dict(payload)
    hypotheses = [dict(item) for item in list(normalized.get("hypotheses") or [])]
    tasks = ensure_task_ids([dict(item) for item in list(normalized.get("eda_tasks") or [])])
    tasks_by_module = {str(task.get("module")): task for task in tasks}
    category_module = {
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
    for module in ALLOWED_EDA_MODULES:
        if module not in tasks_by_module and module in DEFAULT_MODULE_SEQUENCE:
            task = _module_task_template(module, hypotheses)
            tasks.append(task)
            tasks_by_module[module] = task

    for task in tasks:
        module = str(task.get("module"))
        if module in GLOBAL_TASK_MODULES:
            continue
        allowed_categories = {
            category for category, mapped_module in category_module.items() if mapped_module == module
        }
        task["related_hypothesis_ids"] = [
            hypothesis["id"]
            for hypothesis in hypotheses
            if hypothesis.get("category") in allowed_categories
        ]

    for task in tasks:
        if task.get("module") in {"schema_inferer", "table_profiler"}:
            schema_ids = [hypothesis["id"] for hypothesis in hypotheses if hypothesis.get("category") == "dataset_schema"]
            task["related_hypothesis_ids"] = schema_ids

    tasks = ensure_task_ids(_dedupe_tasks(tasks))
    task_ids_by_module: dict[str, list[str]] = {}
    for task in tasks:
        task_ids_by_module.setdefault(str(task.get("module")), []).append(str(task.get("id")))
    for hypothesis in hypotheses:
        for step in list(hypothesis.get("verification_steps") or []):
            if isinstance(step, dict):
                related = task_ids_by_module.get(str(step.get("module")), [])
                if related:
                    step["related_task_ids"] = _unique_strings([*list(step.get("related_task_ids") or []), *related])

    normalized["hypotheses"] = hypotheses
    normalized["eda_tasks"] = tasks
    return normalized


def _temporal_validation_hypothesis(hypothesis_id: str = "val_001") -> dict[str, Any]:
    return {
        "id": hypothesis_id,
        "category": "validation",
        "priority": "P0",
        "claim": TEMPORAL_VALIDATION_CLAIM,
        "why_it_matters": (
            "Stability metrics punish degradation across time. A split that mixes future "
            "periods into training can overestimate performance and select unstable models."
        ),
        "how_to_verify": [
            "Identify reliable time columns such as WEEK_NUM, date_decision, date, month, or period.",
            "Create latest-period out-of-time holdout.",
            "Create rolling or expanding temporal folds where training periods always precede validation periods.",
            "Optionally compare against StratifiedGroupKFold by WEEK_NUM as a diagnostic only.",
            "Compare random/grouped CV estimates with out-of-time estimates.",
        ],
        "expected_evidence_keys": [
            "inferred_schema.global_roles.candidate_time_columns",
            "validation_evidence.recommended_validation",
            "validation_evidence.train_test_time_relation",
            "validation_evidence.target_by_period",
            "baseline_evidence.per_period_metric",
        ],
        "failure_condition": "No reliable temporal column exists and the official metric does not depend on temporal stability.",
        "success_condition": (
            "A reliable temporal column exists or the metric requires stability, and temporal folds "
            "can be created without training on future periods."
        ),
        "provenance": ["kaggle", "heuristic", "not_verified_on_data"],
        "supporting_source_ids": [],
        "confidence": "high",
        "status": "needs_eda",
    }


def ensure_verification_step_ids(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    result: list[dict[str, Any]] = []
    for step in steps:
        item = _normalize_verification_step(step)
        raw_id = _normalize_identifier(item.get("id")) or f"verify_{item['module']}_{_slug(item['operation'])}"
        candidate = raw_id
        suffix = 2
        while candidate in used:
            candidate = f"{raw_id}_{suffix}"
            suffix += 1
        item["id"] = candidate
        used.add(candidate)
        result.append(item)
    return result


def _verification_templates_for_hypothesis(
    hypothesis: dict[str, Any],
    task_ids_by_module: dict[str, list[str]],
) -> list[dict[str, Any]]:
    category = str(hypothesis.get("category") or "")
    text = _hypothesis_text(hypothesis)
    hypothesis_id = str(hypothesis.get("id") or "hyp")
    steps: list[dict[str, Any]] = []
    if category == "validation":
        if any(term in text for term in ("temporal", "out-of-time", "week_num", "rolling", "expanding", "stability")):
            steps.extend(
                [
                    _step(hypothesis_id, "schema_inferer", "identify_time_columns", "Which columns can define chronological validation splits?", ["file_inventory.files", "table_profiles", "train_base"], ["inferred_schema.global_roles.candidate_time_columns"], task_ids_by_module),
                    _step(hypothesis_id, "validation_analyzer", "build_oot_holdout", "Which latest-period out-of-time holdout best matches the competition holdout?", ["train_base", "target", "candidate_time_columns"], ["validation_evidence.oot_holdout"], task_ids_by_module),
                    _step(hypothesis_id, "validation_analyzer", "build_rolling_or_expanding_folds", "Can folds be built so training periods always precede validation periods?", ["train_base", "target", "candidate_time_columns"], ["validation_evidence.temporal_folds"], task_ids_by_module),
                    _step(hypothesis_id, "baseline_runner", "compare_random_vs_oot", "How different are random or grouped CV estimates from strict out-of-time validation?", ["baseline_predictions", "validation_folds"], ["validation_evidence.random_vs_oot", "baseline_evidence.per_period_metric"], task_ids_by_module),
                ]
            )
        else:
            steps.append(_step(hypothesis_id, "validation_analyzer", "recommend_validation_split", "Which validation split best matches the competition holdout?", ["inferred_schema.global_roles"], ["validation_evidence.recommended_validation"], task_ids_by_module))
    elif category == "leakage":
        for operation, question, output in (
            ("check_id_overlap", "Do train and test identifiers overlap in unsafe ways?", "leakage_evidence.id_overlap"),
            ("find_target_like_columns", "Which columns look like target proxies or post-outcome fields?", "leakage_evidence.target_like_columns"),
            ("check_future_rows", "Do secondary rows contain future information relative to prediction time?", "leakage_evidence.future_rows"),
            ("check_group_leakage", "Can group membership leak labels across validation folds?", "leakage_evidence.group_leakage"),
        ):
            steps.append(_step(hypothesis_id, "leakage_checker", operation, question, ["inferred_schema.global_roles"], [output], task_ids_by_module))
    elif category == "metric":
        steps.extend(
            [
                _step(hypothesis_id, "metric_analyzer", "implement_metric", "Can the official metric be reproduced locally on known examples?", ["target", "prediction"], ["metric_evidence.local_metric"], task_ids_by_module),
                _step(hypothesis_id, "metric_analyzer", "verify_probability_or_rank_inputs", "Does the metric require probabilities, ranks, or hard labels?", ["metric_definition", "sample_submission"], ["metric_evidence.requires_probabilities", "metric_evidence.rank_based"], task_ids_by_module),
            ]
        )
        if any(term in text for term in ("gini", "stability", "weekly", "slope", "residual", "std")):
            steps.append(_step(hypothesis_id, "metric_analyzer", "compute_weekly_metric_components", "Which weekly metric components drive the stability score?", ["target", "prediction", "candidate_time_columns"], ["metric_evidence.weekly_components"], task_ids_by_module))
        if "tie" in text or "ties" in text:
            steps.append(_step(hypothesis_id, "metric_analyzer", "tie_sensitivity_analysis", "How sensitive is the metric to tied predictions?", ["prediction_scores"], ["metric_evidence.tie_sensitivity"], task_ids_by_module))
        if any(term in text for term in ("residual", "std", "stability")):
            steps.append(_step(hypothesis_id, "baseline_runner", "compare_metric_components", "How do baseline models trade off mean score and stability penalty?", ["baseline_predictions", "temporal_folds"], ["baseline_evidence.metric_components"], task_ids_by_module))
    elif category == "dataset_schema":
        steps.extend(
            [
                _step(hypothesis_id, "file_inventory", "list_files", "Which train, test, sample submission, CSV, and Parquet files are available?", [], ["file_inventory.files"], task_ids_by_module),
                _step(hypothesis_id, "schema_inferer", "infer_roles", "Which columns are target, ID, time, group, and submission fields?", ["file_inventory.files"], ["inferred_schema.global_roles"], task_ids_by_module),
                _step(hypothesis_id, "table_profiler", "profile_missingness", "Which columns have high missingness by table and split?", ["train_tables", "test_tables"], ["table_profiles.missingness"], task_ids_by_module),
                _step(hypothesis_id, "table_profiler", "profile_dtypes", "Which dtypes and cardinalities appear in each table?", ["train_tables", "test_tables"], ["table_profiles.dtypes"], task_ids_by_module),
            ]
        )
        if "suffix" in text:
            steps.append(_step(hypothesis_id, "schema_inferer", "check_suffix_dtype_consistency", "Are column suffixes consistent with dtypes and semantic roles?", ["inferred_schema.global_roles", "table_profiles.dtypes"], ["schema_evidence.suffix_dtype_consistency"], task_ids_by_module))
    elif category == "drift":
        if any(term in text for term in ("default rate", "target drift", "period", "week", "time")):
            steps.append(_step(hypothesis_id, "validation_analyzer", "compute_target_by_period", "How does target rate change across periods?", ["target", "candidate_time_columns"], ["validation_evidence.target_by_period"], task_ids_by_module))
        for operation, question, output in (
            ("run_adversarial_validation", "How separable are train and test rows using schema-safe features?", "drift_evidence.adversarial_auc"),
            ("compute_numeric_psi", "Which numeric features show train/test distribution shift?", "drift_evidence.numeric_psi"),
            ("compute_categorical_shift", "Which categorical levels shift between train and test?", "drift_evidence.categorical_shift"),
            ("compute_missingness_shift", "Which missingness patterns shift between train and test?", "drift_evidence.missingness"),
        ):
            steps.append(_step(hypothesis_id, "drift_analyzer", operation, question, ["train_tables", "test_tables"], [output], task_ids_by_module))
    elif category == "relationships":
        for operation, question, output in (
            ("detect_base_table", "Which table is the base prediction table?", "relationship_evidence.base_table"),
            ("detect_join_keys", "Which keys connect base and secondary tables?", "relationship_evidence.join_keys"),
            ("measure_join_coverage", "What join coverage and orphan rates exist across tables?", "relationship_evidence.join_coverage"),
            ("check_cutoff_feasibility", "Can secondary rows be cut off before prediction time?", "relationship_evidence.cutoff_feasibility"),
        ):
            steps.append(_step(hypothesis_id, "relationship_inferer", operation, question, ["file_inventory.files", "inferred_schema.global_roles"], [output], task_ids_by_module))
    elif category == "feature_engineering":
        for operation, question, output in (
            ("evaluate_aggregation_feasibility", "Which aggregations are feasible without leakage?", "feature_evidence.aggregation_feasibility"),
            ("evaluate_date_features", "Which date features are safe and predictive?", "feature_evidence.date_features"),
            ("evaluate_missingness_indicators", "Which missingness indicators are safe feature candidates?", "feature_evidence.missingness_indicators"),
            ("evaluate_encoding_safety", "Which encodings can be built inside folds safely?", "feature_evidence.encoding_safety"),
            ("measure_feature_family_lift", "Which feature families improve aligned validation?", "feature_evidence.family_lift"),
        ):
            module = "baseline_runner" if operation == "measure_feature_family_lift" else "feature_probe"
            steps.append(_step(hypothesis_id, module, operation, question, ["validated_schema", "recommended_validation"], [output], task_ids_by_module))
    elif category == "baseline":
        for operation, question, output in (
            ("train_base_table_baseline", "What is the base-table baseline score?", "baseline_evidence.base_table_metric"),
            ("train_simple_aggregation_baseline", "What lift comes from simple aggregation baselines?", "baseline_evidence.simple_aggregation_metric"),
            ("compute_per_period_metric", "How does baseline performance vary by period?", "baseline_evidence.per_period_metric"),
            ("save_feature_importance", "Which features dominate the baseline model?", "baseline_evidence.feature_importance"),
        ):
            steps.append(_step(hypothesis_id, "baseline_runner", operation, question, ["validated_schema", "recommended_validation"], [output], task_ids_by_module))
    elif category == "notebook_reverse_engineering":
        for operation, question, output in (
            ("extract_cv_strategy", "What validation strategies appear in top notebooks?", "notebook_static_analysis.cv_strategy"),
            ("extract_feature_families", "What feature families appear in top notebooks?", "notebook_static_analysis.feature_families"),
            ("extract_metric_code", "What metric code appears in top notebooks?", "notebook_static_analysis.metric_code"),
            ("extract_postprocessing", "What postprocessing appears in top notebooks?", "notebook_static_analysis.postprocessing"),
        ):
            steps.append(_step(hypothesis_id, "notebook_reverse_engineering", operation, question, ["retrieved Kaggle notebook sources"], [output], task_ids_by_module))
    return steps


def _step(
    hypothesis_id: str,
    module: str,
    operation: str,
    question: str,
    inputs: list[str],
    outputs: list[str],
    task_ids_by_module: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "id": f"verify_{hypothesis_id}_{_slug(operation)}",
        "module": module,
        "operation": operation,
        "question": question,
        "inputs": inputs,
        "outputs": outputs,
        "success_criteria": [f"{outputs[0]} is produced with usable evidence."],
        "failure_criteria": [f"{outputs[0]} cannot be produced or is inconclusive."],
        "related_task_ids": task_ids_by_module.get(module, []),
    }


def _generic_structured_step(
    hypothesis: dict[str, Any],
    task_ids_by_module: dict[str, list[str]],
) -> dict[str, Any]:
    module = _module_for_category(str(hypothesis.get("category") or "dataset_schema"))
    return _step(
        str(hypothesis.get("id") or "hyp"),
        module,
        "collect_evidence",
        "Which concrete evidence artifacts verify this hypothesis?",
        ["validated_schema"],
        hypothesis.get("expected_evidence_keys") or [f"{module}.result"],
        task_ids_by_module,
    )


def _how_to_verify_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    return [
        f"{step['operation']}: {step['question']}"
        for step in steps
    ]


def _corrected_category_for_hypothesis(hypothesis: dict[str, Any]) -> tuple[str, str]:
    current = str(hypothesis.get("category") or "feature_engineering")
    text = _hypothesis_text(hypothesis)
    explicit_feature = any(
        term in text
        for term in (
            "feature family",
            "aggregation",
            "encoding",
            "date features",
            "missingness indicators",
            "feature selection",
            "feature lift",
            "feature importance",
        )
    )
    if explicit_feature:
        return current, ""
    drift_terms = (
        "default rate changes over time",
        "target drift",
        "default rate drift",
        "week_num drift",
        "covariate shift",
        "train/test shift",
        "distribution shift",
        "adversarial validation",
        "psi",
        "period drift",
        "default rate over time",
    )
    metric_terms = (
        "residual standard deviation",
        "residual std",
        "metric penalty",
        "weekly gini",
        "slope",
        "tie",
        "ties",
        "threshold search",
        "hard class labels",
        "rank-based",
        "probability-based",
        "calibration",
    )
    schema_terms = (
        "high missing rates",
        ">95% missing",
        "missing value analysis",
        "column suffix",
        "dtype",
        "data type",
        "train/test file list",
        "csv vs parquet",
        "target/id/time columns",
        "sample submission",
    )
    if _contains_terms(text, drift_terms):
        return "drift", "Claim mentions default rate changes over time, drift, shift, PSI, or adversarial validation."
    if _contains_terms(text, metric_terms):
        return "metric", "Claim mentions metric formula, residual penalty, ties, thresholds, ranks, or probabilities."
    if _contains_terms(text, schema_terms):
        return "dataset_schema", "Claim mentions missingness, dtypes, suffixes, file lists, roles, or sample submission."
    return current, ""


def _contains_terms(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
            return True
    return False


def _reassign_hypothesis_id(old_id: str, new_category: str, used_ids: set[str]) -> str:
    prefix = CATEGORY_PREFIX.get(new_category, "hyp")
    suffix = "001"
    match = re.match(r"^[a-z]+_(\d+)$", old_id)
    if match:
        suffix = match.group(1)
    candidate = f"{prefix}_{suffix}"
    counter = int(suffix) if suffix.isdigit() else 1
    while candidate in used_ids and candidate != old_id:
        counter += 1
        candidate = f"{prefix}_{counter:03d}"
    return candidate


def _module_task_template(module: str, hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    category = {
        "file_inventory": "dataset_schema",
        "schema_inferer": "dataset_schema",
        "table_profiler": "dataset_schema",
        "relationship_inferer": "relationships",
        "validation_analyzer": "validation",
        "leakage_checker": "leakage",
        "drift_analyzer": "drift",
        "metric_analyzer": "metric",
        "baseline_runner": "baseline",
        "feature_probe": "feature_engineering",
        "notebook_reverse_engineering": "notebook_reverse_engineering",
    }.get(module)
    related = [item["id"] for item in hypotheses if item.get("category") == category]
    short = MODULE_SHORT_NAMES.get(module, module)
    return _task(
        f"eda_{short}_001",
        "P1",
        module,
        f"Which evidence should {module} produce for Scout hypotheses?",
        f"{module} evidence is required by structured verification steps.",
        [],
        [f"{module}.result"],
        related,
        module in {"file_inventory", "schema_inferer", "table_profiler", "validation_analyzer", "leakage_checker"},
    )


def _module_for_category(category: str) -> str:
    return {
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
    }.get(category, "table_profiler")


def _hypothesis_text(hypothesis: dict[str, Any]) -> str:
    steps = list(hypothesis.get("verification_steps") or [])
    step_text = " ".join(
        " ".join(
            str(step.get(key) or "")
            for key in ("operation", "question", "outputs")
        )
        for step in steps
        if isinstance(step, dict)
    )
    return " ".join(
        [
            str(hypothesis.get("claim") or ""),
            str(hypothesis.get("why_it_matters") or ""),
            " ".join(_string_list(hypothesis.get("how_to_verify"))),
            " ".join(_string_list(hypothesis.get("expected_evidence_keys"))),
            step_text,
        ]
    ).lower()


def _payload_mentions_groupkfold(payload: dict[str, Any]) -> bool:
    text = json.dumps(
        {
            "structured_findings": payload.get("structured_findings"),
            "scout_findings": payload.get("scout_findings"),
            "hypotheses": payload.get("hypotheses"),
            "eda_tasks": payload.get("eda_tasks"),
        },
        ensure_ascii=False,
    ).lower()
    return _mentions_groupkfold(text)


def _mentions_groupkfold(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "stratifiedgroupkfold",
            "groupkfold",
            "groups=week_num",
            "week_num as group",
            "week_num as group",
        )
    )


def _rewrite_temporal_validation_task(task: dict[str, Any]) -> None:
    task["id"] = str(task.get("id") or "eda_val_001")
    task["priority"] = "P0"
    task["question"] = TEMPORAL_VALIDATION_TASK_QUESTION
    task["rationale"] = "Temporal validation is required to match stability-sensitive holdout behavior."
    task["expected_outputs"] = _unique_strings(
        [*list(task.get("expected_outputs") or []), *TEMPORAL_VALIDATION_OUTPUTS]
    )
    task["blocking"] = True


def _is_temporal_scout_payload(payload: dict[str, Any]) -> bool:
    metric = _as_dict(payload.get("metric"))
    text_parts = [
        str(metric.get("name") or ""),
        str(payload.get("competition_desc") or ""),
        json.dumps(payload.get("source_summary") or {}, ensure_ascii=False),
        json.dumps(payload.get("source_quality_summary") or {}, ensure_ascii=False),
        json.dumps(payload.get("hypotheses") or [], ensure_ascii=False),
    ]
    text = " ".join(text_parts).lower()
    return any(term in text for term in ("stability", "week_num", "week", "time", "temporal", "date", "period"))


def _unsafe_or_missing_temporal_policy(hypothesis: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(hypothesis.get("claim") or ""),
            str(hypothesis.get("why_it_matters") or ""),
            " ".join(_string_list(hypothesis.get("how_to_verify"))),
        ]
    ).lower()
    has_primary_temporal = (
        ("out-of-time" in text or "oot" in text or "strict temporal" in text)
        and ("rolling" in text or "expanding" in text or "temporal" in text)
    )
    unsafe_group_primary = any(
        phrase in text
        for phrase in (
            "stratifiedgroupkfold is the primary",
            "stratifiedgroupkfold with groups=week_num ensures",
            "groupkfold by week is sufficient",
            "groupkfold is sufficient",
            "stratifiedgroupkfold is sufficient",
        )
    )
    return unsafe_group_primary or not has_primary_temporal


def _normalize_retrieved_doc(doc: dict[str, Any], index: int) -> dict[str, Any]:
    item = _as_dict(doc)
    metadata = _as_dict(item.get("metadata"))
    doc_id = str(item.get("id") or "").strip()
    if not doc_id:
        title_hash = hashlib.sha1(
            f"{item.get('source')}-{item.get('title')}-{index}".encode("utf-8")
        ).hexdigest()[:10]
        doc_id = f"{item.get('source') or 'source'}-{title_hash}"
    text = " ".join(
        str(value or "")
        for value in (
            item.get("title"),
            item.get("summary"),
            item.get("content"),
            metadata.get("specificity"),
            metadata.get("evidence_type"),
            item.get("source_quality"),
            item.get("specificity"),
            item.get("evidence_type"),
        )
    ).lower()
    return {
        **item,
        "id": doc_id,
        "source": str(item.get("source") or metadata.get("source") or "unknown"),
        "_match_text": text,
    }


def _rank_source_matches(hypothesis: dict[str, Any], docs: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    keywords = _keywords_for_hypothesis(hypothesis)
    preferred_sources = _preferred_sources_for_hypothesis(hypothesis)
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        text = str(doc.get("_match_text") or "")
        keyword_score = sum(1 for keyword in keywords if keyword.lower() in text)
        if keyword_score <= 0:
            continue
        source = str(doc.get("source") or "")
        source_score = _source_preference_score(source, preferred_sources, text)
        scored.append((keyword_score * 10 + source_score, doc))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return scored


def _keywords_for_hypothesis(hypothesis: dict[str, Any]) -> list[str]:
    category = str(hypothesis.get("category") or "")
    claim_text = str(hypothesis.get("claim") or "").lower()
    if category == "metric":
        return ["metric", "gini", "auc", "stability", "weekly", "slope", "residual", "evaluation"]
    if category == "validation":
        return ["week_num", "validation", "fold", "cv", "out-of-time", "time", "temporal", "groupkfold", "stratifiedgroupkfold"]
    if category == "leakage":
        return ["leakage", "future", "date", "target encoding", "id overlap", "case_id", "fold"]
    if category == "feature_engineering":
        return ["feature", "aggregation", "aggregate", "catboost", "lightgbm", "encoding", "suffix", "parquet"]
    if category == "baseline":
        return ["baseline", "lightgbm", "catboost", "xgboost", "model"]
    if category == "drift":
        return ["drift", "shift", "temporal", "period", "week", "covid", "train test"]
    if category == "notebook_reverse_engineering" or "notebook" in claim_text:
        return ["notebook", "kaggle", "code", "cv", "feature", "model"]
    if category == "dataset_schema":
        return ["schema", "table", "column", "file", "parquet", "csv", "week_num", "case_id"]
    return ["kaggle", "validation", "feature", "model", "metric"]


def _preferred_sources_for_hypothesis(hypothesis: dict[str, Any]) -> list[str]:
    category = str(hypothesis.get("category") or "")
    if category == "notebook_reverse_engineering":
        return ["kaggle"]
    if category in {"feature_engineering", "baseline"}:
        return ["kaggle", "github"]
    if category == "drift":
        return ["kaggle", "arxiv", "huggingface_papers"]
    if category == "metric":
        return ["kaggle", "arxiv", "huggingface_papers"]
    return ["kaggle", "github", "arxiv", "huggingface_papers"]


def _source_preference_score(source: str, preferred_sources: list[str], text: str) -> float:
    try:
        base = len(preferred_sources) - preferred_sources.index(source)
    except ValueError:
        base = 0
    specificity = 2 if any(term in text for term in ("competition", "week_num", "home credit", "kaggle")) else 0
    return base + specificity


def _normalize_checklist_item(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return value if value.endswith(".") else f"{value}."


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "task"


def _raw_contract_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    hypotheses = [item for item in list(payload.get("hypotheses") or []) if isinstance(item, dict)]
    tasks = [item for item in list(payload.get("eda_tasks") or []) if isinstance(item, dict)]
    hypothesis_ids = [_normalize_identifier(item.get("id")) for item in hypotheses]
    task_ids = [_normalize_identifier(item.get("id")) for item in tasks]
    if any(not item for item in hypothesis_ids):
        errors.append("hypothesis IDs must not be empty")
    if any(not item for item in task_ids):
        errors.append("EDA task IDs must not be empty")
    duplicate_hypothesis_ids = [item for item, count in Counter(hypothesis_ids).items() if item and count > 1]
    duplicate_task_ids = [item for item, count in Counter(task_ids).items() if item and count > 1]
    if duplicate_hypothesis_ids:
        errors.append("payload must not contain duplicate hypothesis IDs: " + ", ".join(duplicate_hypothesis_ids))
    if duplicate_task_ids:
        errors.append("payload must not contain duplicate task IDs: " + ", ".join(duplicate_task_ids))
    for task in tasks:
        question = str(task.get("question") or "").lower()
        if GENERIC_TABLE_PROFILE_PHRASE in question:
            errors.append(f"EDA task {task.get('id') or '<empty>'} has generic task question")
    return errors


def _has_safe_temporal_validation(hypotheses: list[ResearchHypothesis]) -> bool:
    for hypothesis in hypotheses:
        if hypothesis.category != "validation" or hypothesis.priority != "P0":
            continue
        text = hypothesis.claim.lower()
        has_oot = "out-of-time" in text or "oot" in text
        has_temporal_cv = "rolling" in text or "expanding" in text or "strict temporal" in text
        unsafe = any(
            phrase in text
            for phrase in (
                "stratifiedgroupkfold is the primary",
                "stratifiedgroupkfold with groups=week_num ensures",
                "groupkfold by week is sufficient",
                "stratifiedgroupkfold is sufficient",
            )
        )
        if has_oot and has_temporal_cv and not unsafe:
            return True
    return False


def _is_supported_supervised_task(task_type: str) -> bool:
    text = task_type.lower()
    return any(term in text for term in ("classification", "regression", "supervised", "binary", "multiclass"))


def _is_dataset_dependent(hypothesis: ResearchHypothesis) -> bool:
    text = " ".join(
        [
            hypothesis.claim,
            hypothesis.why_it_matters,
            " ".join(hypothesis.how_to_verify),
        ]
    ).lower()
    return any(
        term in text
        for term in (
            "dataset",
            "train",
            "test",
            "column",
            "feature",
            "leakage",
            "drift",
            "baseline",
            "validation",
            "schema",
            "table",
        )
    )


def _retrieved_sources_available(model: ResearchHypothesesPayload) -> bool:
    source_summary = model.source_summary or {}
    source_quality_summary = model.source_quality_summary or {}
    if any(int(value or 0) > 0 for value in source_summary.values() if isinstance(value, int)):
        return True
    top_sources = source_quality_summary.get("top_sources") if isinstance(source_quality_summary, dict) else None
    return bool(top_sources)


def _ensure_source_attachment_limitations(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    hypotheses = [item for item in list(normalized.get("hypotheses") or []) if isinstance(item, dict)]
    source_summary = _as_dict(normalized.get("source_summary"))
    sources_available = any(
        isinstance(value, int) and value > 0
        for value in source_summary.values()
    )
    if not sources_available or not hypotheses:
        return normalized
    sourced = [item for item in hypotheses if item.get("supporting_source_ids")]
    if len(sourced) / len(hypotheses) >= 0.5:
        return normalized
    normalized["scout_limitations"] = _unique_strings(
        [
            *list(normalized.get("scout_limitations") or []),
            "Fewer than 50% of hypotheses have supporting_source_ids from retrieved documents.",
        ]
    )
    return normalized


def _valid_verification_steps(steps: list[VerificationStep]) -> list[VerificationStep]:
    return [
        step
        for step in steps
        if step.module in ALLOWED_EDA_MODULES
        and bool(step.operation)
        and bool(step.question)
        and bool(step.outputs)
        and (bool(step.success_criteria) or bool(step.failure_criteria))
    ]


def validate_research_hypotheses(payload: dict) -> None:
    raw_errors = _raw_contract_errors(payload)
    if raw_errors:
        raise ValueError("Research Scout validation failed:\n- " + "\n- ".join(raw_errors))

    try:
        model = ResearchHypothesesPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Research Scout payload failed schema validation: {exc}") from exc

    errors: list[str] = []
    hypotheses = model.hypotheses
    tasks = model.eda_tasks
    metric_name = str(model.metric.get("name") or "").lower()
    hypothesis_ids = [item.id for item in hypotheses]
    task_ids = [item.id for item in tasks]
    known_hypothesis_ids = set(hypothesis_ids)
    categories = {item.category for item in hypotheses}

    if len(hypotheses) < 8:
        errors.append("payload must contain at least 8 hypotheses")
    if not any(item.category == "validation" and item.priority == "P0" for item in hypotheses):
        errors.append("payload must contain at least one P0 validation hypothesis")
    if not any(item.category == "leakage" and item.priority == "P0" for item in hypotheses):
        errors.append("payload must contain at least one P0 leakage hypothesis")
    if "metric" not in categories:
        errors.append("payload must contain at least one metric hypothesis")
    if "dataset_schema" not in categories:
        errors.append("payload must contain at least one schema hypothesis")
    if _is_supported_supervised_task(model.task_type) and "baseline" not in categories:
        errors.append("payload must contain at least one baseline hypothesis for supervised tasks")
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        errors.append("payload must not contain duplicate hypothesis IDs")
    if len(task_ids) != len(set(task_ids)):
        errors.append("payload must not contain duplicate task IDs")
    for module in ("file_inventory", "schema_inferer", "table_profiler", "validation_analyzer", "leakage_checker", "metric_analyzer"):
        if not any(task.module == module for task in tasks):
            errors.append(f"payload must contain at least one {module} task")
    if "stability" in metric_name and not _has_safe_temporal_validation(hypotheses):
        errors.append("stability metrics require P0 validation with out-of-time plus rolling/expanding temporal CV")

    for item in hypotheses:
        missing = []
        if not item.id:
            missing.append("id")
        if not item.claim:
            missing.append("claim")
        if not item.why_it_matters:
            missing.append("why_it_matters")
        if not item.how_to_verify:
            missing.append("how_to_verify")
        if not item.expected_evidence_keys:
            missing.append("expected_evidence_keys")
        if not item.provenance:
            missing.append("provenance")
        if not item.confidence:
            missing.append("confidence")
        if missing:
            errors.append(f"hypothesis {item.id} missing required fields: {', '.join(missing)}")
        if _is_dataset_dependent(item) and "not_verified_on_data" not in item.provenance:
            errors.append(f"hypothesis {item.id} dataset-dependent claim must include not_verified_on_data")
        valid_steps = _valid_verification_steps(item.verification_steps)
        if item.priority in {"P0", "P1"} and not valid_steps:
            errors.append(f"hypothesis {item.id} must include at least one structured verification_step")
        if any(is_generic_verification_text(text) for text in item.how_to_verify) and not valid_steps:
            errors.append(f"hypothesis {item.id} has generic how_to_verify without structured verification_steps")
        for step in item.verification_steps:
            step_missing = []
            if step.module not in ALLOWED_EDA_MODULES:
                step_missing.append("valid module")
            if not step.operation:
                step_missing.append("operation")
            if not step.question:
                step_missing.append("question")
            if not step.outputs:
                step_missing.append("outputs")
            if not step.success_criteria and not step.failure_criteria:
                step_missing.append("success_criteria or failure_criteria")
            if step_missing:
                errors.append(f"verification step {step.id} missing required fields: {', '.join(step_missing)}")
    for task in tasks:
        missing = []
        if not task.id:
            missing.append("id")
        if not task.module:
            missing.append("module")
        if not task.question:
            missing.append("question")
        if not task.rationale:
            missing.append("rationale")
        if not task.expected_outputs:
            missing.append("expected_outputs")
        if missing:
            errors.append(f"EDA task {task.id} missing required fields: {', '.join(missing)}")
        if task.module not in ALLOWED_EDA_MODULES:
            errors.append(f"EDA task {task.id} has invalid module {task.module}")
        if GENERIC_TABLE_PROFILE_PHRASE in task.question.lower():
            errors.append(f"EDA task {task.id} has generic task question")
        if task.module not in GLOBAL_TASK_MODULES and not task.related_hypothesis_ids:
            errors.append(f"EDA task {task.id} missing related_hypothesis_ids")
        for hypothesis_id in task.related_hypothesis_ids:
            if hypothesis_id not in known_hypothesis_ids:
                errors.append(f"EDA task {task.id} references unknown hypothesis {hypothesis_id}")

    p0_validation_ids = {
        item.id for item in hypotheses if item.category == "validation" and item.priority == "P0"
    }
    p0_leakage_ids = {
        item.id for item in hypotheses if item.category == "leakage" and item.priority == "P0"
    }
    validation_task_links = {
        hypothesis_id
        for task in tasks
        if task.module == "validation_analyzer"
        for hypothesis_id in task.related_hypothesis_ids
    }
    leakage_task_links = {
        hypothesis_id
        for task in tasks
        if task.module == "leakage_checker"
        for hypothesis_id in task.related_hypothesis_ids
    }
    if not p0_validation_ids <= validation_task_links:
        errors.append("P0 validation hypothesis must be linked to a validation task")
    if not p0_leakage_ids <= leakage_task_links:
        errors.append("P0 leakage hypothesis must be linked to a leakage task")

    if not set(model.recommended_module_sequence) <= ALLOWED_EDA_MODULES:
        errors.append("recommended_module_sequence must only contain allowed module names")
    for item in model.recommended_human_checklist:
        if item in ALLOWED_EDA_MODULES:
            errors.append("recommended_human_checklist must contain natural-language checks, not only module names")
    legacy = model.recommended_eda_sequence
    if any(item in ALLOWED_EDA_MODULES for item in legacy) and any(item not in ALLOWED_EDA_MODULES for item in legacy):
        errors.append("recommended_eda_sequence legacy field must not mix module names and human checklist items")

    if _retrieved_sources_available(model) and hypotheses:
        sourced = [item for item in hypotheses if item.supporting_source_ids]
        if len(sourced) / len(hypotheses) < 0.5:
            limitations_text = " ".join(model.scout_limitations).lower()
            if "fewer than 50% of hypotheses have supporting_source_ids" not in limitations_text:
                errors.append(
                    "at least 50% of hypotheses should have supporting_source_ids when retrieved documents are available"
                )
        p0_without_sources = [
            item.id
            for item in hypotheses
            if item.priority == "P0"
            and not item.supporting_source_ids
            and "heuristic" not in item.provenance
        ]
        if p0_without_sources:
            errors.append("P0 hypotheses should have at least one source ID when possible: " + ", ".join(p0_without_sources))

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
        "## Structured findings",
    ]
    for title, finding_type in (
        ("Observed in sources", "observed_in_sources"),
        ("Recommendations", "recommendation"),
        ("Warnings", "warning"),
        ("Caveats", "caveat"),
        ("Limitations", "limitation"),
    ):
        lines.append(f"### {title}")
        group = [finding for finding in model.structured_findings if finding.finding_type == finding_type]
        if group:
            for finding in group:
                caveat = f" Caveat: {finding.caveat}" if finding.caveat else ""
                lines.append(f"- {finding.id}: {finding.claim}{caveat}")
        else:
            lines.append("- None.")
    corrections = model.category_corrections
    lines.extend(["", "## Category corrections"])
    if corrections:
        for correction in corrections:
            lines.append(
                f"- {correction.get('old_id')} -> {correction.get('new_id')}: "
                f"{correction.get('old_category')} -> {correction.get('new_category')}. "
                f"{correction.get('reason')}"
            )
    else:
        lines.append("- None.")

    verification_steps = [
        step
        for hypothesis in model.hypotheses
        for step in hypothesis.verification_steps
    ]
    generic_legacy = [
        hypothesis.id
        for hypothesis in model.hypotheses
        if any(is_generic_verification_text(text) for text in hypothesis.how_to_verify)
    ]
    steps_by_module = Counter(step.module for step in verification_steps)
    lines.extend(
        [
            "",
            "## Verification coverage",
            f"- Number of hypotheses: {len(model.hypotheses)}",
            f"- Number with structured verification steps: {sum(1 for hypothesis in model.hypotheses if hypothesis.verification_steps)}",
            f"- Number with generic legacy how_to_verify: {len(generic_legacy)}",
            "- Number of verification steps by module: "
            + (", ".join(f"{module}={count}" for module, count in sorted(steps_by_module.items())) or "none"),
            "",
            "## StratifiedGroupKFold caveat",
            _stratified_groupkfold_summary_caveat(model),
        ]
    )
    lines.extend(
        [
        "",
        "## Contract quality checks",
        f"- Number of hypotheses: {len(model.hypotheses)}",
        f"- Number of EDA tasks: {len(model.eda_tasks)}",
        f"- Number of linked tasks: {sum(1 for task in model.eda_tasks if task.related_hypothesis_ids)}",
        f"- Number of hypotheses with supporting_source_ids: {sum(1 for hypothesis in model.hypotheses if hypothesis.supporting_source_ids)}",
        "- Blocking tasks: "
        + (", ".join(task.id for task in model.eda_tasks if task.blocking) or "none"),
        "- Validation policy enforced: "
        + ("yes" if _has_safe_temporal_validation(model.hypotheses) else "no"),
        "",
        "## Canonical EDA module sequence",
        *[f"- {module}" for module in model.recommended_module_sequence],
        "",
        "## Human EDA checklist",
        *[f"- {item}" for item in model.recommended_human_checklist],
        "",
        "## P0 EDA checks",
        ]
    )
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
    p0_without_sources = [
        hypothesis.id
        for hypothesis in model.hypotheses
        if hypothesis.priority == "P0" and not hypothesis.supporting_source_ids
    ]
    heuristic_only = [
        hypothesis.id
        for hypothesis in model.hypotheses
        if "heuristic" in hypothesis.provenance and not hypothesis.supporting_source_ids
    ]
    not_verified = [
        hypothesis.id
        for hypothesis in model.hypotheses
        if "not_verified_on_data" in hypothesis.provenance
    ]
    if p0_without_sources:
        lines.append("- P0 hypotheses lack supporting sources: " + ", ".join(p0_without_sources))
    if heuristic_only:
        lines.append("- Heuristic-only hypotheses: " + ", ".join(heuristic_only))
    if not_verified:
        lines.append("- Claims not verified on data: " + ", ".join(not_verified))
    return "\n".join(lines).rstrip() + "\n"


def split_eda_task_plan(payload: dict) -> dict:
    normalized = dict(payload)
    hypothesis_ids = {
        str(hypothesis.get("id"))
        for hypothesis in list(normalized.get("hypotheses") or [])
        if isinstance(hypothesis, dict)
    }
    normalized["eda_tasks"] = ensure_task_ids(
        cleanup_generic_eda_tasks(list(normalized.get("eda_tasks") or []), hypothesis_ids)
    )
    sequence_payload = split_recommended_sequences(normalized)
    normalized.update(sequence_payload)
    validate_research_hypotheses(normalized)
    model = ResearchHypothesesPayload.model_validate(normalized)
    return {
        "schema_version": model.schema_version,
        "competition_id": model.competition_id,
        "competition_url": model.competition_url,
        "task_type": model.task_type,
        "metric": model.metric,
        "eda_tasks": [task.model_dump(mode="json") for task in model.eda_tasks],
        "structured_findings": [finding.model_dump(mode="json") for finding in model.structured_findings],
        "category_corrections": model.category_corrections,
        "hypothesis_index": {
            hypothesis.id: {
                "category": hypothesis.category,
                "priority": hypothesis.priority,
                "claim": hypothesis.claim,
                "verification_step_count": len(hypothesis.verification_steps),
                "supporting_source_count": len(hypothesis.supporting_source_ids),
            }
            for hypothesis in model.hypotheses
        },
        "recommended_module_sequence": model.recommended_module_sequence,
        "recommended_human_checklist": model.recommended_human_checklist,
        "recommended_sequence": model.recommended_module_sequence,
        "recommended_eda_sequence": model.recommended_eda_sequence,
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
        "verification_steps": [
            _normalize_verification_step(step)
            for step in list(item.get("verification_steps") or [])
            if isinstance(step, dict)
        ],
        "expected_evidence_keys": _string_list(item.get("expected_evidence_keys")),
        "failure_condition": item.get("failure_condition"),
        "success_condition": item.get("success_condition"),
        "provenance": _unique_strings(provenance),
        "supporting_source_ids": _string_list(item.get("supporting_source_ids")),
        "confidence": confidence,
        "status": status,
    }


def _normalize_verification_step(item: dict[str, Any]) -> dict[str, Any]:
    module = str(item.get("module") or "table_profiler")
    if module not in ALLOWED_EDA_MODULES:
        module = "table_profiler"
    operation = str(item.get("operation") or "verify").strip() or "verify"
    question = str(item.get("question") or f"What evidence should {operation} produce?").strip()
    outputs = _string_list(item.get("outputs")) or [f"{module}.{operation}"]
    return {
        "id": _normalize_identifier(item.get("id")) or f"verify_{module}_{_slug(operation)}",
        "module": module,
        "operation": operation,
        "question": question if len(question) >= 10 else f"What evidence should {operation} produce?",
        "inputs": _string_list(item.get("inputs")),
        "outputs": outputs,
        "success_criteria": _string_list(item.get("success_criteria")) or ["Required evidence artifact is produced."],
        "failure_criteria": _string_list(item.get("failure_criteria")) or ["Required evidence artifact cannot be produced."],
        "related_task_ids": _string_list(item.get("related_task_ids")),
    }


def normalize_structured_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_findings = [
        item for item in list(payload.get("structured_findings") or []) if isinstance(item, dict)
    ]
    if not raw_findings:
        raw_findings = [
            {
                "id": f"finding_{index:03d}",
                "finding_type": "observed_in_sources",
                "claim": str(value),
                "provenance": ["heuristic"],
            }
            for index, value in enumerate(_string_list(payload.get("scout_findings")), start=1)
            if str(value).strip()
        ]
    findings = [_normalize_finding(item, index) for index, item in enumerate(raw_findings, start=1)]
    if not findings:
        findings.append(
            _normalize_finding(
                {
                    "id": "limitation_scout_mode",
                    "finding_type": "limitation",
                    "claim": "Research Scout mode did not execute EDA or inspect private train/test data.",
                    "provenance": ["heuristic", "not_verified_on_data"],
                    "confidence": "high",
                },
                1,
            )
        )
    return findings


def _normalize_finding(item: dict[str, Any], index: int) -> dict[str, Any]:
    finding_type = str(item.get("finding_type") or "observed_in_sources")
    if finding_type not in {"observed_in_sources", "recommendation", "warning", "caveat", "limitation"}:
        finding_type = "observed_in_sources"
    provenance = [value for value in _string_list(item.get("provenance")) if value in VALID_PROVENANCE]
    if not provenance:
        provenance = ["heuristic"]
    confidence = str(item.get("confidence") or "medium")
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "id": _normalize_identifier(item.get("id")) or f"finding_{index:03d}",
        "finding_type": finding_type,
        "claim": str(item.get("claim") or "Scout finding requires review.").strip(),
        "implication": item.get("implication"),
        "caveat": item.get("caveat"),
        "provenance": _unique_strings(provenance),
        "supporting_source_ids": _string_list(item.get("supporting_source_ids")),
        "related_hypothesis_ids": _string_list(item.get("related_hypothesis_ids")),
        "confidence": confidence,
    }


def _legacy_findings_from_structured(findings: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for finding in findings:
        line = f"{finding['finding_type']}: {finding['claim']}"
        if finding.get("caveat"):
            line += f" Caveat: {finding['caveat']}"
        lines.append(line)
    return lines


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
            "eda_profile_global",
            "P0",
            "table_profiler",
            "What are the row counts, column types, missingness, cardinality, and basic distributions for each train/test table?",
            "Global table profiling is required before validation, leakage checks, drift analysis, and baseline modeling.",
            ["file_inventory.files", "inferred_schema.tables"],
            [
                "table_profiles",
                "table_profiles.missingness",
                "table_profiles.cardinality",
                "table_profiles.dtypes",
            ],
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


def _stratified_groupkfold_summary_caveat(model: ResearchHypothesesPayload) -> str:
    text = json.dumps(model.model_dump(mode="json"), ensure_ascii=False).lower()
    if "stratifiedgroupkfold" in text or "groupkfold" in text or _has_safe_temporal_validation(model.hypotheses):
        return (
            "For temporal/stability competitions, StratifiedGroupKFold with WEEK_NUM may be useful "
            "as a secondary diagnostic, but it is not sufficient as primary validation because it "
            "does not guarantee chronological train-before-validation order. Primary validation "
            "should be strict out-of-time holdout plus rolling or expanding temporal CV."
        )
    return "No StratifiedGroupKFold caveat was triggered."


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
