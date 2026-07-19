from __future__ import annotations

import re
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import Field, field_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.errors import ContractError
from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.eda.metrics.registry import (
    MetricFamily,
    TaskType,
    infer_metric_spec,
    normalize_metric_name,
)


IssueSeverity = Literal["error", "warning"]


class ContractIssue(ContractModel):
    code: str = Field(min_length=1)
    severity: IssueSeverity
    path: str = Field(min_length=1)
    message: str = Field(min_length=1)
    related_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None
    category: str | None = None
    check_ref: str | None = None
    check_module: str | None = None
    allowed_modules: list[str] = Field(default_factory=list)
    task_id: str | None = None
    module: str | None = None
    task_blocking: bool | None = None
    listed_in_blocking_tasks: bool | None = None

    @field_validator("path", "message", mode="before")
    @classmethod
    def _sanitize_scalar(cls, value: Any) -> str:
        return _sanitize_text(value)

    @field_validator("related_ids", mode="before")
    @classmethod
    def _sanitize_identifiers(cls, value: Any) -> list[str]:
        return [_sanitize_text(item) for item in (value or [])][:16]


class ResearchToEdaContractValidationResult(ContractModel):
    valid: bool
    errors: list[ContractIssue] = Field(default_factory=list)
    warnings: list[ContractIssue] = Field(default_factory=list)


class ResearchToEdaContractError(ContractError):
    def __init__(self, result: ResearchToEdaContractValidationResult) -> None:
        self.result = result
        codes = ", ".join(
            issue.code
            + (
                f"[{issue.hypothesis_id or issue.task_id}]"
                if issue.hypothesis_id or issue.task_id
                else ""
            )
            for issue in result.errors[:8]
        )
        suffix = f": {codes}" if codes else ""
        super().__init__(
            f"Research-to-EDA contract validation failed{suffix}",
            contract="research_to_eda",
        )

    def as_manifest_error(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "stage": self.stage,
            "contract": self.contract,
            "recoverable": False,
            "issues": [issue.model_dump(mode="json") for issue in self.result.errors],
            "warnings": [issue.model_dump(mode="json") for issue in self.result.warnings],
        }


class ResearchOutputSchemaError(ResearchToEdaContractError):
    pass


class ResearchOutputVersionError(ResearchToEdaContractError):
    pass


class ResearchEdaCompetitionMismatchError(ResearchToEdaContractError):
    pass


class ResearchEdaReferenceIntegrityError(ResearchToEdaContractError):
    pass


class ResearchEdaModulePlanError(ResearchToEdaContractError):
    pass


class ResearchEdaMetricTaskMismatchError(ResearchToEdaContractError):
    pass


# This registry names deterministic checks backed by the current EDA modules.  Some
# names are compatibility aliases emitted by older Scout prompts and fixtures.
_CHECKS_BY_MODULE: dict[str, frozenset[str]] = {
    "file_inventory": frozenset({
        "detect_table_roles", "roles", "file_formats", "train_test_pairs",
    }),
    "schema_inferer": frozenset({
        "roles", "detect_roles", "detect_global_roles", "detect_target", "detect_id",
        "detect_prediction_column", "detect_time_columns", "detect_group_columns",
        "detect_query_columns",
    }),
    "table_profiler": frozenset({"profile_tables", "column_profiles", "target_profile"}),
    "metric_analyzer": frozenset({
        "basic", "basic_metric_contract", "registry", "resolve_metric",
    }),
    "validation_analyzer": frozenset({
        "policy", "primary_policy", "primary_validation", "select_strategy",
        "select_primary_validation", "period_distribution", "temporal_cv_feasibility",
        "group_policy", "group_cv_feasibility", "ranking_policy",
        "ranking_validation", "iid_feasibility",
    }),
    "leakage_checker": frozenset({
        "basic", "direct_checks", "target_proxy_scan", "target_absent_from_test",
        "target_in_test", "train_test_id_overlap", "group_overlap",
        "ranking_query_overlap",
    }),
    "relationship_inferer": frozenset({"join_key_coverage", "relationships", "cardinality"}),
    "drift_analyzer": frozenset({"generic", "feature_shift", "train_test_shift"}),
    "baseline_runner": frozenset({"honest_baseline", "baseline_feasibility"}),
    "feature_probe": frozenset({"feature_family_probe"}),
    "notebook_static_analysis": frozenset({"static_patterns"}),
}

EDA_CHECK_REGISTRY: Mapping[str, frozenset[str]] = MappingProxyType(_CHECKS_BY_MODULE)

KNOWN_EDA_MODULES = frozenset({
    *_CHECKS_BY_MODULE,
    "target_diagnostics", "feature_diagnostics", "baseline_ablations",
    "interaction_diagnostics", "slice_diagnostics", "visual_diagnostics",
    "source_claim_validation",
})

CORE_EDA_MODULES = frozenset({
    "file_inventory", "schema_inferer", "table_profiler", "metric_analyzer",
    "validation_analyzer", "leakage_checker",
})

MODULE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "schema_inferer": ("file_inventory",),
    "table_profiler": ("schema_inferer",),
    "metric_analyzer": ("table_profiler",),
    "validation_analyzer": ("metric_analyzer",),
    "leakage_checker": ("validation_analyzer",),
    "relationship_inferer": ("schema_inferer",),
    "drift_analyzer": ("table_profiler",),
    "baseline_runner": ("validation_analyzer",),
    "feature_probe": ("relationship_inferer",),
    "baseline_ablations": ("baseline_runner",),
    "interaction_diagnostics": ("baseline_runner",),
    "slice_diagnostics": ("baseline_runner",),
})

CATEGORY_PREFIXES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "schema": ("schema_",),
    "metric": ("metric_",),
    "validation": ("val_", "validation_"),
    "leakage": ("leak_", "leakage_"),
    "relationship": ("relationship_", "rel_"),
    "drift": ("drift_",),
    "baseline": ("baseline_", "base_"),
    "feature": ("feature_", "feat_"),
    "notebook": ("notebook_", "nb_"),
    "leaderboard": ("leaderboard_", "lb_"),
    "data_quality": ("data_quality_", "dq_"),
})

CATEGORY_ALLOWED_MODULES: Mapping[str, frozenset[str]] = MappingProxyType({
    "schema": frozenset({"file_inventory", "schema_inferer", "table_profiler"}),
    "metric": frozenset({"metric_analyzer"}),
    "validation": frozenset({"validation_analyzer"}),
    "leakage": frozenset({"leakage_checker"}),
    "relationship": frozenset({"relationship_inferer"}),
    "drift": frozenset({"drift_analyzer"}),
    "baseline": frozenset({"baseline_runner"}),
    "feature": frozenset({"feature_probe"}),
    "notebook": frozenset({"notebook_static_analysis"}),
    "leaderboard": frozenset({"drift_analyzer"}),
    "data_quality": frozenset({
        "file_inventory", "schema_inferer", "table_profiler", "leakage_checker",
    }),
})

# Backwards-compatible name for callers that imported the original registry.
CATEGORY_CHECK_MODULES = CATEGORY_ALLOWED_MODULES

CATEGORY_ALIASES: Mapping[str, str] = MappingProxyType({
    "dataset_schema": "schema",
    "relationships": "relationship",
    "feature_engineering": "feature",
    "notebook_reverse_engineering": "notebook",
    "leaderboard_risk": "leaderboard",
})

MODULE_ALIASES: Mapping[str, str] = MappingProxyType({
    "notebook_reverse_engineering": "notebook_static_analysis",
    "relationship_analyzer": "relationship_inferer",
})

CANONICAL_BLOCKING_MODULES = frozenset({
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
})

REQUIRED_P0_CATEGORIES = frozenset({"schema", "metric", "validation", "leakage"})

STABLE_ERROR_CODES = frozenset({
    "blocking_task_conflict", "competition_id_mismatch", "custom_metric",
    "custom_metric_limitation_missing", "custom_metric_resolution_missing",
    "dataset_contract_contains_secret", "duplicate_blocking_module", "duplicate_eda_check",
    "duplicate_hypothesis_id", "duplicate_hypothesis_index_mapping",
    "duplicate_module_sequence_entry", "duplicate_related_hypothesis_id",
    "duplicate_task_id", "empty_expected_eda_checks", "empty_hypothesis_id_suffix",
    "forced_temporal_without_evidence", "hypothesis_check_category_mismatch",
    "hypothesis_driven_task_without_hypothesis",
    "hypothesis_id_category_prefix_mismatch", "metric_semantics_mismatch",
    "metric_task_type_mismatch", "missing_p0_hypothesis",
    "missing_p0_module_from_sequence", "module_dependency_missing_from_sequence",
    "module_dependency_order_violation", "one_way_hypothesis_task_mapping",
    "one_way_task_hypothesis_mapping", "p0_depends_only_on_optional_module",
    "premature_eda_factual_claim", "ranking_group_check_missing",
    "ranking_validation_check_missing", "temporal_metric_check_missing",
    "unknown_blocking_module", "unknown_eda_check", "unknown_eda_module",
    "unknown_hypothesis_index_key", "unknown_hypothesis_reference", "unknown_metric",
    "unknown_metric_claims_local_implementation", "unknown_sequence_module",
    "unknown_task_reference", "unplanned_blocking_module", "unstable_hypothesis_id",
})

_STABLE_ID = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")
_PREMATURE_PATTERNS = (
    re.compile(r"\beda\s+(?:has\s+)?confirmed\b", re.I),
    re.compile(r"\bleakage\s+(?:was\s+|has\s+been\s+)?found\b", re.I),
    re.compile(r"\btrain\s+and\s+test\s+overlap\b", re.I),
    re.compile(r"\btarget\s+drift\s+is\s+high\b", re.I),
    re.compile(r"\b(?:the\s+)?dataset\s+contains\s+[\d,]+\s+rows\b", re.I),
    re.compile(r"\btemporal\s+folds\s+are\s+feasible\b", re.I),
    re.compile(r"\bbaseline\s+(?:has\s+)?achieved\b", re.I),
    re.compile(r"\bcolumn\s+\S+\s+is\s+leaking\b", re.I),
    re.compile(r"\btest\s+is\s+definitely\s+after\s+train\b", re.I),
)
_HYPOTHETICAL_PREFIX = re.compile(
    r"^\s*(?:check\s+whether|hypothesis\s*:|may\s+require|sources?\s+suggest|"
    r"eda\s+should\s+verify|potential\s+risk|test\s+whether|verify\s+whether)\b",
    re.I,
)
_SECRET_PATTERNS = (
    re.compile(r"postgresql://[^\s/@:]+:[^\s/@]+@", re.I),
    re.compile(r"\b(?:sk[-_]|ghp_)[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:DEEPSEEK_API_KEY|KAGGLE_KEY|GITHUB_TOKEN)\s*[=:]\s*\S+", re.I),
)

_REFERENCE_DRIVEN_MODULES = frozenset({
    "metric_analyzer", "validation_analyzer", "leakage_checker",
    "relationship_inferer", "drift_analyzer", "baseline_runner", "feature_probe",
})


class ResearchToEdaCanonicalizationRepair(ContractModel):
    code: str = Field(min_length=1)
    path: str = Field(min_length=1)
    original: Any = None
    canonical: Any = None


class ResearchToEdaCanonicalizationResult(ContractModel):
    research_hypotheses: ResearchHypotheses
    eda_task_plan: EdaTaskPlan
    repairs: list[ResearchToEdaCanonicalizationRepair] = Field(default_factory=list)


def extract_check_module(check_ref: str) -> str:
    """Return the normalized module component without changing check semantics."""

    text = str(check_ref or "").strip()
    if not text:
        return ""
    return normalize_module_name(text.split(".", maxsplit=1)[0])


def normalize_module_name(value: Any) -> str:
    name = str(value or "").strip().lower()
    return MODULE_ALIASES.get(name, name)


def normalize_check_reference(check_ref: Any) -> str:
    text = str(check_ref or "").strip()
    if not text:
        return ""
    module, separator, check_name = text.partition(".")
    normalized_module = normalize_module_name(module)
    if not separator:
        return normalized_module
    return f"{normalized_module}.{check_name.strip()}"


def normalize_hypothesis_category(
    value: Any,
    expected_eda_checks: list[str] | tuple[str, ...] = (),
) -> str:
    """Normalize explicit aliases; infer only an unambiguous non-canonical category."""

    category = str(value or "").strip().lower()
    normalized = CATEGORY_ALIASES.get(category, category)
    if normalized in CATEGORY_ALLOWED_MODULES:
        return normalized
    modules = {extract_check_module(check) for check in expected_eda_checks}
    modules.discard("")
    candidates = [
        candidate
        for candidate, allowed in CATEGORY_ALLOWED_MODULES.items()
        if modules and modules <= allowed
    ]
    return candidates[0] if len(candidates) == 1 else normalized


def is_module_blocking(module: Any) -> bool:
    return normalize_module_name(module) in CANONICAL_BLOCKING_MODULES


def synchronize_hypothesis_index(tasks: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            continue
        for hypothesis_id in _unique_text(task.get("related_hypothesis_ids")):
            index.setdefault(hypothesis_id, []).append(task_id)
    return index


def synchronize_blocking_tasks(tasks: list[Mapping[str, Any]]) -> list[str]:
    return _unique_text(
        task.get("module") for task in tasks if bool(task.get("blocking"))
    )


def synchronize_recommended_module_sequence(
    values: Any,
    tasks: list[Mapping[str, Any]],
) -> list[str]:
    requested = _unique_text(
        normalize_module_name(value) for value in (values if isinstance(values, list) else [])
    )
    for task in tasks:
        module = str(task.get("module") or "")
        if task.get("priority") == "P0" and module not in requested:
            requested.append(module)

    requested_set = set(requested)
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(module: str) -> None:
        if module in ordered or module in visiting:
            return
        visiting.add(module)
        for dependency in MODULE_DEPENDENCIES.get(module, ()):
            if dependency in requested_set:
                visit(dependency)
        visiting.remove(module)
        ordered.append(module)

    for module in requested:
        visit(module)
    return ordered


def canonicalize_research_to_eda_contract(
    hypotheses: Mapping[str, Any] | ResearchHypotheses,
    task_plan: Mapping[str, Any] | EdaTaskPlan,
) -> ResearchToEdaCanonicalizationResult:
    """Repair structural Scout aliases and dependent fields before strict validation."""

    research_payload = deepcopy(
        hypotheses.model_dump(mode="json")
        if isinstance(hypotheses, ResearchHypotheses)
        else dict(hypotheses)
    )
    plan_payload = deepcopy(
        task_plan.model_dump(mode="json")
        if isinstance(task_plan, EdaTaskPlan)
        else dict(task_plan)
    )
    repairs: list[ResearchToEdaCanonicalizationRepair] = []

    normalized_hypotheses: list[dict[str, Any]] = []
    for index, raw in enumerate(research_payload.get("hypotheses") or []):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        raw_checks = item.get("expected_eda_checks") or []
        checks = _unique_text(normalize_check_reference(value) for value in raw_checks)
        category = normalize_hypothesis_category(item.get("category"), checks)
        if category != item.get("category"):
            repairs.append(ResearchToEdaCanonicalizationRepair(
                code="hypothesis_category_alias_normalized",
                path=f"research_hypotheses.hypotheses[{index}].category",
                original=item.get("category"),
                canonical=category,
            ))
        if checks != list(raw_checks):
            repairs.append(ResearchToEdaCanonicalizationRepair(
                code="hypothesis_checks_normalized",
                path=f"research_hypotheses.hypotheses[{index}].expected_eda_checks",
                original=list(raw_checks),
                canonical=checks,
            ))
        item["category"] = category
        item["expected_eda_checks"] = checks
        normalized_hypotheses.append(item)
    research_payload["hypotheses"] = normalized_hypotheses
    known_hypotheses = {
        str(item.get("hypothesis_id") or "") for item in normalized_hypotheses
    }

    normalized_tasks: list[dict[str, Any]] = []
    task_by_module: dict[str, dict[str, Any]] = {}
    task_id_remap: dict[str, str] = {}
    for index, raw in enumerate(plan_payload.get("eda_tasks") or []):
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        original_module = item.get("module")
        module = normalize_module_name(original_module)
        item["module"] = module
        task_id = str(item.get("task_id") or "").strip()
        item["related_hypothesis_ids"] = [
            value for value in _unique_text(item.get("related_hypothesis_ids"))
            if value in known_hypotheses
        ]
        item["blocking"] = is_module_blocking(module)
        if module != original_module:
            repairs.append(ResearchToEdaCanonicalizationRepair(
                code="task_module_alias_normalized",
                path=f"eda_task_plan.eda_tasks[{index}].module",
                original=original_module,
                canonical=module,
            ))
        existing = task_by_module.get(module)
        if existing is not None:
            existing["related_hypothesis_ids"] = _unique_text([
                *existing.get("related_hypothesis_ids", []),
                *item.get("related_hypothesis_ids", []),
            ])
            task_id_remap[task_id] = str(existing.get("task_id") or "")
            repairs.append(ResearchToEdaCanonicalizationRepair(
                code="duplicate_module_task_merged",
                path=f"eda_task_plan.eda_tasks[{index}]",
                original=task_id,
                canonical=existing.get("task_id"),
            ))
            continue
        task_by_module[module] = item
        task_id_remap[task_id] = task_id
        normalized_tasks.append(item)

    known_task_ids = {str(item.get("task_id") or "") for item in normalized_tasks}
    for item in normalized_tasks:
        task_id = str(item.get("task_id") or "")
        item["dependencies"] = [
            dependency
            for dependency in _unique_text(
                task_id_remap.get(value, value)
                for value in (item.get("dependencies") or [])
            )
            if dependency in known_task_ids and dependency != task_id
        ]

    plan_payload["eda_tasks"] = normalized_tasks
    plan_payload["hypothesis_index"] = synchronize_hypothesis_index(normalized_tasks)
    plan_payload["recommended_module_sequence"] = synchronize_recommended_module_sequence(
        plan_payload.get("recommended_module_sequence"), normalized_tasks,
    )
    plan_payload["blocking_tasks"] = synchronize_blocking_tasks(normalized_tasks)
    research_payload["eda_tasks"] = deepcopy(normalized_tasks)

    return ResearchToEdaCanonicalizationResult(
        research_hypotheses=ResearchHypotheses.model_validate(research_payload),
        eda_task_plan=EdaTaskPlan.model_validate(plan_payload),
        repairs=repairs,
    )


def validate_research_to_eda_contract(
    hypotheses: ResearchHypotheses,
    task_plan: EdaTaskPlan,
) -> ResearchToEdaContractValidationResult:
    """Validate the published Scout artifacts without data IO, EDA execution, or repair."""

    errors: list[ContractIssue] = []
    warnings: list[ContractIssue] = []

    def add(
        code: str,
        path: str,
        message: str,
        *,
        severity: IssueSeverity = "error",
        related_ids: list[str] | None = None,
        hypothesis_id: str | None = None,
        category: str | None = None,
        check_ref: str | None = None,
        check_module: str | None = None,
        allowed_modules: list[str] | None = None,
        task_id: str | None = None,
        module: str | None = None,
        task_blocking: bool | None = None,
        listed_in_blocking_tasks: bool | None = None,
    ) -> None:
        target = errors if severity == "error" else warnings
        target.append(ContractIssue(
            code=code,
            severity=severity,
            path=path,
            message=_sanitize_text(message),
            related_ids=[_sanitize_text(value) for value in (related_ids or [])][:16],
            hypothesis_id=hypothesis_id,
            category=category,
            check_ref=check_ref,
            check_module=check_module,
            allowed_modules=allowed_modules or [],
            task_id=task_id,
            module=module,
            task_blocking=task_blocking,
            listed_in_blocking_tasks=listed_in_blocking_tasks,
        ))

    if hypotheses.competition_id != task_plan.competition_id:
        add(
            "competition_id_mismatch",
            "competition_id",
            "Research hypotheses and EDA task plan target different competitions.",
        )

    hypothesis_ids = [str(item.hypothesis_id) for item in hypotheses.hypotheses]
    _report_duplicates(hypothesis_ids, "hypotheses", "duplicate_hypothesis_id", add)
    known_hypotheses = set(hypothesis_ids)

    p0_categories: set[str] = set()
    for index, hypothesis in enumerate(hypotheses.hypotheses):
        path = f"hypotheses[{index}]"
        identifier = str(hypothesis.hypothesis_id)
        if hypothesis.priority == "P0":
            p0_categories.add(hypothesis.category)
        prefixes = CATEGORY_PREFIXES[hypothesis.category]
        matching_prefix = next((prefix for prefix in prefixes if identifier.startswith(prefix)), None)
        if matching_prefix is None:
            add(
                "hypothesis_id_category_prefix_mismatch",
                f"{path}.hypothesis_id",
                "Hypothesis ID prefix does not match its semantic category.",
                severity="warning",
                related_ids=[identifier],
            )
        elif not identifier[len(matching_prefix):]:
            add(
                "empty_hypothesis_id_suffix",
                f"{path}.hypothesis_id",
                "Hypothesis ID requires a stable suffix after the category prefix.",
                related_ids=[identifier],
            )
        if not _STABLE_ID.fullmatch(identifier):
            add(
                "unstable_hypothesis_id",
                f"{path}.hypothesis_id",
                "Hypothesis ID must use stable lowercase snake-case characters.",
                related_ids=[identifier],
            )

        checks = list(hypothesis.expected_eda_checks)
        _report_duplicates(checks, f"{path}.expected_eda_checks", "duplicate_eda_check", add)
        if not checks:
            add(
                "empty_expected_eda_checks",
                f"{path}.expected_eda_checks",
                "Each hypothesis requires at least one executable EDA check.",
                related_ids=[identifier],
            )
        valid_check_modules: set[str] = set()
        allowed_modules = CATEGORY_ALLOWED_MODULES[hypothesis.category]
        reported_category_mismatch = False
        for check_index, check in enumerate(checks):
            check_path = (
                f"research_hypotheses.{path}.expected_eda_checks[{check_index}]"
            )
            module = extract_check_module(check)
            _, separator, check_name = check.partition(".")
            if not separator or not check_name or check_name not in EDA_CHECK_REGISTRY.get(module, ()):
                add(
                    "unknown_eda_check",
                    check_path,
                    "Expected EDA check is not present in the deterministic check registry.",
                    related_ids=[identifier],
                    hypothesis_id=identifier,
                    category=hypothesis.category,
                    check_ref=check,
                    check_module=module,
                    allowed_modules=sorted(allowed_modules),
                )
            else:
                valid_check_modules.add(module)
            if module and module not in allowed_modules:
                reported_category_mismatch = True
                add(
                    "hypothesis_check_category_mismatch",
                    check_path,
                    f"{hypothesis.category.replace('_', ' ').title()} hypothesis references "
                    f"module {module!r}, which is not allowed for its category.",
                    related_ids=[identifier],
                    hypothesis_id=identifier,
                    category=hypothesis.category,
                    check_ref=check,
                    check_module=module,
                    allowed_modules=sorted(allowed_modules),
                )
        if checks and not reported_category_mismatch and not (
            valid_check_modules & allowed_modules
        ):
            first_check = checks[0]
            add(
                "hypothesis_check_category_mismatch",
                f"research_hypotheses.{path}.expected_eda_checks",
                "Expected checks do not include a registered module suitable for the "
                "hypothesis category.",
                related_ids=[identifier],
                hypothesis_id=identifier,
                category=hypothesis.category,
                check_ref=first_check,
                check_module=extract_check_module(first_check),
                allowed_modules=sorted(allowed_modules),
            )
        if hypothesis.priority == "P0" and valid_check_modules and not (
            valid_check_modules & CORE_EDA_MODULES
        ):
            add(
                "p0_depends_only_on_optional_module",
                f"{path}.expected_eda_checks",
                "A P0 hypothesis cannot depend only on optional EDA modules.",
                related_ids=[identifier],
            )
        _check_premature_text(hypothesis.claim, f"{path}.claim", identifier, add)

    for category in sorted(REQUIRED_P0_CATEGORIES - p0_categories):
        add(
            "missing_p0_hypothesis",
            "hypotheses",
            f"Missing required P0 semantic hypothesis category: {category}.",
            related_ids=[category],
        )

    for index, finding in enumerate(hypotheses.structured_findings):
        if isinstance(finding, dict):
            for key in ("finding", "claim", "summary", "description"):
                value = finding.get(key)
                if isinstance(value, str):
                    _check_premature_text(value, f"structured_findings[{index}].{key}", None, add)

    task_ids = [str(task.task_id) for task in task_plan.eda_tasks]
    _report_duplicates(task_ids, "eda_tasks", "duplicate_task_id", add)
    known_tasks = set(task_ids)
    task_by_id = {str(task.task_id): task for task in task_plan.eda_tasks}
    planned_modules = {task.module for task in task_plan.eda_tasks}

    for index, task in enumerate(task_plan.eda_tasks):
        path = f"eda_tasks[{index}]"
        task_id = str(task.task_id)
        if task.module not in KNOWN_EDA_MODULES:
            add(
                "unknown_eda_module",
                f"{path}.module",
                "Task references an EDA module that is not implemented by this engine.",
                related_ids=[task_id],
            )
        related = [str(value) for value in task.related_hypothesis_ids]
        _report_duplicates(
            related, f"{path}.related_hypothesis_ids", "duplicate_related_hypothesis_id", add,
        )
        for value in related:
            if value not in known_hypotheses:
                add(
                    "unknown_hypothesis_reference",
                    f"{path}.related_hypothesis_ids",
                    "Task references an unknown hypothesis ID.",
                    related_ids=[task_id, value],
                )
        if task.module in _REFERENCE_DRIVEN_MODULES and not related:
            add(
                "hypothesis_driven_task_without_hypothesis",
                f"{path}.related_hypothesis_ids",
                "Hypothesis-driven EDA task requires at least one hypothesis reference.",
                related_ids=[task_id],
            )
        _walk_contract_text(task.params, f"{path}.params", task_id, add)

    for hypothesis_id, indexed_task_ids in task_plan.hypothesis_index.items():
        hypothesis_id = str(hypothesis_id)
        path = f"hypothesis_index.{hypothesis_id}"
        if hypothesis_id not in known_hypotheses:
            add(
                "unknown_hypothesis_index_key",
                path,
                "Hypothesis index contains an unknown hypothesis ID.",
                related_ids=[hypothesis_id],
            )
        values = [str(value) for value in indexed_task_ids]
        _report_duplicates(values, path, "duplicate_hypothesis_index_mapping", add)
        for task_id in values:
            if task_id not in known_tasks:
                add(
                    "unknown_task_reference",
                    path,
                    "Hypothesis index references an unknown task ID.",
                    related_ids=[hypothesis_id, task_id],
                )
            elif hypothesis_id not in {str(value) for value in task_by_id[task_id].related_hypothesis_ids}:
                add(
                    "one_way_hypothesis_task_mapping",
                    path,
                    "Hypothesis index mapping is absent from the task's related hypotheses.",
                    related_ids=[hypothesis_id, task_id],
                )
    for task in task_plan.eda_tasks:
        task_id = str(task.task_id)
        for hypothesis_id in task.related_hypothesis_ids:
            indexed = {str(value) for value in task_plan.hypothesis_index.get(hypothesis_id, [])}
            if task_id not in indexed:
                add(
                    "one_way_task_hypothesis_mapping",
                    f"hypothesis_index.{hypothesis_id}",
                    "Task-to-hypothesis relation is missing from the hypothesis index.",
                    related_ids=[str(hypothesis_id), task_id],
                )

    blocking = list(task_plan.blocking_tasks)
    _report_duplicates(blocking, "blocking_tasks", "duplicate_blocking_module", add)
    for module in blocking:
        if module not in KNOWN_EDA_MODULES:
            add("unknown_blocking_module", "blocking_tasks", "Blocking module is unknown.")
        if module not in planned_modules:
            add("unplanned_blocking_module", "blocking_tasks", "Blocking module has no planned task.")
    for index, task in enumerate(task_plan.eda_tasks):
        listed = task.module in blocking
        if listed != task.blocking:
            add(
                "blocking_task_conflict",
                f"eda_task_plan.eda_tasks[{index}].blocking",
                "Task blocking flag conflicts with the plan-level blocking module list.",
                related_ids=[str(task.task_id)],
                task_id=str(task.task_id),
                module=task.module,
                task_blocking=task.blocking,
                listed_in_blocking_tasks=listed,
            )

    sequence = list(task_plan.recommended_module_sequence)
    _report_duplicates(sequence, "recommended_module_sequence", "duplicate_module_sequence_entry", add)
    for module in sequence:
        if module not in KNOWN_EDA_MODULES:
            add("unknown_sequence_module", "recommended_module_sequence", "Module sequence contains an unknown module.")
    positions = {module: index for index, module in enumerate(sequence)}
    for module, dependencies in MODULE_DEPENDENCIES.items():
        if module not in positions:
            continue
        for dependency in dependencies:
            if dependency in planned_modules and dependency not in positions:
                add(
                    "module_dependency_missing_from_sequence",
                    "recommended_module_sequence",
                    "Recommended sequence omits a planned dependency.",
                    related_ids=[dependency, module],
                )
                continue
            if dependency in positions and positions[dependency] > positions[module]:
                add(
                    "module_dependency_order_violation",
                    "recommended_module_sequence",
                    "A downstream module appears before its dependency.",
                    related_ids=[dependency, module],
                )
    required_sequence_modules = {
        task.module for task in task_plan.eda_tasks if task.priority == "P0"
    }
    for module in sorted(required_sequence_modules - set(sequence)):
        add(
            "missing_p0_module_from_sequence",
            "recommended_module_sequence",
            "Recommended sequence omits a planned P0 module.",
            related_ids=[module],
        )

    _validate_metric_task_contract(hypotheses, task_plan, add)
    _check_dataset_secrets(task_plan.dataset, add)

    return ResearchToEdaContractValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
    )


def require_valid_research_to_eda_contract(
    hypotheses: ResearchHypotheses,
    task_plan: EdaTaskPlan,
) -> ResearchToEdaContractValidationResult:
    result = validate_research_to_eda_contract(hypotheses, task_plan)
    if not result.valid:
        codes = {issue.code for issue in result.errors}
        if "competition_id_mismatch" in codes:
            error_type = ResearchEdaCompetitionMismatchError
        elif codes & {
            "duplicate_hypothesis_id", "duplicate_related_hypothesis_id",
            "unknown_hypothesis_reference", "unknown_hypothesis_index_key",
            "unknown_task_reference", "one_way_hypothesis_task_mapping",
            "one_way_task_hypothesis_mapping", "duplicate_hypothesis_index_mapping",
        }:
            error_type = ResearchEdaReferenceIntegrityError
        elif codes & {
            "unknown_eda_module", "unknown_blocking_module", "unplanned_blocking_module",
            "blocking_task_conflict", "unknown_sequence_module",
            "module_dependency_order_violation", "module_dependency_missing_from_sequence",
            "missing_p0_module_from_sequence",
        }:
            error_type = ResearchEdaModulePlanError
        elif codes & {
            "metric_task_type_mismatch", "metric_semantics_mismatch",
            "unknown_metric_claims_local_implementation", "ranking_validation_check_missing",
            "ranking_group_check_missing", "temporal_metric_check_missing",
            "forced_temporal_without_evidence", "custom_metric_resolution_missing",
            "custom_metric_limitation_missing",
        }:
            error_type = ResearchEdaMetricTaskMismatchError
        else:
            error_type = ResearchToEdaContractError
        raise error_type(result)
    return result


def _validate_metric_task_contract(hypotheses: ResearchHypotheses, task_plan: EdaTaskPlan, add: Any) -> None:
    metric_name = str((task_plan.metric or {}).get("name") or "unknown")
    normalized = normalize_metric_name(metric_name)
    spec = infer_metric_spec(metric_name, task_plan.task_type)
    try:
        task_type = TaskType(str(task_plan.task_type))
    except ValueError:
        task_type = TaskType.UNKNOWN

    if spec.family in {MetricFamily.UNKNOWN, MetricFamily.CUSTOM}:
        add(
            "unknown_metric" if spec.family is MetricFamily.UNKNOWN else "custom_metric",
            "metric.name",
            "Metric requires explicit local resolution before EDA can score it.",
            severity="warning",
        )
        if task_plan.metric.get("local_metric_available") is True:
            add(
                "unknown_metric_claims_local_implementation",
                "metric.local_metric_available",
                "An unknown metric cannot claim an available local implementation.",
            )
    elif task_type is not TaskType.UNKNOWN and task_type not in spec.task_types:
        add(
            "metric_task_type_mismatch",
            "metric.name",
            "Metric family is incompatible with the declared task type.",
        )

    explicit_flags = {
        "requires_probabilities": spec.requires_probabilities,
        "requires_calibration": spec.requires_calibration,
        "requires_groups": spec.requires_groups,
        "requires_time": spec.requires_time,
        "requires_query_groups": spec.requires_query_groups,
        "rank_based": spec.rank_based,
        "threshold_search_needed": spec.threshold_search_needed,
    }
    for field, expected in explicit_flags.items():
        if field in task_plan.metric and task_plan.metric[field] is not None:
            if bool(task_plan.metric[field]) != bool(expected):
                add(
                    "metric_semantics_mismatch",
                    f"metric.{field}",
                    "Explicit metric semantics contradict the local metric registry.",
                )

    all_checks = {
        check for hypothesis in hypotheses.hypotheses for check in hypothesis.expected_eda_checks
    }
    if spec.family in {MetricFamily.UNKNOWN, MetricFamily.CUSTOM}:
        if not any(check.startswith("metric_analyzer.") for check in all_checks):
            add(
                "custom_metric_resolution_missing",
                "hypotheses",
                "Custom or unknown metrics require a metric-analyzer resolution check.",
            )
        metric_limitations = [
            limitation
            for hypothesis in hypotheses.hypotheses
            if hypothesis.category == "metric"
            for limitation in hypothesis.limitations
        ]
        if not metric_limitations and not hypotheses.scout_limitations:
            add(
                "custom_metric_limitation_missing",
                "hypotheses",
                "Custom or unknown metric contract must preserve an explicit limitation.",
            )
    if spec.family is MetricFamily.RANKING and not any("ranking" in check for check in all_checks):
        add(
            "ranking_validation_check_missing",
            "hypotheses",
            "Ranking metrics require a ranking-aware validation or query-overlap check.",
        )
    if spec.family is MetricFamily.RANKING and not any(
        token in check for check in all_checks for token in ("query", "group")
    ):
        add(
            "ranking_group_check_missing",
            "hypotheses",
            "Ranking metrics require a query/group integrity check.",
        )
    if spec.requires_time and not any(
        token in check for check in all_checks for token in ("temporal", "period", "time")
    ):
        add(
            "temporal_metric_check_missing",
            "hypotheses",
            "Temporal/stability metric requires time and fold-feasibility checks.",
        )

    forced_temporal = any(
        _contains_forced_temporal(task.params)
        for task in task_plan.eda_tasks
        if task.module == "validation_analyzer"
    )
    if forced_temporal and not spec.requires_time and task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.REGRESSION,
    }:
        add(
            "forced_temporal_without_evidence",
            "eda_tasks",
            "IID task plan forces temporal validation without temporal metric evidence.",
        )


def _contains_forced_temporal(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            ("validation" in str(key).lower() or "policy" in str(key).lower())
            and _contains_forced_temporal(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forced_temporal(item) for item in value)
    if isinstance(value, str):
        normalized = value.lower().replace("-", "_")
        return normalized in {"temporal", "time_series", "rolling", "expanding", "out_of_time"}
    return False


def _report_duplicates(values: list[str], path: str, code: str, add: Any) -> None:
    seen: set[str] = set()
    reported: set[str] = set()
    for value in values:
        if value in seen and value not in reported:
            add(code, path, "Contract collection contains duplicate values.", related_ids=[value])
            reported.add(value)
        seen.add(value)


def _check_premature_text(text: str, path: str, identifier: str | None, add: Any) -> None:
    if _HYPOTHETICAL_PREFIX.search(text):
        return
    if any(pattern.search(text) for pattern in _PREMATURE_PATTERNS):
        add(
            "premature_eda_factual_claim",
            path,
            "Pre-EDA artifact contains an unsupported factual result claim.",
            related_ids=[identifier] if identifier else [],
        )


def _walk_contract_text(value: Any, path: str, identifier: str, add: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and any(
                token in str(key).lower() for token in ("claim", "description", "summary", "finding")
            ):
                _check_premature_text(item, f"{path}.{key}", identifier, add)
            elif isinstance(item, (dict, list)):
                _walk_contract_text(item, f"{path}.{key}", identifier, add)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_contract_text(item, f"{path}[{index}]", identifier, add)


def _check_dataset_secrets(dataset: dict[str, Any], add: Any) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{path}.{key}"
                if any(token in str(key).lower() for token in ("password", "secret", "token", "api_key")):
                    add("dataset_contract_contains_secret", key_path, "Dataset contract must not carry credentials.")
                walk(item, key_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            add("dataset_contract_contains_secret", path, "Dataset contract must not carry credentials.")

    walk(dataset, "dataset")


def _unique_text(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _sanitize_text(value: Any) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = re.sub(r"(?i)(?:api[_-]?key|password|token)\s*[=:]\s*\S+", "credential=[REDACTED]", text)
    text = re.sub(
        r"(?i)(?:[A-Z]:\\(?:Users|Documents and Settings)\\|/(?:home|Users)/)[^\s,;]+",
        "[USER_PATH]",
        text,
    )
    return text[:500]


__all__ = [
    "CANONICAL_BLOCKING_MODULES",
    "CATEGORY_ALIASES",
    "CATEGORY_ALLOWED_MODULES",
    "CATEGORY_CHECK_MODULES",
    "CATEGORY_PREFIXES",
    "CORE_EDA_MODULES",
    "ContractIssue",
    "EDA_CHECK_REGISTRY",
    "KNOWN_EDA_MODULES",
    "MODULE_ALIASES",
    "MODULE_DEPENDENCIES",
    "ResearchToEdaCanonicalizationRepair",
    "ResearchToEdaCanonicalizationResult",
    "ResearchEdaCompetitionMismatchError",
    "ResearchEdaMetricTaskMismatchError",
    "ResearchEdaModulePlanError",
    "ResearchEdaReferenceIntegrityError",
    "ResearchOutputSchemaError",
    "ResearchOutputVersionError",
    "ResearchToEdaContractError",
    "ResearchToEdaContractValidationResult",
    "STABLE_ERROR_CODES",
    "canonicalize_research_to_eda_contract",
    "extract_check_module",
    "is_module_blocking",
    "normalize_check_reference",
    "normalize_hypothesis_category",
    "normalize_module_name",
    "require_valid_research_to_eda_contract",
    "synchronize_blocking_tasks",
    "synchronize_hypothesis_index",
    "synchronize_recommended_module_sequence",
    "validate_research_to_eda_contract",
]
