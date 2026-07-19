from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from kaggle_researcher.contracts.evidence import (
    EvidencePathResolutionError,
    resolve_evidence_ref,
)


_PRIORITY = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_STRENGTH = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class ColumnEvidence:
    name: str
    evidence_ref: str
    role: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class CompiledAction:
    section_id: str
    rule_id: str
    priority: str
    evidence_strength: str
    action: str
    reason: str
    evidence_refs: tuple[str, ...]
    hypothesis_categories: tuple[str, ...]
    confidence: str = "medium"
    feature_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CompiledExperiment:
    rule_id: str
    experiment_id: str
    priority: str
    evidence_strength: str
    name: str
    hypothesis: str
    change: str
    feature_inputs: tuple[str, ...]
    model_family: str
    validation_strategy: str
    success_metric: str
    acceptance_rule: str
    evidence_refs: tuple[str, ...]
    hypothesis_categories: tuple[str, ...]
    risks: tuple[str, ...]
    fit_scope: str


@dataclass(frozen=True)
class StrategyCompilation:
    actions: tuple[CompiledAction, ...]
    experiments: tuple[CompiledExperiment, ...]


@dataclass(frozen=True)
class StrategyContext:
    competition_id: str
    task_type: str
    metric: Mapping[str, Any]
    metric_name: str
    greater_is_better: bool | None
    requires_threshold: bool
    validation_strategy: str
    primary_validation: Mapping[str, Any]
    target_column: str | None
    primary_id_column: str | None
    feature_columns_by_role: Mapping[str, tuple[str, ...]]
    baseline_model_family: str | None
    baseline_evidence: Mapping[str, Any]
    baseline_ablation_evidence: Mapping[str, Any]
    leakage_evidence: tuple[Mapping[str, Any], ...]
    drift_evidence: Mapping[str, Any]
    relationship_evidence: Mapping[str, Any]
    hypothesis_results: tuple[Mapping[str, Any], ...]
    train_rows: int | None
    train_rows_ref: str | None
    multiple_seed_diagnostics_supported: bool
    columns: tuple[ColumnEvidence, ...]
    feature_diagnostics: Mapping[str, Any]
    feature_probes: tuple[Mapping[str, Any], ...]
    ablation_findings: tuple[Mapping[str, Any], ...]
    evidence_pack: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_evidence(
        cls,
        *,
        competition_id: str,
        evidence_pack: Mapping[str, Any],
        task_type: str,
        metric_name: str,
        validation_strategy: str,
    ) -> "StrategyContext":
        metric = _mapping(evidence_pack.get("metric_evidence"))
        validation = _mapping(evidence_pack.get("validation_evidence"))
        primary_validation = _mapping(validation.get("primary_validation"))
        schema = _mapping(evidence_pack.get("inferred_schema"))
        baseline = _mapping(evidence_pack.get("baseline_evidence"))
        baseline_status = str(baseline.get("status") or "").casefold()
        model_family = (
            str(baseline.get("model_type"))
            if baseline_status in {"completed", "complete", "success", "successful", "succeeded"}
            and baseline.get("model_type")
            else None
        )
        ablations = _mapping(evidence_pack.get("baseline_ablation_evidence"))
        columns = _collect_columns(evidence_pack)
        train_rows, train_rows_ref = _train_row_evidence(evidence_pack, schema)
        return cls(
            competition_id=competition_id,
            task_type=task_type,
            metric=metric,
            metric_name=metric_name,
            greater_is_better=metric.get("greater_is_better"),
            requires_threshold=metric.get("requires_threshold") is True,
            validation_strategy=validation_strategy,
            primary_validation=primary_validation,
            target_column=(str(schema["target_column"]) if schema.get("target_column") else None),
            primary_id_column=(str(schema["primary_id_column"]) if schema.get("primary_id_column") else None),
            feature_columns_by_role=_columns_by_role(columns),
            baseline_model_family=model_family,
            baseline_evidence=baseline,
            baseline_ablation_evidence=ablations,
            leakage_evidence=tuple(
                item for item in evidence_pack.get("leakage_evidence") or []
                if isinstance(item, Mapping)
            ),
            drift_evidence=_mapping(evidence_pack.get("drift_evidence")),
            relationship_evidence=_mapping(evidence_pack.get("relationship_evidence")),
            hypothesis_results=tuple(
                item for item in evidence_pack.get("hypothesis_results") or []
                if isinstance(item, Mapping)
            ),
            train_rows=train_rows,
            train_rows_ref=train_rows_ref,
            multiple_seed_diagnostics_supported=bool(
                validation.get("multiple_seed_diagnostics_supported")
                or primary_validation.get("multiple_seed_diagnostics_supported")
                or primary_validation.get("repeated_cv")
            ),
            columns=columns,
            feature_diagnostics=_mapping(evidence_pack.get("feature_diagnostics")),
            feature_probes=tuple(
                item for item in evidence_pack.get("feature_probe_evidence") or []
                if isinstance(item, Mapping)
            ),
            ablation_findings=tuple(
                item for item in ablations.get("feature_block_findings") or []
                if isinstance(item, Mapping)
            ),
            evidence_pack=evidence_pack,
        )

    @property
    def model_family(self) -> str | None:
        return self.baseline_model_family or _registry_model_family(self.task_type)

    def column(self, name: str) -> ColumnEvidence | None:
        folded = name.casefold()
        return next((item for item in self.columns if item.name.casefold() == folded), None)

    def diagnostic_ref(self, column_name: str, family: str) -> str | None:
        collection = _mapping(self.feature_diagnostics.get(family)).get("columns") or []
        for index, raw in enumerate(collection):
            item = _mapping(raw)
            if str(item.get("column") or "").casefold() == column_name.casefold():
                return f"feature_diagnostics.{family}.columns[{index}]"
        return None

    def ablation(self, block: str) -> Mapping[str, Any] | None:
        return next(
            (
                item for item in self.ablation_findings
                if str(item.get("feature_block") or item.get("configuration") or "") == block
            ),
            None,
        )


class StrategyRule(Protocol):
    rule_id: str
    required_evidence_refs: tuple[str, ...]
    priority: str

    def applies(self, context: StrategyContext) -> bool: ...

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]: ...

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]: ...


class _FeatureProbeRule:
    rule_id = "feature.probe"
    required_evidence_refs = ("feature_probe_evidence",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(context.feature_probes and context.validation_strategy and context.model_family)

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        actions: list[CompiledAction] = []
        for probe in context.feature_probes:
            family = str(probe.get("feature_family") or "").strip()
            status = str(probe.get("status") or probe.get("potential") or "").casefold()
            if not family or status not in {"high_potential", "high"}:
                continue
            ref = f"feature_probe_evidence.{family}"
            if not _resolves(context, ref):
                continue
            conflict = _probe_conflict(context, family)
            if _probe_has_concrete_non_conflicting_ablation(context, family):
                continue
            actions.append(CompiledAction(
                section_id="feature_priorities",
                rule_id=f"{self.rule_id}.{_safe_id(family)}",
                priority="P2" if conflict else "P1",
                evidence_strength="low" if conflict else "medium",
                action=f"Test the `{family}` feature family as an isolated, fold-controlled ablation.",
                reason=(
                    "The EDA probe is promising but conflicts with an adverse or unstable baseline ablation; treat it as a control experiment."
                    if conflict else
                    "The structured EDA probe marked this feature family high-potential; it remains a hypothesis until paired validation confirms it."
                ),
                evidence_refs=(ref,),
                hypothesis_categories=("feature", "validation"),
                confidence="low" if conflict else "medium",
            ))
        return actions

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        experiments: list[CompiledExperiment] = []
        for action in self.build_actions(context):
            family = action.rule_id.rsplit(".", 1)[-1]
            experiments.append(_experiment(
                context,
                rule_id=action.rule_id,
                experiment_id=f"fallback_exp_probe_{family}",
                priority=action.priority,
                evidence_strength=action.evidence_strength,
                name=f"Controlled {family} feature-family ablation",
                hypothesis=f"The `{family}` feature family adds stable validation signal beyond the current baseline.",
                change=f"Add only the `{family}` feature family while holding folds, model, and all other inputs fixed.",
                feature_inputs=(),
                evidence_refs=action.evidence_refs,
                risks=("The probe may be fold-unstable or redundant with existing features.",),
                fit_scope="within_fold",
            ))
        return experiments


class _AblationRule:
    rule_id = "feature.ablation"
    required_evidence_refs = ("baseline_ablation_evidence.feature_block_findings",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(context.ablation_findings)

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        actions: list[CompiledAction] = []
        for finding in context.ablation_findings:
            block = str(finding.get("feature_block") or finding.get("configuration") or "").strip()
            status = str(finding.get("status") or "not_testable")
            if not block or status == "not_testable":
                continue
            ref = f"baseline_ablation_evidence.feature_block_findings.{block}"
            if not _resolves(context, ref):
                continue
            delta = finding.get("delta_metric", finding.get("delta_vs_best_prior"))
            stability = str(finding.get("stability") or finding.get("stability_vs_best_prior") or "unknown")
            confidence = str(finding.get("confidence") or "medium")
            if confidence not in {"low", "medium", "high"}:
                confidence = "medium"
            delta_text = f"; recorded delta={delta}" if isinstance(delta, (int, float)) else ""
            verb = "Retain" if status in {"helped", "best_overall", "competitive"} else "Retest"
            actions.append(CompiledAction(
                section_id="feature_priorities",
                rule_id=f"{self.rule_id}.{_safe_id(block)}",
                priority="P1" if status in {"helped", "best_overall"} and stability == "stable" else "P2",
                evidence_strength="high" if stability == "stable" else "medium",
                action=f"{verb} the `{block}` feature block only under a paired ablation against the comparison anchor.",
                reason=f"Baseline ablation status={status}, stability={stability}{delta_text}; this is diagnostic evidence, not a score promise.",
                evidence_refs=(ref,),
                hypothesis_categories=("feature", "baseline", "validation"),
                confidence=confidence,
            ))
        return actions

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        experiments: list[CompiledExperiment] = []
        for action in self.build_actions(context):
            block = action.rule_id.rsplit(".", 1)[-1]
            experiments.append(_experiment(
                context,
                rule_id=action.rule_id,
                experiment_id=f"fallback_exp_ablation_{block}",
                priority=action.priority,
                evidence_strength=action.evidence_strength,
                name=f"Paired {block} block verification",
                hypothesis=(
                    f"The observed `{block}` block contribution is reproducible on "
                    "the locked folds."
                ),
                change=(
                    f"Toggle only the `{block}` block against the recorded comparison "
                    "anchor using identical folds, preprocessing boundaries, and model."
                ),
                feature_inputs=(), evidence_refs=action.evidence_refs,
                risks=(
                    "The observed ablation delta may be small, unstable, or specific to the diagnostic baseline.",
                ),
                fit_scope="within_fold",
            ))
        return experiments


class _StructuralFeatureRule:
    rule_id = "feature.structural"
    required_evidence_refs = ("inferred_schema",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(context.validation_strategy and context.model_family and context.columns)

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        return [action for action, _ in _structural_candidates(context)]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        return [experiment for _, experiment in _structural_candidates(context)]


class _PreprocessingRule:
    rule_id = "model.preprocessing"
    required_evidence_refs = ("feature_diagnostics",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(context.validation_strategy and context.model_family)

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        categorical = _diagnostic_columns(context, "categorical_feature_diagnostics")
        numeric_rows = _mapping(
            context.feature_diagnostics.get("numeric_feature_diagnostics")
        ).get("columns") or []
        for index, raw in enumerate(numeric_rows):
            item = _mapping(raw)
            name = str(item.get("column") or "").strip()
            kind = str(
                item.get("feature_value_type")
                or item.get("feature_numeric_kind")
                or ""
            )
            if (
                name
                and context.column(name)
                and kind in {"ordinal_low_cardinality", "binary"}
            ):
                categorical.append((
                    name,
                    f"feature_diagnostics.numeric_feature_diagnostics.columns[{index}]",
                ))
        categorical = list(dict.fromkeys(categorical))
        if not categorical:
            return []
        refs = tuple(item[1] for item in categorical[:6])
        names = [item[0] for item in categorical[:6]]
        return [CompiledAction(
            section_id="modeling_plan",
            rule_id=self.rule_id,
            priority="P1",
            evidence_strength="high",
            action="Fit categorical preprocessing inside each validation fold and preserve unknown-category handling.",
            reason=f"Feature diagnostics identify actual categorical inputs: {', '.join(names)}.",
            evidence_refs=refs,
            hypothesis_categories=("schema", "feature", "validation"),
            confidence="high",
        )]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        return []


class _GenericFeatureDiagnosticsRule:
    rule_id = "feature.diagnostics"
    required_evidence_refs = ("feature_diagnostics",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(
            context.validation_strategy
            and context.model_family
            and context.feature_diagnostics
        )

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        return [action for action, _ in _generic_diagnostic_candidates(context)]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        return [experiment for _, experiment in _generic_diagnostic_candidates(context)]


class _RelationshipAggregationRule:
    rule_id = "feature.relationship_aggregation"
    required_evidence_refs = ("relationship_evidence.relationships",)
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(
            context.validation_strategy
            and context.model_family
            and context.relationship_evidence.get("relationships")
        )

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        return [action for action, _ in _relationship_candidates(context)]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        return [experiment for _, experiment in _relationship_candidates(context)]


class _ModelingRule:
    rule_id = "model.staged"
    required_evidence_refs = ("validation_evidence.primary_validation",)
    priority = "P2"

    def applies(self, context: StrategyContext) -> bool:
        return bool(context.validation_strategy and context.model_family)

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        validation_ref = "validation_evidence.primary_validation"
        if not _resolves(context, validation_ref):
            return []
        comparison_family = _nonlinear_model_family(context.task_type)
        return [
            CompiledAction(
                section_id="modeling_plan",
                rule_id=f"{self.rule_id}.nonlinear",
                priority="P1",
                evidence_strength="medium",
                action=f"After the comparison anchor is reproducible, run one controlled `{comparison_family or context.model_family}` nonlinear/tabular comparison.",
                reason="The project deterministic model registry supports this task type; current-dataset superiority is not assumed.",
                evidence_refs=(validation_ref,),
                hypothesis_categories=("model", "baseline", "validation"),
                confidence="low" if not context.baseline_model_family else "medium",
            ),
            CompiledAction(
                section_id="modeling_plan",
                rule_id=f"{self.rule_id}.ensemble_gate",
                priority="P3",
                evidence_strength="low",
                action="Consider an ensemble only after separate models show complementary out-of-fold errors on the locked folds.",
                reason="Ensembling is gated on measured OOF complementarity; no ensemble benefit is inferred in advance.",
                evidence_refs=(validation_ref,),
                hypothesis_categories=("model", "validation"),
                confidence="low",
            ),
        ]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        comparison_family = _nonlinear_model_family(context.task_type)
        if not context.baseline_model_family or not comparison_family:
            return []
        refs = ["validation_evidence.primary_validation"]
        for ref in ("baseline_evidence.status", "baseline_evidence.model_type"):
            if _resolves(context, ref):
                refs.append(ref)
        return [_experiment(
            context,
            rule_id=f"{self.rule_id}.nonlinear",
            experiment_id="fallback_exp_model_family_comparison",
            priority="P2",
            evidence_strength="medium",
            name="Conservative nonlinear model-family comparison",
            hypothesis=(
                f"A `{comparison_family}` may capture stable nonlinear signal beyond "
                f"the recorded `{context.baseline_model_family}` anchor."
            ),
            change=(
                f"Compare `{comparison_family}` with `{context.baseline_model_family}` "
                "on identical features, preprocessing boundaries, folds, and metric."
            ),
            feature_inputs=(), evidence_refs=tuple(refs),
            risks=(
                "Higher-capacity models may overfit or amplify drift-sensitive features.",
            ),
            fit_scope="within_fold",
            model_family=comparison_family,
        )]


_RULES: tuple[StrategyRule, ...] = (
    _FeatureProbeRule(),
    _AblationRule(),
    _StructuralFeatureRule(),
    _GenericFeatureDiagnosticsRule(),
    _RelationshipAggregationRule(),
    _PreprocessingRule(),
    _ModelingRule(),
)


class _SmallDatasetStabilityRule:
    rule_id = "validation.small_dataset_stability"
    required_evidence_refs = (
        "validation_evidence.primary_validation", "table_profiles",
    )
    priority = "P1"

    def applies(self, context: StrategyContext) -> bool:
        return bool(
            context.validation_strategy
            and context.train_rows is not None
            and context.train_rows_ref
            and context.train_rows < 5_000
        )

    def build_actions(self, context: StrategyContext) -> list[CompiledAction]:
        if context.multiple_seed_diagnostics_supported:
            action = (
                "Use the project-approved repeated or multiple-seed diagnostic in "
                "addition to the locked primary validation; never replace its folds."
            )
            reason = (
                f"The dataset has {context.train_rows} training rows and the validation "
                "policy explicitly permits multiple-seed diagnostics."
            )
        else:
            action = (
                "Treat improvements from any single fold split as provisional and "
                "require consistent direction across the locked primary folds."
            )
            reason = (
                f"The dataset has {context.train_rows} training rows; the current policy "
                "does not authorize an additional repeated or multiple-seed protocol."
            )
        return [CompiledAction(
            section_id="metric_and_validation", rule_id=self.rule_id,
            priority="P1", evidence_strength="high", action=action, reason=reason,
            evidence_refs=(
                str(context.train_rows_ref), "validation_evidence.primary_validation",
            ),
            hypothesis_categories=("validation", "metric"), confidence="high",
        )]

    def build_experiments(self, context: StrategyContext) -> list[CompiledExperiment]:
        return []


_RULES = tuple(sorted(
    (*_RULES, _SmallDatasetStabilityRule()),
    key=lambda rule: (_PRIORITY[rule.priority], rule.rule_id),
))


def compile_competition_strategy(context: StrategyContext) -> StrategyCompilation:
    actions: list[CompiledAction] = []
    experiments: list[CompiledExperiment] = []
    for rule in _RULES:
        if rule.applies(context):
            actions.extend(rule.build_actions(context))
            experiments.extend(rule.build_experiments(context))
    actions.sort(key=lambda item: (
        _PRIORITY[item.priority], _STRENGTH[item.evidence_strength], item.rule_id,
        item.action,
    ))
    experiments.sort(key=lambda item: (
        _PRIORITY[item.priority], _STRENGTH[item.evidence_strength], item.rule_id,
        item.experiment_id,
    ))
    return StrategyCompilation(tuple(actions), tuple(experiments))


def _structural_candidates(
    context: StrategyContext,
) -> list[tuple[CompiledAction, CompiledExperiment]]:
    specs: list[tuple[str, tuple[str, ...], str, str, str, str, str]] = [
        ("family_size", ("SibSp", "Parch"), "sum SibSp and Parch and add one", "per_row", "low", "Compare paired fold delta and stability.", "low_cardinality_categorical"),
        ("is_alone", ("SibSp", "Parch"), "set one when SibSp + Parch equals zero", "per_row", "low", "Check segment support and paired fold delta.", "low_cardinality_categorical"),
        ("title", ("Name",), "extract a normalized honorific token from Name", "per_row", "medium", "Check train/test category coverage and paired fold delta.", "text_code_simple"),
        ("ticket_group_size", ("Ticket",), "count each Ticket value using training-fold data only", "within_fold", "medium", "Check unseen-ticket handling and paired fold delta.", "high_cardinality_categorical"),
        ("cabin_known_and_deck", ("Cabin",), "derive cabin-known and normalized first-character deck", "per_row", "medium", "Check missingness slices, category coverage, and paired fold delta.", "text_code_simple"),
    ]
    candidates: list[tuple[CompiledAction, CompiledExperiment]] = []
    for feature_name, inputs, transform, fit_scope, leakage, diagnostic, block in specs:
        columns = [context.column(name) for name in inputs]
        if any(column is None for column in columns):
            continue
        refs = tuple(dict.fromkeys(column.evidence_ref for column in columns if column))
        conflict = context.ablation(block)
        conflict_status = str((conflict or {}).get("status") or "")
        adverse = conflict_status in {"hurt", "neutral", "unstable", "not_better"}
        rule_id = f"feature.structural.{feature_name}"
        action = CompiledAction(
            section_id="feature_priorities",
            rule_id=rule_id,
            priority="P2" if adverse else "P1",
            evidence_strength="low" if adverse else "medium",
            action=f"Create `{feature_name}` from {', '.join(f'`{name}`' for name in inputs)} as an isolated deterministic feature experiment.",
            reason=(
                f"The required input columns exist, but the related `{block}` ablation status={conflict_status}; use a control experiment."
                if adverse else
                "The inferred schema contains every required input column; usefulness remains a validation hypothesis."
            ),
            evidence_refs=refs,
            hypothesis_categories=("feature", "schema", "validation"),
            confidence="low" if adverse else "medium",
            feature_metadata={
                "input_columns": list(inputs),
                "deterministic_transform": transform,
                "fit_scope": fit_scope,
                "leakage_risk": leakage,
                "validation_strategy": context.validation_strategy,
                "expected_diagnostic": diagnostic,
            },
        )
        experiment = _experiment(
            context,
            rule_id=rule_id,
            experiment_id=f"fallback_exp_{feature_name}",
            priority=action.priority,
            evidence_strength=action.evidence_strength,
            name=f"{feature_name} isolated ablation",
            hypothesis=f"The deterministic `{feature_name}` representation adds stable signal beyond its raw inputs.",
            change=f"Add only `{feature_name}` ({transform}); keep raw inputs, model, folds, and all other preprocessing fixed.",
            feature_inputs=inputs,
            evidence_refs=refs,
            risks=(
                "Derived group statistics can leak validation information unless fitted inside each fold."
                if fit_scope == "within_fold" else
                "The derived representation may be redundant or fold-unstable."
            ,),
            fit_scope=fit_scope,
        )
        candidates.append((action, experiment))

    age = context.column("Age")
    age_ref = context.diagnostic_ref("Age", "missingness_diagnostics")
    if age and age_ref:
        rule_id = "feature.structural.age_imputation"
        refs = (age.evidence_ref, age_ref)
        action = CompiledAction(
            section_id="feature_priorities", rule_id=rule_id, priority="P1",
            evidence_strength="high",
            action="Impute `Age` inside each training fold and retain an explicit missingness indicator.",
            reason="EDA recorded missing Age values; fold-scoped imputation prevents validation leakage.",
            evidence_refs=refs, hypothesis_categories=("feature", "schema", "validation"),
            confidence="high",
            feature_metadata={
                "input_columns": [age.name],
                "deterministic_transform": "fit the imputer on each training fold and add an Age-is-missing indicator",
                "fit_scope": "within_fold", "leakage_risk": "medium",
                "validation_strategy": context.validation_strategy,
                "expected_diagnostic": "Report paired fold delta and missing-Age slice behavior.",
            },
        )
        candidates.append((action, _experiment(
            context, rule_id=rule_id, experiment_id="fallback_exp_age_imputation",
            priority="P1", evidence_strength="high", name="Fold-safe Age imputation",
            hypothesis="Fold-fitted Age imputation plus a missingness indicator is more robust than a global or implicit fill.",
            change="Fit Age imputation inside every training fold; compare with the unchanged missing-value baseline.",
            feature_inputs=(age.name,), evidence_refs=refs,
            risks=("Global imputation would leak validation-distribution information.",),
            fit_scope="within_fold",
        )))

    fare = context.column("Fare")
    pclass = context.column("Pclass")
    fare_diagnostic_ref = context.diagnostic_ref(
        "Fare", "numeric_feature_diagnostics"
    )
    if fare and pclass and fare_diagnostic_ref:
        rule_id = "feature.structural.fare_transforms"
        refs = (fare.evidence_ref, pclass.evidence_ref, fare_diagnostic_ref)
        action = CompiledAction(
            section_id="feature_priorities", rule_id=rule_id, priority="P2",
            evidence_strength="medium",
            action="Test `Fare` log and fold-fitted within-`Pclass` representations as a controlled ablation.",
            reason=(
                "The actual Fare and Pclass columns and Fare numeric diagnostic are "
                "available; no benefit is assumed before paired validation."
            ),
            evidence_refs=refs,
            hypothesis_categories=("feature", "schema", "validation"),
            confidence="medium",
            feature_metadata={
                "input_columns": [fare.name, pclass.name],
                "deterministic_transform": (
                    "compute log1p(Fare) per row and fit within-Pclass Fare location/scale "
                    "statistics on each training fold"
                ),
                "fit_scope": "within_fold", "leakage_risk": "medium",
                "validation_strategy": context.validation_strategy,
                "expected_diagnostic": (
                    "Compare raw Fare, log1p Fare, and within-Pclass Fare using paired "
                    "fold deltas and stability."
                ),
            },
        )
        candidates.append((action, _experiment(
            context, rule_id=rule_id, experiment_id="fallback_exp_fare_transforms",
            priority="P2", evidence_strength="medium",
            name="Fare representation ablation",
            hypothesis=(
                "A log or fold-fitted within-Pclass Fare representation may be more "
                "stable than raw Fare alone."
            ),
            change=(
                "Compare raw Fare with log1p(Fare) and a within-fold, within-Pclass "
                "normalized Fare arm; change no other feature or model setting."
            ),
            feature_inputs=(fare.name, pclass.name), evidence_refs=refs,
            risks=(
                "Within-group statistics leak validation information if computed before fold splitting.",
            ),
            fit_scope="within_fold",
        )))
    return candidates


def _generic_diagnostic_candidates(
    context: StrategyContext,
) -> list[tuple[CompiledAction, CompiledExperiment]]:
    candidates: list[tuple[CompiledAction, CompiledExperiment]] = []

    def add(
        *,
        key: str,
        column_name: str,
        ref: str,
        transform: str,
        fit_scope: str,
        leakage_risk: str,
        diagnostic: str,
        priority: str = "P2",
    ) -> None:
        column = context.column(column_name)
        if column is None or not _resolves(context, ref):
            return
        rule_id = f"feature.diagnostics.{key}.{_safe_id(column.name)}"
        refs = tuple(dict.fromkeys((column.evidence_ref, ref)))
        action = CompiledAction(
            section_id="feature_priorities", rule_id=rule_id,
            priority=priority, evidence_strength="high",
            action=(
                f"Test `{key}_{_safe_id(column.name)}` from `{column.name}` as an "
                "isolated diagnostic feature."
            ),
            reason=(
                "The feature diagnostic identifies this concrete column and supports "
                "the proposed representation; validation benefit is not assumed."
            ),
            evidence_refs=refs,
            hypothesis_categories=("feature", "schema", "validation"),
            confidence="medium",
            feature_metadata={
                "input_columns": [column.name],
                "deterministic_transform": transform,
                "fit_scope": fit_scope,
                "leakage_risk": leakage_risk,
                "validation_strategy": context.validation_strategy,
                "expected_diagnostic": diagnostic,
            },
        )
        experiment = _experiment(
            context, rule_id=rule_id,
            experiment_id=f"fallback_exp_{key}_{_safe_id(column.name)}",
            priority=priority, evidence_strength="high",
            name=f"{column.name} {key.replace('_', ' ')} ablation",
            hypothesis=(
                f"The deterministic `{key}` representation of `{column.name}` may "
                "add stable signal beyond the unchanged baseline."
            ),
            change=(
                f"Add only the `{key}` representation of `{column.name}` ({transform}); "
                "hold folds, model, and all other inputs fixed."
            ),
            feature_inputs=(column.name,), evidence_refs=refs,
            risks=(
                "The representation may be redundant, sparse, or fold-unstable.",
            ),
            fit_scope=fit_scope,
        )
        candidates.append((action, experiment))

    missing = _mapping(
        context.feature_diagnostics.get("missingness_diagnostics")
    )
    if missing.get("recommended_indicators"):
        missing_collection = "recommended_indicators"
        missing_rows = list(enumerate(missing[missing_collection]))
    else:
        missing_collection = "columns"
        missing_rows = [
            (index, item)
            for index, item in enumerate(missing.get("columns") or [])
            if isinstance(_mapping(item).get("missing_pct"), (int, float))
            and float(_mapping(item)["missing_pct"]) >= 0.05
        ]
    for index, raw in missing_rows[:4]:
        name = str(_mapping(raw).get("column") or "").strip()
        add(
            key="missing_indicator", column_name=name,
            ref=f"feature_diagnostics.missingness_diagnostics.{missing_collection}[{index}]",
            transform="set one when the source value is missing, otherwise zero",
            fit_scope="per_row", leakage_risk="low",
            diagnostic="Compare paired fold delta and missing/non-missing slice behavior.",
        )

    categorical = _mapping(
        context.feature_diagnostics.get("categorical_feature_diagnostics")
    )
    if categorical.get("high_cardinality_candidates"):
        high_collection = "high_cardinality_candidates"
        high_cardinality = list(enumerate(categorical[high_collection]))
    else:
        high_collection = "columns"
        high_cardinality = [
            (index, item)
            for index, item in enumerate(categorical.get("columns") or [])
            if str(_mapping(item).get("feature_value_type") or "")
            in {"high_cardinality_categorical", "code_like", "mixed_text_code"}
        ]
    for index, raw in high_cardinality[:4]:
        name = str(_mapping(raw).get("column") or "").strip()
        add(
            key="fold_frequency", column_name=name,
            ref=f"feature_diagnostics.categorical_feature_diagnostics.{high_collection}[{index}]",
            transform=(
                "fit category frequencies on the training fold, map validation/test "
                "values, and reserve an unknown bucket"
            ),
            fit_scope="within_fold", leakage_risk="medium",
            diagnostic="Report unseen-category coverage, paired fold delta, and fold stability.",
        )

    text = _mapping(context.feature_diagnostics.get("text_feature_diagnostics"))
    for index, raw in enumerate((text.get("columns") or [])[:4]):
        name = str(_mapping(raw).get("column") or "").strip()
        add(
            key="text_shape", column_name=name,
            ref=f"feature_diagnostics.text_feature_diagnostics.columns[{index}]",
            transform=(
                "derive string length, token count, digit count, punctuation count, "
                "and missingness without using token identity"
            ),
            fit_scope="per_row", leakage_risk="low",
            diagnostic="Compare structural-only text summaries with the unchanged baseline.",
        )

    dates = _mapping(context.feature_diagnostics.get("date_time_diagnostics"))
    for index, raw in enumerate((dates.get("columns") or [])[:4]):
        name = str(_mapping(raw).get("column") or "").strip()
        add(
            key="date_parts", column_name=name,
            ref=f"feature_diagnostics.date_time_diagnostics.columns[{index}]",
            transform=(
                "parse with the EDA-detected format and derive calendar components "
                "without using future rows"
            ),
            fit_scope="per_row", leakage_risk="medium",
            diagnostic="Check parse coverage, temporal slices, and paired fold delta.",
        )

    unique: dict[str, tuple[CompiledAction, CompiledExperiment]] = {}
    for action, experiment in candidates:
        unique.setdefault(experiment.experiment_id, (action, experiment))
    return list(unique.values())


def _relationship_candidates(
    context: StrategyContext,
) -> list[tuple[CompiledAction, CompiledExperiment]]:
    candidates: list[tuple[CompiledAction, CompiledExperiment]] = []
    relationships = context.relationship_evidence.get("relationships") or []
    for index, raw in enumerate(relationships):
        relationship = _mapping(raw)
        if relationship.get("requires_aggregation") is not True:
            continue
        join_key = str(relationship.get("selected_join_key") or "").strip()
        table = str(relationship.get("table") or "").strip()
        column = context.column(join_key)
        ref = f"relationship_evidence.relationships[{index}]"
        if not join_key or not table or column is None or not _resolves(context, ref):
            continue
        slug = _safe_id(table)
        rule_id = f"feature.relationship_aggregation.{slug}"
        refs = (column.evidence_ref, ref)
        action = CompiledAction(
            section_id="feature_priorities", rule_id=rule_id,
            priority="P1", evidence_strength=str(
                relationship.get("confidence") or "medium"
            ) if relationship.get("confidence") in {"low", "medium", "high"} else "medium",
            action=(
                f"Aggregate secondary table `{table}` by `{join_key}` before joining it "
                "to the base table; never direct-join multiplying rows."
            ),
            reason=(
                f"Relationship evidence reports {relationship.get('relationship_type')} "
                "cardinality and requires aggregation."
            ),
            evidence_refs=refs,
            hypothesis_categories=("feature", "schema", "leakage", "validation"),
            confidence=str(relationship.get("confidence") or "medium")
            if relationship.get("confidence") in {"low", "medium", "high"}
            else "medium",
            feature_metadata={
                "input_columns": [join_key],
                "deterministic_transform": (
                    f"group `{table}` by `{join_key}` and derive row-count summaries; "
                    "fit any value summaries within each training fold"
                ),
                "fit_scope": "within_fold", "leakage_risk": "medium",
                "validation_strategy": context.validation_strategy,
                "expected_diagnostic": (
                    "Verify base-row count preservation, join coverage, paired fold delta, "
                    "and no leakage-check regression."
                ),
            },
        )
        experiment = _experiment(
            context, rule_id=rule_id,
            experiment_id=f"fallback_exp_secondary_{slug}",
            priority="P1", evidence_strength=action.evidence_strength,
            name=f"Aggregated {table} relationship features",
            hypothesis=(
                f"Fold-safe aggregates from `{table}` add stable signal without "
                "multiplying base rows."
            ),
            change=(
                f"Add only row-count aggregates from `{table}` grouped by `{join_key}`; "
                "retain identical folds, model, and base features."
            ),
            feature_inputs=(join_key,), evidence_refs=refs,
            risks=(
                "A direct join can multiply rows, and global value aggregates can cross validation boundaries.",
            ),
            fit_scope="within_fold",
        )
        candidates.append((action, experiment))
    return candidates


def _experiment(
    context: StrategyContext,
    *,
    rule_id: str,
    experiment_id: str,
    priority: str,
    evidence_strength: str,
    name: str,
    hypothesis: str,
    change: str,
    feature_inputs: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    risks: tuple[str, ...],
    fit_scope: str,
    model_family: str | None = None,
) -> CompiledExperiment:
    direction = (
        "increase" if context.greater_is_better is True else
        "decrease" if context.greater_is_better is False else
        "improve in the registered metric direction"
    )
    return CompiledExperiment(
        rule_id=rule_id,
        experiment_id=experiment_id,
        priority=priority,
        evidence_strength=evidence_strength,
        name=name,
        hypothesis=hypothesis,
        change=change,
        feature_inputs=feature_inputs,
        model_family=str(model_family or context.model_family),
        validation_strategy=context.validation_strategy,
        success_metric=context.metric_name,
        acceptance_rule=(
            f"Adopt only if paired OOF {context.metric_name} results {direction} and the direction is stable across folds; otherwise retain the comparison anchor."
        ),
        evidence_refs=tuple(ref for ref in evidence_refs if _resolves(context, ref)),
        hypothesis_categories=("feature", "validation", "baseline"),
        risks=risks,
        fit_scope=fit_scope,
    )


def _collect_columns(evidence_pack: Mapping[str, Any]) -> tuple[ColumnEvidence, ...]:
    found: dict[str, ColumnEvidence] = {}

    def add(name: Any, ref: str, role: Any = None, dtype: Any = None) -> None:
        value = str(name or "").strip()
        if not value:
            return
        key = value.casefold()
        candidate = ColumnEvidence(value, ref, str(role) if role else None, str(dtype) if dtype else None)
        if key not in found or _column_ref_rank(ref) < _column_ref_rank(found[key].evidence_ref):
            found[key] = candidate

    schema = _mapping(evidence_pack.get("inferred_schema"))
    for field, role in (("target_column", "target"), ("primary_id_column", "primary_id"), ("prediction_column", "prediction")):
        if schema.get(field):
            add(schema[field], f"inferred_schema.{field}", role)
    for table_index, raw_table in enumerate(schema.get("tables") or []):
        table = _mapping(raw_table)
        roles = {
            str(_mapping(item).get("name")): _mapping(item).get("role")
            for item in table.get("column_roles") or []
        }
        for column_index, raw_column in enumerate(table.get("columns") or []):
            column = _mapping(raw_column)
            add(
                column.get("name"),
                f"inferred_schema.tables[{table_index}].columns[{column_index}].name",
                roles.get(str(column.get("name"))),
                column.get("dtype"),
            )
    global_roles = _mapping(schema.get("global_roles"))
    for index, name in enumerate(global_roles.get("all_columns") or []):
        add(name, f"inferred_schema.global_roles.all_columns[{index}]")
    diagnostics = _mapping(evidence_pack.get("feature_diagnostics"))
    for index, name in enumerate(diagnostics.get("safe_feature_columns") or []):
        add(name, f"feature_diagnostics.safe_feature_columns[{index}]", "feature")
    for profile_index, raw_profile in enumerate(evidence_pack.get("table_profiles") or []):
        profile = _mapping(raw_profile)
        for column_index, raw_column in enumerate(profile.get("columns") or []):
            column = _mapping(raw_column)
            add(
                column.get("name"),
                f"table_profiles[{profile_index}].columns[{column_index}].name",
                dtype=column.get("dtype"),
            )
    return tuple(sorted(found.values(), key=lambda item: item.name.casefold()))


def _columns_by_role(
    columns: tuple[ColumnEvidence, ...],
) -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        role = column.role or "unassigned"
        grouped.setdefault(role, []).append(column.name)
    return {
        role: tuple(sorted(names, key=str.casefold))
        for role, names in sorted(grouped.items())
    }


def _train_row_evidence(
    evidence_pack: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    train_path = str(schema.get("train_base_table") or "")
    profiles = evidence_pack.get("table_profiles") or []
    for index, raw_profile in enumerate(profiles):
        profile = _mapping(raw_profile)
        path = str(profile.get("path") or "")
        if train_path and path != train_path:
            continue
        rows = profile.get("n_rows")
        if isinstance(rows, int) and rows >= 0:
            return rows, f"table_profiles[{index}].n_rows"
    for index, raw_profile in enumerate(profiles):
        rows = _mapping(raw_profile).get("n_rows")
        if isinstance(rows, int) and rows >= 0:
            return rows, f"table_profiles[{index}].n_rows"
    return None, None


def _diagnostic_columns(context: StrategyContext, family: str) -> list[tuple[str, str]]:
    rows = _mapping(context.feature_diagnostics.get(family)).get("columns") or []
    result: list[tuple[str, str]] = []
    for index, raw in enumerate(rows):
        name = str(_mapping(raw).get("column") or "").strip()
        if name and context.column(name):
            result.append((name, f"feature_diagnostics.{family}.columns[{index}]"))
    return result


def _probe_conflict(context: StrategyContext, family: str) -> bool:
    finding = context.ablation(_probe_ablation_block(family))
    return str((finding or {}).get("status") or "") in {
        "hurt", "neutral", "unstable", "not_better",
    }


def _probe_has_concrete_non_conflicting_ablation(
    context: StrategyContext,
    family: str,
) -> bool:
    finding = context.ablation(_probe_ablation_block(family))
    status = str((finding or {}).get("status") or "")
    return bool(finding and status not in {
        "", "not_testable", "hurt", "neutral", "unstable", "not_better",
    })


def _probe_ablation_block(family: str) -> str:
    return {
        "categorical_encoding": "low_cardinality_categorical",
        "high_cardinality_categorical": "high_cardinality_categorical",
        "text_features": "text_code_simple",
        "missingness_indicators": "missingness_indicators",
    }.get(family, family)


def _resolves(context: StrategyContext, ref: str) -> bool:
    try:
        resolve_evidence_ref(context.evidence_pack, ref)
    except EvidencePathResolutionError:
        return False
    return True


def _column_ref_rank(ref: str) -> int:
    if ref.startswith("inferred_schema.tables"):
        return 0
    if ref.startswith("feature_diagnostics"):
        return 1
    if ref.startswith("table_profiles"):
        return 2
    return 3


def _registry_model_family(task_type: str) -> str | None:
    normalized = task_type.strip().casefold()
    if normalized in {"binary_classification", "multiclass_classification", "classification"}:
        return "LogisticRegression/HistGradientBoostingClassifier"
    if normalized == "regression":
        return "LinearRegression/HistGradientBoostingRegressor"
    return None


def _nonlinear_model_family(task_type: str) -> str | None:
    normalized = task_type.strip().casefold()
    if normalized in {"binary_classification", "multiclass_classification", "classification"}:
        return "HistGradientBoostingClassifier"
    if normalized == "regression":
        return "HistGradientBoostingRegressor"
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_id(value: str) -> str:
    return "_".join(part for part in "".join(
        character if character.isalnum() else " " for character in value.casefold()
    ).split() if part) or "unknown"


__all__ = [
    "ColumnEvidence", "CompiledAction", "CompiledExperiment", "StrategyCompilation",
    "StrategyContext", "StrategyRule", "compile_competition_strategy",
]
