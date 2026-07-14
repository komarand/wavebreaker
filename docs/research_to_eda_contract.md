# Research Reasoner → EDA boundary contract

This document records the implemented, deterministic contract between published Research
Scout artifacts and the EDA orchestrator. The source of truth is
`kaggle_researcher/contracts/research_to_eda.py`; validation performs no dataset reads,
network calls, LLM calls, or EDA execution.

## Canonical boundary models

All canonical models inherit `ContractModel` and therefore use `extra="forbid"`, assignment
validation, whitespace stripping, and isolated `default_factory` collections.

| Model | Actual public fields |
|---|---|
| `ResearchHypothesis` | `hypothesis_id`, `category`, `claim`, `rationale`, `expected_eda_checks`, `priority`, `confidence_before_eda`, `source_refs`, `status`, `limitations` |
| `ResearchHypotheses` | `contract_family`, `schema_version`, `competition_id`, `created_at`, `hypotheses`, `eda_tasks`, `structured_findings`, `scout_limitations`, `models_used` |
| `EdaTask` | `task_id`, `module`, `priority`, `blocking`, `related_hypothesis_ids`, `dependencies`, `expected_outputs`, `params` |
| `EdaTaskPlan` | `contract_family`, `schema_version`, `competition_id`, `task_type`, `metric`, `dataset`, `eda_tasks`, `hypothesis_index`, `recommended_module_sequence`, `recommended_human_checklist`, `blocking_tasks` |
| `EdaRunConfig` | competition/input paths, optional local dataset/output paths, resource limits, module switches, and seed/failure policy |
| `EdaRunResult` | competition/run identity, output paths, module statuses, hypothesis count, warnings, limitations, and duration |

The canonical schema version is exactly `1.0`. Missing-version legacy envelopes are handled
only by explicit pre-publication migration. Unsupported old or future versions fail. Migration
does not change semantic IDs, task type, metric, or the number of hypotheses.

## Semantic invariants

| Area | Blocking invariant | Stable issue codes |
|---|---|---|
| Identity | Both documents name the same competition | `competition_id_mismatch` |
| Hypotheses | IDs are unique and stable; schema/metric/validation/leakage each have a P0 hypothesis | `duplicate_hypothesis_id`, `unstable_hypothesis_id`, `empty_hypothesis_id_suffix`, `missing_p0_hypothesis` |
| Prefix policy | Category-prefix mismatch is currently a warning | `hypothesis_id_category_prefix_mismatch` |
| Checks | Checks exist in the local registry, suit the category, and P0 does not depend only on optional modules | `unknown_eda_check`, `duplicate_eda_check`, `empty_expected_eda_checks`, `hypothesis_check_category_mismatch`, `p0_depends_only_on_optional_module` |
| Task references | Task IDs are unique; every relation and index entry resolves in both directions | `duplicate_task_id`, `unknown_hypothesis_reference`, `unknown_hypothesis_index_key`, `unknown_task_reference`, `one_way_hypothesis_task_mapping`, `one_way_task_hypothesis_mapping`, `duplicate_related_hypothesis_id`, `duplicate_hypothesis_index_mapping`, `hypothesis_driven_task_without_hypothesis` |
| Module plan | Modules are known, blocking flags agree with the module list, P0 modules are sequenced, and dependencies are ordered | `unknown_eda_module`, `unknown_blocking_module`, `unplanned_blocking_module`, `blocking_task_conflict`, `unknown_sequence_module`, `duplicate_blocking_module`, `duplicate_module_sequence_entry`, `missing_p0_module_from_sequence`, `module_dependency_missing_from_sequence`, `module_dependency_order_violation` |
| Metric/task | Registry semantics cannot contradict task type or explicit flags | `metric_task_type_mismatch`, `metric_semantics_mismatch`, `forced_temporal_without_evidence` |
| Ranking/time | Ranking needs ranking plus query/group checks; stability metrics need temporal feasibility checks | `ranking_validation_check_missing`, `ranking_group_check_missing`, `temporal_metric_check_missing` |
| Custom metric | Unknown/custom metrics are warnings but require resolution and an explicit limitation; they cannot claim a local implementation | `unknown_metric`, `custom_metric`, `custom_metric_resolution_missing`, `custom_metric_limitation_missing`, `unknown_metric_claims_local_implementation` |
| Pre-EDA claims | Published Scout artifacts cannot claim completed EDA facts | `premature_eda_factual_claim` |
| Secrets | Dataset contracts and diagnostics cannot expose credentials or user-specific paths | `dataset_contract_contains_secret` |

Duplicate collections are rejected and are not silently deduplicated by the boundary validator.
The validator returns all deterministic issues in a structured result; `require_*` raises a
typed aggregate error before run-directory creation and dataset resolution.

## EDA plan dependency graph

```text
file_inventory → schema_inferer → table_profiler → metric_analyzer
                                             → drift_analyzer
metric_analyzer → validation_analyzer → leakage_checker
                                     → baseline_runner → baseline_ablations
                                                       → interaction_diagnostics
                                                       → slice_diagnostics
schema_inferer → relationship_inferer → feature_probe
```

Dependencies are plan checks only; no module is run by the validator.

## Task × metric semantics

| Task/metric | Required semantics |
|---|---|
| Binary/multiclass + ROC AUC/Gini | probability scores, rank-based, no threshold search |
| Binary/multiclass + LogLoss | probabilities and calibration-sensitive evaluation |
| Binary/multiclass + F1 | threshold-dependent; threshold selection belongs inside validation |
| Regression/forecasting + RMSE/MAE | regression error; no classification threshold requirement |
| Ranking + NDCG/MAP/Recall@K | ranking-aware validation and query/group integrity |
| Binary + Gini Stability | probability/rank semantics plus temporal feasibility evidence |
| Survival + concordance index | survival task contract |
| Custom/unknown | controlled warning, metric resolution check, explicit limitation, no invented implementation |

A date column alone never forces temporal validation for ordinary IID classification or
regression. No boundary rule globally requires `case_id`, `WEEK_NUM`, `date_decision`, or any
other Home Credit-specific column.

## Fixtures and file boundary

Reusable model factories live in `tests/contracts/factories.py`. Static JSON examples live in
`tests/fixtures/contracts/research_to_eda/`; they include a valid IID binary pair and an invalid
competition-mismatch pair. The wider parameterized matrix covers regression, grouped binary,
ranking, temporal stability, unknown metrics, dangling references, invalid checks/modules,
forced temporal policy, missing P0 categories, and premature claims.

JSON loaders enforce UTF-8 objects, reject corrupt JSON, arrays, and duplicate object keys, and
bound validation diagnostics. Writers use the canonical atomic JSON policy. The EDA orchestrator
loads and validates both files before creating its artifact writer or resolving a dataset.

## Compatibility decisions

- Canonical version `1.0` remains strict and does not accept unknown fields.
- A rich legacy Scout envelope is explicitly projected and adapted before publication. Known
  legacy module/check names are mapped to canonical registry entries there, not repaired by EDA.
- Missing-version legacy payloads continue through the existing explicit migration path.
- Published invalid artifacts are never repaired by the consumer.
- `hypothesis_index` is treated as an asserted bidirectional index when relations are present;
  incomplete one-way fixture mappings were corrected instead of weakening validation.
