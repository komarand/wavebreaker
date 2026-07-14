# Contract Topology Audit

Audit date: 2026-07-13. This inventory records the topology found before the
canonical package migration and the compatibility state after the foundation
work. Canonical means an inter-stage artifact model, not an LLM draft model or
an EDA-internal diagnostic type.

| Canonical concept | Definitions found | Producer | Consumers | Artifact | Mismatch / namespace / migration |
| --- | --- | --- | --- | --- | --- |
| `ResearchHypotheses` / `ResearchHypothesis` | Canonical: `contracts/research.py`. Compatibility re-export: `contracts/research_hypotheses.py`, `eda/schemas.py`. Draft input types remain in `research_scout_schemas.py` and `research_scout/schemas.py`. | Research Scout | EDA, Final Strategy | `research/research_hypotheses.json` | Legacy `id`, plural/category aliases, statement aliases, and missing confidence migrate explicitly. Hypothesis IDs use the hypothesis namespace. |
| `EdaTaskPlan` / `EdaTask` / `HypothesisIndexEntry` | Canonical: `contracts/eda.py`. Compatibility re-export: `contracts/eda_task_plan.py`, `eda/schemas.py`. `EdaTaskPlanDraft` and `ScoutEdaTask` are producer drafts. | Research Scout | EDA | `research/eda_task_plan.json` | Legacy `id`, object index entries, and category aliases migrate. Task IDs, hypothesis IDs, and dependency IDs are separate. |
| `EdaEvidencePack` | Canonical: `contracts/eda.py`; `eda/schemas.py` re-exports the same class. EDA-internal diagnostic models remain module-owned. | EDA orchestrator | reasoning, Final Strategy, validator, renderer | `eda/eda_evidence_pack.json` | Machine evidence is separate from Markdown. `experiment_candidates`, `eda_strategy_hints`, and `eda_risk_register` are deprecated projections. Exact dictionary paths and registered semantic list paths are evidence addresses. |
| `ValidationResult` / `ValidationPolicy` | Canonical: `contracts/validation.py`; `schemas.py` re-exports the same class. | Validation Architect | Experiment Planner, Leaderboard Auditor, Final Strategy | `reasoning/validation_result.json` | Primary is required; secondary is nullable. Collections normalize `null` only on the explicit allowlist. Evidence IDs use the evidence namespace. |
| `ExperimentPlan` / `ExperimentItem` | Canonical: `contracts/experiments.py`; `schemas.py` re-exports `ExperimentItem`. | Experiment Planner | reviewer, Final Strategy | `reasoning/experiment_plan.json` | Existing list artifacts load through a compatibility adapter into the versioned wrapper. `experiment_id`, `source_hypothesis_ids`, `evidence_ids`, and dependency IDs have distinct types. Raw planner items may lack identity; the serialized plan may not. |
| `SkepticalReview` / `ReviewResult` | Canonical: `contracts/review.py`; `schemas.py` re-exports `ReviewResult`. | Skeptical Reviewer | Final Strategy | `reasoning/skeptical_review.json` | Reviewed/approved/rejected values are experiment IDs. Approved and rejected sets cannot overlap. Revised sections cannot introduce unknown IDs at bundle validation. |
| `FinalStrategyResult` | Canonical: `contracts/final_strategy.py`; `reasoning/final_synthesizer.py` imports and re-exports it. | Final Synthesizer | renderer, artifact validator | `final/final_strategy.json` | Evidence, EDA result, source, hypothesis, and experiment fields have documented distinct namespaces. Context labels are forbidden evidence. One bounded reference repair is allowed. |
| Evidence registry/resolver | `contracts/evidence.py` implementation; canonical facade and typed refs in `contracts/references.py`. | EDA and reasoning boundary preparation | planner, synthesizer, validator | derived, not a standalone artifact | No recursive/fuzzy/substring search. Semantic lists require a registered path and identity field and exactly one match. |
| Typed registries | Canonical registries in `contracts/registries.py`; compatibility experiment/evidence registries remain in `contracts/experiments.py` and `contracts/evidence.py` for migrated callers. | boundary adapters | prompts and cross-artifact validators | derived | Separate mappings prevent one untyped `set[str]` from authorizing unrelated namespaces. |
| Run manifest | Canonical: `contracts/manifest.py`; `contracts/pipeline.py` re-exports it. The orchestrator still builds a dictionary before boundary validation. | full-run orchestrator | resume, CLI, diagnostics | `run_manifest.json` | Stage IDs and artifact paths remain distinct. Manifest errors can embed structured contract issues. |
| Stage results | Canonical frozen dataclasses in `contracts/artifacts.py`. `FullRunResult` remains the CLI-facing orchestration result. | stage orchestrators | full-run orchestration | not serialized directly | Generic orchestration dictionaries remain as a compatibility implementation detail; canonical artifacts loaded into them are typed models. |

## Field-level audit

| Field | Canonical namespace / semantics | Historical mismatch |
| --- | --- | --- |
| `hypothesis_id`, `source_hypothesis_ids`, `hypothesis_ids` | hypothesis | Used as experiment identity in Final Strategy. |
| `task_id`, task dependencies, `hypothesis_index` values | EDA task | Legacy producer emitted `id`; index could be an object. |
| `experiment_id`, `experiment_ids`, reviewer decision IDs | experiment | Hypothesis IDs appeared in experiment decision fields. |
| `evidence_id`, `evidence_ids`, `evidence_refs` | global/allowed evidence | Reasoning invented IDs; `validation_policy` required an explicit alias; context labels appeared as evidence. |
| `eda_result_refs` | exact address inside `EdaEvidencePack` | Nested list-backed paths were not resolved consistently. |
| `validation_requirements` | validation requirement objects/IDs | Sometimes converted into empty experiments. |
| `safety_constraints` | safety constraint objects/IDs | Sometimes lost during final synthesis. |
| `approved_experiments` | context collection label only | Incorrectly accepted as an evidence value. |
| `experiment_candidates` | deprecated compatibility projection | Previously consumed as canonical reasoning input. |
| `testable_hypotheses` | EDA hypothesis records; identity is `hypothesis_id` | Previously mixed with planned experiments. |

## Remaining compatibility work

- Change the full-run orchestration map to stage-result dataclasses throughout;
  it currently stores validated canonical objects under established keys.
- Convert legacy list-shaped experiment artifacts to the versioned wrapper at
  the producer write point, then retire list loading.
- Remove compatibility registry implementations after all prompt builders use
  `contracts/registries.py`.
