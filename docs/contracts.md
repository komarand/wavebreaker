# Research Pipeline Contracts

The contract package is the single source of truth for serialized inter-stage
protocols. A producer may complete only after its outputs satisfy the canonical
contract. A consumer may assume canonical input but must still verify artifact
integrity and supported version.

Canonical files are split by responsibility: `base.py`, `versions.py`,
`errors.py`, and `ids.py` are dependency-free foundations; domain models live
in `research.py`, `eda.py`, `validation.py`, `experiments.py`, `review.py`, and
`final_strategy.py`; registries and addressing live in `registries.py` and
`references.py`; boundary I/O and prepared prompt inputs live in `artifacts.py`
and `synthesis_context.py`.

Every versioned JSON object includes `contract_family` and `schema_version`.
Missing versions are legacy input, `1.0` is current, and unknown future versions
fail without repair. Version strings are centralized in `versions.py`.

The canonical runtime inventory lives in `kaggle_researcher/contracts/registry.py`.
This document summarizes the producer-consumer boundaries that the contract test
suite exercises. Pytest remains the source of truth.

| Contract | Producer | Consumers | Version | Nullable fields | Reference fields | Migration |
| --- | --- | --- | --- | --- | --- | --- |
| `ResearchHypotheses` | Research Scout | EDA, Final Strategy | 1.0 | `created_at`, hypothesis rationale | hypothesis IDs, source refs | unversioned legacy to 1.0 |
| `EdaTaskPlan` | Research Scout | EDA | 1.0 | `task_type` | task IDs, hypothesis IDs, hypothesis index | unversioned legacy to 1.0 |
| `EdaEvidencePack` | EDA Engine | reasoning, Final Strategy, validator | 1.0 | module-specific values | evidence refs, source refs, hypothesis refs | none |
| `ValidationResult` | Validation Architect | planner, auditor, synthesis | 1.0 | `secondary_validation` | evidence IDs | unversioned compatibility input |
| `MetricResult` | Metric Specialist | planner, synthesis | internal | none | evidence IDs | null collections only |
| `LeakageRiskResult` | Leakage Risk Analyst | planner, synthesis | internal | none | evidence IDs | null collections only |
| `LeaderboardAuditResult` | Leaderboard Auditor | synthesis | internal | none | evidence IDs | null collections only |
| `ExperimentPlan` | Experiment Planner | reviewer, synthesis | 1.0 | none after planner canonicalization | experiment IDs, source hypothesis IDs, evidence IDs | legacy list wrapper plus explicit evidence aliases |
| `ReviewResult` | Skeptical Reviewer | synthesis | 1.0 | none | evidence IDs, reviewed/approved/rejected experiment IDs | unversioned compatibility input and null collections |
| `FinalStrategyResult` | Final Strategy | report, validator | 1.0 | task type, recommended validation | evidence, hypothesis, source, approved experiment IDs | deterministic repair then fallback |
| run manifest | full-run orchestrator | resume, summary | 1.0 | stage error/timestamps | typed stage IDs and artifact pointers | recognized unversioned manifests to 1.0 |
| final report | report renderer | validator, human reader | n/a | n/a | rendered from validated strategy | none |

## ID Namespaces

- `hypothesis`: created by Research Scout and consumed by EDA and Final Strategy.
- `eda_task`: created in `EdaTaskPlan`; task and index references are bundle-validated.
- `eda_evidence`: top-level evidence IDs and exact nested paths from the serialized EDA pack.
- `reasoning_evidence`: canonical `validation_result`, `leakage_result`, and `metric_result` IDs.
- `source_claim`: retrieved-document and source-claim IDs. These never resolve as EDA evidence merely because the strings collide.
- `experiment`: assigned deterministically at the Experiment Planner boundary. `source_hypothesis_ids` preserve provenance without reusing hypothesis identity. Reviewer and Final Strategy consume only planner IDs.
- `risk`, `validation_requirement`, and `safety_constraint`: distinct EDA-owned namespaces consumed structurally by Final Strategy.
- `stage` and module-local IDs remain in their owning artifacts and are not merged into the evidence namespace.

`EvidenceRegistry` performs exact lookup in explicitly allowed namespaces. Approved
nested paths such as `baseline_evidence.metric_value` and
`validation_evidence.primary_validation` resolve only when the serialized path exists.
List-backed semantic references are allowed only for explicitly registered collections.
For example,
`baseline_ablation_evidence.feature_block_findings.low_cardinality_categorical`
resolves by the collection's declared `feature_block` identity field. Duplicate
identities are ambiguous and fail validation; unknown identities are never guessed.
There is no edit-distance, substring, or semantic guessing.

## Normalization Policy

Safe compatibility rules are centralized in
`kaggle_researcher/contracts/normalization.py`:

- legacy `id` fields migrate to canonical hypothesis/task IDs;
- four documented legacy hypothesis categories migrate to canonical enums;
- `validation_policy`, `primary_validation`, and `validation_strategy` map to `validation_result` only at the Experiment Planner boundary;
- `null` becomes an empty list/dictionary only for an allowlisted collection field.

Normalization is deterministic and idempotent. Unknown enums and references fail.
An empty required object is not converted to `None` and still fails model validation.

Collections serialize as `[]` or `{}` when empty. Meaningful optional objects,
including `secondary_validation`, preserve JSON `null`. Canonical writers use
UTF-8, stable key ordering and indentation, fsync the temporary file, and use
an atomic Windows-compatible replace.

## Repair Policy

The shared protocol is implemented by `contracts/repair.py`: normalize, validate,
make one bounded repair with structured issues/canonical fields/allowlists, normalize
again, validate again, then fail hard. Reasoning responses receive at most one schema repair attempt. Experiment evidence
references receive at most one bounded repair using the supplied allowlist. Final
Strategy receives the exact global-evidence, EDA-evidence, and approved-experiment
allowlists in its prompt. Invalid references receive one reference-only model repair;
references that remain invalid after that attempt fail the stage.
Contract errors expose the stage, field paths, invalid IDs, recovery status, and a
suggested rerun stage; console summaries are bounded while full validation details
remain available on the exception and in logs.

## Pipeline Invariants

- Scout hypotheses and EDA tasks use the same canonical classes in both packages.
- Every task/index cross-reference resolves before EDA runs.
- Required reasoning objects remain non-null; optional secondary validation may be absent.
- Planner evidence IDs and reviewer experiment decisions resolve in typed namespaces.
- Final Strategy references existing evidence, hypotheses, sources, and experiments and cannot restore reviewer-rejected experiments.
- Final Strategy structurally acknowledges every critical risk, mandatory validation requirement, blocking safety constraint, and optional-stage failure.
- A hypothesis mistakenly placed in `experiment_ids` is repaired only when exactly one approved experiment names it in `source_hypothesis_ids`; ambiguous and missing mappings fail with `CrossNamespaceReferenceError`.
- The final report is non-empty and derives from a schema-valid final strategy.
- A reusable Scout or EDA stage must have semantically valid artifacts, not merely existing files.
- Forcing a stage invalidates every transitive dependent from the canonical stage registry.

Prepared `ExperimentPlanningContext` and `FinalSynthesisContext` models expose
exact allowlists and only the evidence, constraints, requirements, limitations,
and approved decisions required by an LLM stage. Raw artifact dumps and
deprecated projections are not authoritative prompt context.

## Producer and consumer responsibilities

Producers normalize only registered representation differences, migrate known
versions, validate canonical models and cross-references, and write atomically.
An LLM boundary may make one bounded repair attempt; repair metadata belongs in
the stage manifest. Consumers load through boundary adapters, reject unsupported
versions, and use typed registries for cross-artifact references.

Deprecated compatibility projections are `experiment_candidates`,
`eda_strategy_hints`, and `eda_risk_register`. They remain serialized while
legacy renderers migrate, but canonical reasoning contexts do not depend on them.

`RunManifest` is the only authoritative resume index. Its artifact pointers carry
relative paths, contract family, size, and SHA-256. Supported legacy manifests
are backed up and migrated atomically; unknown versions, aliases, external paths,
and integrity mismatches fail. See `contract_cleanup_audit.md` for the lifecycle
and remaining compatibility inventory.

## Running Contract Checks

```powershell
$py = "E:\wavebreaker\.venv-win\Scripts\python.exe"

& $py -m pytest -q -m contract
& $py -m pytest -q -m manifest_migration
& $py -m pytest -q -m pipeline_smoke
& $py -m pytest -q -m "contract or pipeline_smoke"
```

These checks are deterministic and require no network, Kaggle credentials, database,
GPU, embeddings, or live LLM calls.

## Final Strategy Field Namespaces

Final Strategy generation uses a separate `FinalStrategyDraft` contract before the
strict public result boundary. Draft actions express support as typed
`{namespace, ref_id}` objects and never expose raw `evidence_refs`. The temporary
legacy normalizer resolves old reference fields through `ReferenceCatalog`, preserves
first-seen order, removes duplicates, and fails on unresolved or ambiguous IDs.

P1.1 does not compile `FinalStrategyDraft` into `FinalStrategyResult`; that conversion
belongs to the Strategy Compiler boundary. The public result schema and renderer remain
unchanged.

| Field | Namespace |
| --- | --- |
| `action_id` | Final Strategy item identity |
| `experiment_ids` | reviewer-approved Experiment Planner IDs only |
| `hypothesis_ids`, `related_hypothesis_ids` | Scout or EDA hypothesis IDs |
| `evidence_refs` | exact EDA evidence paths or documented global/source evidence IDs |
| `eda_result_refs` | exact EDA evidence paths only |
| `source_refs` | retrieved source document IDs |
| `risk_ids` | EDA risk IDs |
| `validation_requirement_ids` | EDA validation-requirement IDs |
| `safety_constraint_ids` | EDA safety-constraint IDs |
| `acknowledged_risk_ids` | global EDA risk acknowledgments |
| `selected_validation_requirement_ids` | global selected validation requirements |
| `enforced_safety_constraint_ids` | global enforced safety constraints |
| `validation_strategy` | validation method enum, not an ID |

The three constraint families are never resolved through the evidence namespace.
Legacy generic references migrate only when an exact, unambiguous namespace match
exists. There is no generic `review_issue_ids` namespace.
Context collection labels such as `approved_experiments`, `experiment_plan`, and
`skeptical_review` are prompt structure, not evidence IDs. Approved experiments are
referenced through concrete `experiment_ids` only.

To rerun only final synthesis and its dependents, use the registered stage ID:

```powershell
python -m kaggle_researcher.main full-run `
  --resume-run-dir "<existing-run-dir>" `
  --force-rerun-stage final_strategy
```

This reuses semantically valid Scout, EDA, reasoning, Experiment Planner, and Skeptical
Reviewer artifacts, then reruns `final_strategy`, `final_report`, and `artifact_validation`.
