# Research Pipeline Contracts

The canonical runtime inventory lives in `kaggle_researcher/contracts/registry.py`.
This document summarizes the producer-consumer boundaries that the contract test
suite exercises. Pytest remains the source of truth.

| Contract | Producer | Consumers | Version | Nullable fields | Reference fields | Migration |
| --- | --- | --- | --- | --- | --- | --- |
| `ResearchHypotheses` | Research Scout | EDA, Final Strategy | 1.0 | `created_at`, hypothesis rationale | hypothesis IDs, source refs | unversioned legacy to 1.0 |
| `EdaTaskPlan` | Research Scout | EDA | 1.0 | `task_type` | task IDs, hypothesis IDs, hypothesis index | unversioned legacy to 1.0 |
| `EdaEvidencePack` | EDA Engine | reasoning, Final Strategy, validator | 1.0 | module-specific values | evidence refs, source refs, hypothesis refs | none |
| `ValidationResult` | Validation Architect | planner, auditor, synthesis | internal | `secondary_validation` | evidence IDs | none |
| `MetricResult` | Metric Specialist | planner, synthesis | internal | none | evidence IDs | null collections only |
| `LeakageRiskResult` | Leakage Risk Analyst | planner, synthesis | internal | none | evidence IDs | null collections only |
| `LeaderboardAuditResult` | Leaderboard Auditor | synthesis | internal | none | evidence IDs | null collections only |
| `ExperimentItem[]` | Experiment Planner | reviewer, synthesis | internal | none after planner canonicalization | experiment IDs, evidence IDs | explicit evidence aliases |
| `ReviewResult` | Skeptical Reviewer | synthesis | internal | none | evidence IDs, approved/rejected experiment IDs | null collections only |
| `FinalStrategyResult` | Final Strategy | report, validator | 1.0 | task type, recommended validation | evidence, hypothesis, source, experiment IDs | deterministic repair then fallback |
| run manifest | full-run orchestrator | resume, summary | 1.0 | stage error/timestamps | stage IDs and artifact paths | none |
| final report | report renderer | validator, human reader | n/a | n/a | rendered from validated strategy | none |

## ID Namespaces

- `hypothesis`: created by Research Scout and consumed by EDA and Final Strategy.
- `eda_task`: created in `EdaTaskPlan`; task and index references are bundle-validated.
- `eda_evidence`: top-level evidence IDs and exact nested paths from the serialized EDA pack.
- `reasoning_evidence`: canonical `validation_result`, `leakage_result`, and `metric_result` IDs.
- `source_claim`: retrieved-document and source-claim IDs. These never resolve as EDA evidence merely because the strings collide.
- `experiment`: assigned deterministically at the Experiment Planner boundary and consumed by reviewer/final strategy decisions.
- `risk`, `stage`, and module-local IDs remain in their owning artifacts and are not merged into the evidence namespace.

`EvidenceRegistry` performs exact lookup in explicitly allowed namespaces. Approved
nested paths such as `baseline_evidence.metric_value` and
`validation_evidence.primary_validation` resolve only when the serialized path exists.
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

## Repair Policy

Reasoning responses receive at most one schema repair attempt. Experiment evidence
references receive at most one bounded repair using the supplied allowlist. Final
strategy payloads use the existing deterministic repair and evidence-derived fallback.
Contract errors expose the stage, field paths, invalid IDs, recovery status, and a
suggested rerun stage; console summaries are bounded while full validation details
remain available on the exception and in logs.

## Pipeline Invariants

- Scout hypotheses and EDA tasks use the same canonical classes in both packages.
- Every task/index cross-reference resolves before EDA runs.
- Required reasoning objects remain non-null; optional secondary validation may be absent.
- Planner evidence IDs and reviewer experiment decisions resolve in typed namespaces.
- Final Strategy references existing evidence, hypotheses, sources, and experiments and cannot restore reviewer-rejected experiments.
- The final report is non-empty and derives from a schema-valid final strategy.
- A reusable Scout or EDA stage must have semantically valid artifacts, not merely existing files.
- Forcing a stage invalidates every transitive dependent from the canonical stage registry.

## Running Contract Checks

```powershell
$py = "E:\wavebreaker\.venv-win\Scripts\python.exe"

& $py -m pytest -q -m contract
& $py -m pytest -q -m pipeline_smoke
& $py -m pytest -q -m "contract or pipeline_smoke"
```

These checks are deterministic and require no network, Kaggle credentials, database,
GPU, embeddings, or live LLM calls.
