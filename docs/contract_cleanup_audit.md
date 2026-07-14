# Contract-system cleanup audit

This audit records the authoritative post-cleanup boundaries. The contract test
suite is executable evidence; this document names the remaining compatibility
surfaces instead of treating them as canonical APIs.

## Typed orchestration state

`FullRunState` owns the manifest and typed `ResearchStageResult`,
`EdaStageResult`, `ReasoningStageResult`, and `FinalStageResult` values. Stage
functions accept `(stage_id, state)` and obtain dependencies through typed
accessors that raise `MissingStageDependencyError`. Runtime clients and config
are isolated in `RuntimeServices`; they are not serialized into stage contracts.

| Former context key family | Canonical replacement |
| --- | --- |
| `hypotheses`, `task_plan`, `plan_data`, documents | `ResearchStageResult` |
| EDA pack, summary, artifact paths | `EdaStageResult` |
| metric, validation, leakage, leaderboard, plan, review | `ReasoningStageResult` |
| strategy and report paths | `FinalStageResult` |
| arbitrary final-synthesis map | `FinalSynthesisContext` |

Resume follows one lifecycle: load and migrate `run_manifest.json`, verify each
declared artifact pointer (relative path, contract family, size and SHA-256),
load it through its canonical boundary adapter, mark invalid stages stale, then
invalidate transitive dependents. No latest-directory or inferred-file fallback
is authoritative.

## Registry ownership

`contracts/registries.py` owns hypothesis, EDA-task, experiment, review, risk,
validation-requirement, and safety-constraint registries plus
`ContractRegistries`. `contracts/evidence.py` owns `EvidenceRegistry`.
`contracts/references.py` re-exports that evidence type; the dynamic
`ExperimentRegistry` export in `contracts/experiments.py` is compatibility only.
Production synthesis and pipeline validation use `build_contract_registries`.

Legacy flat resolver helpers in `contracts/experiments.py` remain for external
and old test callers, but no production caller uses them. They may be removed in
a later breaking release after downstream imports have migrated.

## Final Strategy namespaces

Risks, validation requirements, and safety constraints are distinct producer-
owned namespaces. Actions reference them through `risk_ids`,
`validation_requirement_ids`, and `safety_constraint_ids`. Global acknowledgments
use `acknowledged_risk_ids`, `selected_validation_requirement_ids`, and
`enforced_safety_constraint_ids`. Cross-namespace and unknown IDs fail.
Critical risks, mandatory validation requirements, and blocking safety
constraints must be represented structurally; prose alone is insufficient.

The legacy generic constraint/reference payload migrates only when its key and
value map exactly to one namespace. Ambiguous values and unknown IDs fail rather
than being guessed.

## Manifest versions and migrations

The canonical `RunManifest` family/version is `run_manifest`/`1.0`. Supported
inputs are current 1.0 and recognized unversioned manifests. Migrations normalize
documented stage/status aliases, typed errors, artifact pointers, final outputs,
and paths inside the run directory. The first migration preserves
`run_manifest.legacy.json`; the canonical replacement is atomic. Future versions,
unknown aliases, traversal, external paths, missing files, size mismatches, and
hash mismatches fail explicitly. Re-loading canonical output is idempotent.

## Compatibility and deferred boundaries

`experiment_candidates`, `eda_strategy_hints`, and `eda_risk_register` remain
serialized for legacy readers. Human-only summaries may read those projections
as fallbacks. Canonical reasoning registries and synthesis contexts do not.

EDA safety constraints, validation requirements, testable hypotheses, and risks
are typed. Some module-specific diagnostic payloads remain dictionaries because
their shapes are heterogeneous and opaque to orchestration; promoting every
diagnostic family is deferred. Metric, leakage, and leaderboard results remain
separate typed families because they have distinct producers and semantics; they
are composed in `ReasoningStageResult`, not merged into one generic result map.

## Verification

Run `pytest -q -m contract`, `pytest -q -m manifest_migration`, and `pytest -q`.
The static checks in the migration suite reject generic orchestration context
lookups and duplicate registry class definitions outside approved modules.
