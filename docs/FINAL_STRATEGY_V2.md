# Final Strategy v2

Final Strategy schema `2.0` separates the LLM-facing draft from the canonical
machine artifact. The model returns `FinalStrategyDraft` fields such as
`narrative`, `support_refs`, and `evidence_summary_refs`; the deterministic draft
compiler resolves those typed references and emits only fields accepted by
`FinalStrategyResult`.

The adapter canonicalizes known representation aliases (including Unicode dash
variants in `evidence_origin`) but does not relax strict validation. Unknown
evidence, source, hypothesis, experiment, risk, validation-requirement, and safety
IDs remain invalid.

## Canonical changes

- Evidence previews live once in top-level `evidence_catalog`.
- Legacy action `evidence_bindings` are accepted on read and excluded on write.
- Actions use `primary_evidence_refs`, bounded `evidence_refs`, optional
  `limitation_evidence_refs`, and semantic hypothesis roles.
- `feature_experiment_families` stores controlled multi-arm feature tests.
- `core_experiments` and `experiment_backlog` are derived from one ranked budget.
- `experiment_budget`, `quality_metrics`, and `diagnostics_summary` make selection
  and final quality gates inspectable.
- Model comparisons store canonical family and implementation IDs from the typed
  deterministic registry.

`upgrade_final_strategy_v1_to_v2` reads a v1 artifact, ignores serialized legacy
preview duplication, rebuilds the evidence catalog from the validated EDA pack,
assigns semantic roles, groups supported feature families, and reapplies the
experiment budget. It never fabricates source provenance or metric values.

## Budget configuration

The defaults are eight core experiments, four during the first 24 hours, eight
during the first 48 hours, and at most two high-cost experiments. Override them
with:

- `FINAL_STRATEGY_MAX_CORE_EXPERIMENTS`
- `FINAL_STRATEGY_MAX_FIRST_24H_EXPERIMENTS`
- `FINAL_STRATEGY_MAX_FIRST_48H_EXPERIMENTS`
- `FINAL_STRATEGY_MAX_HIGH_COST_EXPERIMENTS`

Baseline reproduction is pinned first when completed baseline evidence exists.
A distinct supported model comparison, OOF-only threshold postprocessing when the
metric requires it, and submission integrity checks are dependency ordered near
the end. Remaining valid experiments stay visible in the backlog.
