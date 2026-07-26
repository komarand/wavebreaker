# EDA Engine tasks 40–64

This archive contains 25 standalone Codex task files extracted from
`EDA_ENGINE_CODEX_TASKS_40_PLUS_GENERIC.md`.

## How to use

1. Read `00_GLOBAL_RULES.md` before starting the sequence.
2. Execute tasks in numeric order unless the repository state proves that an
   earlier task is already complete.
3. Give Codex one file from `tasks/` at a time.
4. Treat `EDA_ENGINE_SPEC.md` in the repository as the normative specification.
5. Do not use the archived v4 document to override the EDA Engine v5 contracts.

Each task file includes its original goal, file scope, Codex prompt, acceptance
criteria, and a copy of the global rules so it can be used independently.

## Task index

| No. | Task file | Goal |
|---:|---|---|
| 40 | [40_eda_validation_policy_and_split_helpers](tasks/40_eda_validation_policy_and_split_helpers.md) | Implement generic validation policy helpers instead of assuming temporal validation by default. |
| 41 | [41_eda_validation_analyzer_generic](tasks/41_eda_validation_analyzer_generic.md) | Build factual validation evidence using the generic ValidationPolicySelector. |
| 42 | [42_eda_leakage_checker_generic](tasks/42_eda_leakage_checker_generic.md) | Implement generic leakage checks for tabular competitions, not only Home Credit-like datasets. |
| 43 | [43_eda_hypothesis_evaluator_generic](tasks/43_eda_hypothesis_evaluator_generic.md) | Evaluate Research Scout hypotheses against generic EDA evidence. |
| 44 | [44_eda_recommendations_generic](tasks/44_eda_recommendations_generic.md) | Build evidence-backed next actions without assuming temporal validation or Gini by default. |
| 45 | [45_eda_mvp_orchestrator_and_cli_generic](tasks/45_eda_mvp_orchestrator_and_cli_generic.md) | Wire P0 generic EDA modules into a working local-dataset MVP. |
| 46 | [46_eda_relationship_inferer_generic](tasks/46_eda_relationship_inferer_generic.md) | Infer relationships between base and secondary tables for generic multi-table tabular competitions. |
| 47 | [47_eda_drift_analyzer_generic](tasks/47_eda_drift_analyzer_generic.md) | Analyze drift as optional evidence, not as a universal assumption. |
| 48 | [48_eda_baseline_runner_generic](tasks/48_eda_baseline_runner_generic.md) | Run an honest baseline appropriate to task_type and metric family. |
| 49 | [49_eda_feature_probe_generic](tasks/49_eda_feature_probe_generic.md) | Assess promising feature families across generic tabular tasks. |
| 50 | [50_eda_notebook_static_analysis_generic](tasks/50_eda_notebook_static_analysis_generic.md) | Statically extract patterns from notebook source text without executing notebooks. |
| 51 | [51_eda_p1_orchestrator_wiring_generic](tasks/51_eda_p1_orchestrator_wiring_generic.md) | Wire optional P1 modules into the EDA orchestrator without making them mandatory. |
| 52 | [52_eda_p1_hypothesis_and_recommendation_rules_generic](tasks/52_eda_p1_hypothesis_and_recommendation_rules_generic.md) | Extend hypothesis evaluation and recommendations to use generic P1 evidence. |
| 53 | [53_research_scout_schemas_generic](tasks/53_research_scout_schemas_generic.md) | Define schemas for Research Scout outputs that feed the generic EDA Engine. |
| 54 | [54_research_scout_reasoner_generic](tasks/54_research_scout_reasoner_generic.md) | Implement the Research Scout reasoning module that generates generic EDA hypotheses from retrieved sources. |
| 55 | [55_research_pipeline_writes_scout_outputs](tasks/55_research_pipeline_writes_scout_outputs.md) | Wire Research Scout into `run_research` so the research pipeline can produce EDA input files. |
| 56 | [56_final_synthesizer_schema_generic](tasks/56_final_synthesizer_schema_generic.md) | Define a structured contract for final strategy synthesis that can consume generic EDA evidence. |
| 57 | [57_final_synthesizer_reasoner_generic](tasks/57_final_synthesizer_reasoner_generic.md) | Implement the final strategy synthesizer that combines retrieved sources and generic EDA evidence. |
| 58 | [58_full_research_to_eda_to_strategy_cli_generic](tasks/58_full_research_to_eda_to_strategy_cli_generic.md) | Add an optional full workflow that runs research, writes Scout outputs, runs EDA, and synthesizes final strategy. |
| 59 | [59_eda_quality_gates_generic](tasks/59_eda_quality_gates_generic.md) | Validate EDA outputs before they are used by Final Synthesizer. |
| 60 | [60_eda_resource_limits_and_sampling](tasks/60_eda_resource_limits_and_sampling.md) | Centralize row caps, memory-safe sampling, and module runtime limits. |
| 61 | [61_eda_error_handling_and_partial_runs](tasks/61_eda_error_handling_and_partial_runs.md) | Make partial EDA runs reproducible and useful even when some modules fail. |
| 62 | [62_eda_summary_generator_generic](tasks/62_eda_summary_generator_generic.md) | Generate a concise human-readable `eda_summary.md` from evidence pack without adding unsupported claims. |
| 63 | [63_eda_integration_fixture_full_p1_generic](tasks/63_eda_integration_fixture_full_p1_generic.md) | Add offline integration tests that run MVP + P1 modules on generic fixture data. |
| 64 | [64_eda_production_cli_docs_generic](tasks/64_eda_production_cli_docs_generic.md) | Document practical commands and expected outputs for generic EDA Engine production use. |

## Source integrity

- Task source SHA-256: `3ef45dbb837879329d24546a2ad264290628727ed7402ab452875329629a0f84`
- EDA specification SHA-256: `f83bb58b51f7d1cff58f95fa20fb929431eff0abf7ad018beb98215978b6f43c`
- Extracted task range: `40–64`
- Extracted task count: `25`
