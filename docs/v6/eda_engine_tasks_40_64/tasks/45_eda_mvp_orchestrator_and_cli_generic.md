# 45_eda_mvp_orchestrator_and_cli_generic

## Goal

Wire P0 generic EDA modules into a working local-dataset MVP.

## Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/main.py
kaggle_eda_engine/main.py
tests/eda/test_eda_orchestrator_mvp.py
tests/eda/test_eda_cli.py
```

## Codex prompt

```text
Implement the generic EDA MVP orchestrator and CLI.

Function:
- async run_eda(config: EdaRunConfig) -> EdaRunResult

Execution order:
1. load and validate research_hypotheses.json
2. load and validate eda_task_plan.json
3. verify competition_id consistency
4. create run directory
5. copy input JSON files into run directory
6. resolve dataset path
7. build DatasetReader
8. file_inventory
9. schema_inferer
10. table_profiler
11. metric_analyzer using MetricRegistry
12. validation_analyzer using ValidationPolicySelector
13. leakage_checker
14. write module JSON artifacts
15. evaluate_hypotheses
16. build recommended_next_actions
17. build eda_evidence_pack.json
18. build eda_summary.md
19. return EdaRunResult

Requirements:
- Support local dataset path mode.
- Dataset download path may exist but tests must use local dataset.
- Write skipped placeholders for P1 modules:
  - relationship_evidence
  - drift_evidence
  - baseline_evidence
  - feature_probe_evidence
  - notebook_static_analysis
- Do not assume Home Credit as default.
- Home Credit fixture should still work through metric/preset/schema evidence.
```

## Acceptance criteria

- Offline Home Credit-like fixture creates eda_evidence_pack.json.
- Additional iid classification fixture selects StratifiedKFold.
- Additional regression fixture selects KFold.
- CLI test runs without network calls.
- Tests pass.

---

## Rules inherited by this task

The EDA Engine is a **generic tabular evidence engine**.

It must support Home Credit-like competitions, but Home Credit must not define the architecture.

Important rules:

```text
- Generic logic first.
- MetricRegistry determines metric requirements.
- ValidationPolicySelector determines validation policy.
- Competition presets may provide hints, but they must not be required.
- Gini Stability is only one metric registry entry.
- A time column alone must never force temporal validation.
- Home Credit-specific behavior must emerge from metric/schema/preset evidence.
- Notebook execution remains forbidden.
```

The old mental model:

```text
WEEK_NUM exists -> temporal validation
Gini Stability -> default metric worldview
case_id -> default join key
```

must be replaced with:

```text
task_type + metric_family + schema evidence + data signals + scout hypotheses
        -> validation policy
        -> evidence-backed recommendations
```
