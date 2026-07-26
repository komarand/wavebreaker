# 51_eda_p1_orchestrator_wiring_generic

## Goal

Wire optional P1 modules into the EDA orchestrator without making them mandatory.

## Files to create/change

```text
kaggle_researcher/eda/orchestrator.py
kaggle_researcher/eda/main.py
tests/eda/test_eda_orchestrator_p1.py
```

## Codex prompt

```text
Wire P1 modules into run_eda.

Modules:
- relationship_inferer
- drift_analyzer
- baseline_runner
- feature_probe
- notebook_static_analysis

Requirements:
- P1 modules run only when:
  - --enable-p1-modules is set, or
  - module is explicitly listed in --modules, or
  - eda_task_plan recommended_module_sequence includes it and config allows P1.
- baseline_runner additionally requires --enable-baseline.
- Non-blocking P1 failures should:
  - write module JSON with status=failed/skipped
  - add warning
  - continue run
- Update EdaEvidencePack fields.
- Update eda_summary.md with P1 sections when available.
- P1 modules must respect generic task_type and metric_evidence.
```

## Acceptance criteria

- MVP run still works without P1 flags.
- P1 fixture run writes relationship_evidence.json and drift_evidence.json.
- Baseline does not run unless --enable-baseline.
- P1 failure is warning, not fatal.
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
