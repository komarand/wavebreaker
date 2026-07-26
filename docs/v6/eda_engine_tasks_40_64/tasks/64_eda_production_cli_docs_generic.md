# 64_eda_production_cli_docs_generic

## Goal

Document practical commands and expected outputs for generic EDA Engine production use.

## Files to create/change

```text
README.md
docs/EDA_ENGINE_SPEC.md
docs/RUNBOOK.md
```

## Codex prompt

```text
Add production usage documentation for EDA Engine.

Include commands:
1. Run research-only pipeline.
2. Generate Research Scout EDA plan.
3. Run EDA MVP with local dataset.
4. Run EDA with P1 modules.
5. Run EDA with baseline.
6. Run full research -> EDA -> final strategy workflow.

Use Windows PowerShell examples with:
E:\wavebreaker\.venv-win\Scripts\python.exe

Document expected outputs:
- research_hypotheses.json
- eda_task_plan.json
- eda_evidence_pack.json
- eda_summary.md
- final_strategy.json/md/docx

Document safety notes:
- Kaggle rules must be accepted before dataset download.
- Notebook execution is never performed.
- Baseline is sanity check, not final score optimization.
- Large datasets may be sampled and sampled=true must be respected.

Document generic tabular behavior:
- ordinary classification can use StratifiedKFold.
- ordinary regression can use KFold.
- grouped tasks can use group-aware validation.
- temporal validation is used only when evidence supports it.
- Gini Stability is supported but not the default worldview.
```

## Acceptance criteria

- README has EDA section.
- RUNBOOK has copy-pasteable commands.
- Docs mention local dataset mode.
- Docs mention no notebook execution.
- Docs mention generic tabular validation behavior.
- No tests required unless docs lint exists.

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
