# 46_eda_relationship_inferer_generic

## Goal

Infer relationships between base and secondary tables for generic multi-table tabular competitions.

## Files to create/change

```text
kaggle_researcher/eda/modules/relationship_inferer.py
tests/eda/test_relationship_inferer.py
```

## Codex prompt

```text
Implement relationship_inferer module.

Function:
- infer_relationships(
    inferred_schema: InferredSchema,
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
) -> dict

Requirements:
- Identify base_table and base_id_column.
- For each secondary train/test table:
  - find candidate join keys shared with base table.
  - compute relationship_type:
    one_to_one, one_to_many, many_to_one, many_to_many, unknown
  - compute coverage_left_to_right.
  - compute orphan_rate_right.
  - compute avg_rows_per_left and max_rows_per_left.
  - compute row_multiplication_risk low/medium/high.
  - detect candidate group/query/entity keys.
  - detect candidate date cutoff columns.
  - assign confidence.
- Do not recommend direct one-to-many joins without aggregation.
- Do not assume case_id as the only possible join key.
- Use schema hints, sample_submission keys, and shared column patterns.
- Must support sampled checks for large tables.
```

## Acceptance criteria

- Home Credit fixture detects relationship by case_id.
- Generic fixture detects relationship by customer_id/order_id.
- one-to-many relationship is detected when multiple rows per base id exist.
- Missing join key returns unknown relationship with warning.
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
