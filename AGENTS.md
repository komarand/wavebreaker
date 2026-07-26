# KaggleResearcher — repository instructions

This file defines the working rules for coding agents in this repository.
Keep it short and stable. Product and architecture requirements belong in the
canonical specification; operational commands belong in `RUNBOOK.md`.

## Required reading

Before making changes:

1. Read `RUNBOOK.md`.
2. Read the currently assigned task file.
3. Read only the canonical specification sections referenced by that task.
4. Inspect the directly affected implementation and tests.

Do not load the entire specification, the entire backlog, or archived documents
unless the current task explicitly requires them.

If `RUNBOOK.md`, the assigned task file, or a referenced canonical
specification section is missing, stop and report the missing source. Do not
guess its requirements.

## Sources of truth

Use the following precedence for product and implementation requirements:

1. `docs/specs/KAGGLE_RESEARCHER_V6_SPEC.md` — canonical architecture and
   contracts.
2. The explicitly assigned task file — scope and acceptance criteria.
3. `RUNBOOK.md` — environment, commands, and operational procedures.
4. Existing public contracts and tests.
5. Existing implementation.

`docs/archive/`, v4 documents, v5 specifications, old prompts, generated
reports, Markdown summaries, DOCX files, and legacy evidence packs are
non-normative. Use them only when the current task explicitly requests
historical context or migration analysis. They must not override v6.

If two applicable sources conflict, stop before editing and report:

- the conflicting paths and sections;
- the concrete contradiction;
- the smallest decision needed to unblock implementation.

Do not silently choose one interpretation.

## Task identity

Every implementation request must identify one task by its full path or unique
ID.

- v6 task IDs use a namespaced form such as `V6-P0-001`.
- Revised EDA v5 tasks use `V5-EDA-040` through `V5-EDA-064`.
- A bare number such as `40` is ambiguous and must not be executed.

The revised EDA tasks numbered 40–64 replace earlier wording with the same
numbers. Treat them as a legacy v5 task family, not as the v6 implementation
sequence. Execute a v5 task only when the user explicitly assigns its
namespaced task file.

If a legacy task conflicts with canonical v6 contracts, stop and report the
conflict before changing code.

## Scope discipline

- Implement only the currently assigned task.
- Do not implement later phases or adjacent features opportunistically.
- Limit changes to files required by the task and its tests.
- Preserve unrelated working-tree changes.
- Do not rename, move, or reformat unrelated files.
- Do not create a branch, commit, push, or pull request unless explicitly
  requested.
- Do not change dependencies, lock files, CI, deployment configuration, or
  public APIs unless they are in the assigned scope.

When the requested behavior cannot be implemented literally within scope, stop
and explain the blocker instead of expanding the task.

## Architecture invariants

- Public cross-module contracts are strict, typed, and reject unknown fields.
- Do not weaken validation or add permissive defaults to accept legacy
  payloads.
- Legacy compatibility belongs in explicit boundary adapters.
- Deterministic contract validation, evidence validation, and safety gates must
  not depend on LLM interpretation.
- `EvidenceCatalog` is the single public authority for registering, resolving,
  selecting, and validating evidence references.
- Public evidence references must not depend on JSON paths, list indexes, or
  physical storage layout.
- Derived views such as Markdown, DOCX, summaries, dashboards, and legacy packs
  are outputs, not core-module inputs.
- Required invalid input is a typed failure, not degraded success.
- A degraded result must remain contract-valid and state what was omitted and
  why.
- The orchestrator coordinates execution; domain modules own analytical
  decisions.
- Core tests must remain offline and must not require PostgreSQL, pgvector,
  vLLM, Kaggle, or external LLM services.
- Never execute downloaded notebooks, scripts, repository code, or other
  untrusted artifacts. Notebook analysis is static only.

For generic tabular EDA:

- Metric requirements come from `MetricRegistry`.
- Validation policy comes from `ValidationPolicySelector`.
- A time column alone must never force temporal validation.
- Gini Stability is one supported metric, not a global default.
- Home Credit behavior must emerge from metric, schema, preset, and data
  evidence; it must not define generic architecture.

## Implementation workflow

1. Confirm the assigned task, referenced specification sections, affected
   contracts, and allowed file scope.
2. Inspect current code and tests before editing.
3. Check whether the task is already fully or partially implemented.
4. Identify contract or scope conflicts before making changes.
5. Make the smallest coherent implementation that satisfies the task.
6. Add or update tests for every behavior change and acceptance criterion.
7. Run focused tests first, then the relevant regression suite using the
   interpreter and commands from `RUNBOOK.md`.
8. Review the final diff for scope creep, accidental compatibility weakening,
   and unrelated changes.

Do not replace the project interpreter, external services, or test commands
with guessed alternatives. If a documented command cannot run in the current
environment, report the exact limitation and continue with the strongest
available offline verification that does not change project configuration.

## Completion report

At the end of a coding task, report:

- what changed;
- files changed;
- tests and checks run, with results;
- tests not run and the exact reason;
- remaining limitations or follow-up tasks;
- any contract, migration, or compatibility risk discovered.

Do not claim completion when required acceptance criteria are untested or known
to fail.
