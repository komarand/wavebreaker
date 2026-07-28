# V6-P0-002 — Offline characterization fixtures

## Status
ready

## Depends on
- V6-P0-001

## Goal
Make current v5 behavior reproducible offline, at full-run and stage level,
with no network, no LLM call, and no dataset download. This is the Phase 0
exit condition in §33.

## Normative specification
- §29.1, §29.6
- §33 Phase 0
- §20.3

## Inputs and outputs
Input: the existing test suite (131 test files, 87 of which touch validation or
leakage) and existing fixtures under `tests/fixtures/`.

Output: a documented offline characterization set and, only where genuinely
missing, new fixtures. If existing tests already cover a case, this task
records that fact and does not duplicate them.

## Allowed files
- `tests/fixtures/**`
- `tests/integration/**`
- `tests/conftest.py`
- `docs/implementation/V6_BASELINE.md` (append a characterization section)
- `AGENTS.md` (create)
- `docs/archive/` (create, and move superseded task documents into it)
- `docs/v6/README.md` (status column only)
- `docs/RUNBOOK.md`


## Forbidden scope
- No change to any module under `kaggle_researcher/`.
- No new v6 contracts.
- No refactoring of existing tests beyond what is needed to run them offline.

## Acceptance criteria
1. A documented command runs the full pipeline offline from fixtures with no
   network access and no LLM call.
2. A documented command runs each stage independently from stored inputs.
3. Both commands pass from a clean checkout with no Kaggle credentials present.
4. An audit table lists each characterization case and states whether it was
   already covered, and by which test, or newly added.
5. Negative: with network disabled, no test attempts an outbound connection;
   any test that requires network is explicitly marked and excluded from the
   offline command.
6. Negative: removing a stored fixture causes a clear failure naming the
   missing fixture, not an unrelated error deep in a module.
7. `AGENTS.md` exists and states the repository conventions the execution
   prompt assumes.
8. `docs/archive/` exists and contains the superseded task documents, and no
   file in `docs/tasks/v6/` references them as requirements.

## Verification
```bash
pytest tests/ -m "not network" -q
pytest tests/integration -q
```
Record the exact offline invocation in `docs/RUNBOOK.md`.

## Stop conditions
- An existing test cannot be made to run offline without changing module code.
  Report it; do not change the module.
