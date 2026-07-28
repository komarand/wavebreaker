# V6-P0-004 — Reusable algorithm inventory

## Status
planned

## Depends on
- V6-P0-001

## Goal
Separate v5 analytical logic that is worth carrying into v6 from the contract
plumbing around it, so Phase 2 and 3 port algorithms rather than rewriting or
blindly inheriting them.

## Normative specification
- §15
- §33 Phase 0
- §5.1

## Inputs and outputs
Output: an inventory section in `docs/V6_BASELINE.md` classifying each
analytical unit as portable, portable-with-changes, or replace.

An algorithm is portable when it depends only on typed inputs and returns
values, with no dependency on `EdaEvidencePack`, dotted-path addressing, or the
partially built pack.

## Allowed files
- `docs/V6_BASELINE.md`

## Forbidden scope
- No code changes at all. This task produces a document.

## Acceptance criteria
1. Every module under `kaggle_researcher/eda/modules/` appears with a
   classification and a one-line reason.
2. The validation engine under `kaggle_researcher/eda/validation/` is
   classified per policy, not as a whole, and the inventory records which
   policies exist today: `GroupKFold`, `StratifiedGroupKFold`,
   `StratifiedKFold`, and temporal holdout variants.
3. The inventory records that no purged or embargoed time-series policy exists,
   and flags it as a Phase 3 gap against the §23 constraint example.
4. The nine leakage checks in `leakage_checker.py` are listed individually with
   a classification.
5. Each unit marked portable names the pure function or class to lift.
6. Each unit marked replace names the specification section that makes the
   current form unusable.
7. Negative: no unit is left unclassified.

## Stop conditions
- A module's dependencies cannot be determined by reading it, because behavior
  is decided by the shape of the incoming pack at runtime. Record it as
  replace, with that as the reason.
