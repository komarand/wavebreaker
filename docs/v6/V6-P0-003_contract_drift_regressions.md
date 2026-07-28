# V6-P0-003 — Contract drift regressions

## Status
planned

## Depends on
- V6-P0-002

## Goal
Pin the known contract failures as failing-or-xfail regression tests, so the v6
migration proves it fixed them rather than asserting it did.

## Normative specification
- §2
- §5.3, §5.4, §5.5
- §36
- §29.1

## Inputs and outputs
Input: `docs/contract_audit.md` and the anti-pattern findings from
V6-P0-001.

Output: regression tests that characterize each known drift class. Tests
document current behavior; they do not fix it.

## Allowed files
- `tests/contracts/**`
- `docs/V6_BASELINE.md` (drift section)

## Forbidden scope
- No fix for any drift. This task only pins behavior.
- No change to `kaggle_researcher/**`.

## Acceptance criteria
1. A test characterizes the allowed-reference path: the prompt's allowed set is
   produced by `generate_allowed_evidence_refs`, and the validator's accepted
   set is derived independently.
2. A test characterizes evidence addressing by dotted dictionary path,
   including at least one parameterized address of the
   `baseline_ablation_evidence.ablations.<id>` shape.
3. A test characterizes pack drift: at least two live fields representing one
   concept, such as the strategy-hints and risk-register duplicates.
4. A test characterizes namespace collision behavior when two producers write
   overlapping evidence areas.
5. A test characterizes the partial-pack read: a module consuming
   `evidence_pack_partial` before it is fully assembled.
6. Negative: each regression fails, or is marked `xfail` with a reason naming
   the specification section that forbids it, so that fixing the behavior turns
   the test green rather than leaving it silently passing.
7. Every regression names the specification section it maps to.

## Verification
```bash
pytest tests/contracts -q -rxX
```

## Stop conditions
- A drift class from the audit cannot be reproduced deterministically. Record
  it as unreproducible rather than inventing a synthetic case.
