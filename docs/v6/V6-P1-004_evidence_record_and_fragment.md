# V6-P1-004 — Evidence record and fragment

## Status
planned

## Depends on
- V6-P1-001
- V6-P1-003

## Goal
Implement `EvidenceRecord`, its locator, provenance rules, and the
`EvidenceFragment` container that a node publishes.

## Normative specification
- §11.2, §11.3, §11.7
- §10.5 (fragment contract only)
- §29.1

## Package placement note
`kaggle_researcher/contracts/` already exists with legacy modules of the same
names. New v6 contracts land in `kaggle_researcher/contracts/v6/` and are never
merged into the legacy modules during Phase 1. Legacy code reaches v6 contracts
only through adapters (§8), never by import from a legacy contract module.

## Allowed files
- `kaggle_researcher/contracts/v6/evidence.py`
- `tests/contracts/v6/test_evidence_record.py`
- `tests/contracts/v6/test_evidence_fragment.py`

## Forbidden scope
- No catalog, no registration, no atomic publication.
- No legacy promotion table beyond the type that will hold it.

## Acceptance criteria
1. `EvidenceRecord` carries `domain` with values `source`, `dataset`, `system`.
2. `quality="derived"` requires non-empty `derived_from_refs` and a
   `derivation_method` naming a registered transform.
3. `quality="measured"` requires empty `derived_from_refs`.
4. Provenance records the producing module and implementation version, and
   neither participates in `evidence_id`.
5. `EvidenceFragment` carries `run_id`, `competition_id`, `node_id`,
   `attempt_id`, and a tuple of records.
6. Negative: a derived record with no parents is rejected with a typed error.
7. Negative: a record whose locator cannot resolve is rejected.
8. Negative: a record carrying a dotted physical path as its identity is
   rejected; §11.7 promotion is the only path from a legacy address.

## Verification
```bash
pytest tests/contracts/v6/test_evidence_record.py tests/contracts/v6/test_evidence_fragment.py -q
```

## Stop conditions
- A v5 address in the promotion corpus cannot be expressed as
  `(name, dimensions)` without an array index. Record it as a blocking
  migration error; do not embed the index.
