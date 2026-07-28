# V6-P1-005 — Evidence catalog registration

## Status
planned

## Depends on
- V6-P1-004

## Goal
Implement the canonical catalog and its registration invariants: atomic
all-or-nothing fragment registration, scope validation, and duplicate and
collision detection.

## Normative specification
- §5.4
- §11.4, §11.5
- §29.1

## Package placement note
New v6 contracts land in `kaggle_researcher/contracts/v6/`. Legacy modules of
the same name are not modified during Phase 1.

## Allowed files
- `kaggle_researcher/contracts/v6/catalog.py`
- `tests/contracts/v6/test_catalog_registration.py`

## Forbidden scope
- No catalog view, no allowed-reference generation.
- No filesystem publication protocol, no markers, no generations.
- No second catalog of any kind. Source evidence is `domain="source"` inside
  the one canonical catalog.

## Acceptance criteria
1. One catalog belongs to exactly one `(run_id, competition_id)` pair.
2. Registration is all-or-nothing per fragment: a fragment is wholly registered
   or wholly absent.
3. Registration fails when an ID maps to a different key, when two records
   share a namespace, key, and dimensions in one scope, when two modules
   declare one evidence namespace, when scope does not match, when provenance
   is missing, or when a derived record cites an ID absent from the catalog.
4. `evidence_id` values are unique and resolvable within the owning run.
5. Negative: cross-run resolution is rejected with a typed error; it never
   falls back to a lookup in another run.
6. Negative: a partially registered fragment leaves no record behind after the
   failure.
7. Negative: registering the same fragment twice is rejected rather than
   silently merged.
8. Negative: a genuine ID collision reports both keys and does not overwrite.

## Verification
```bash
pytest tests/contracts/v6/test_catalog_registration.py -q
```

## Stop conditions
- An invariant cannot be enforced without reading published bundles from disk.
  That belongs to V6-P1-008; report the boundary rather than pulling it in.
