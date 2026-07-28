# V6-P1-006 — Evidence catalog view

## Status
planned

## Depends on
- V6-P1-005

## Goal
Implement the immutable prompt-facing projection and make it the single source
of allowed references for both the prompt builder and the reference validator.

## Normative specification
- §11.6
- §22 (view contract only)
- §36
- §29.1

## Package placement note
New v6 contracts land in `kaggle_researcher/contracts/v6/`. Legacy modules of
the same name are not modified during Phase 1.

## Allowed files
- `kaggle_researcher/contracts/v6/catalog_view.py`
- `tests/contracts/v6/test_catalog_view.py`

## Forbidden scope
- No synthesis context, no prompt building, no LLM interaction.
- No standalone allowed-reference generator of any kind.

## Acceptance criteria
1. `EvidenceCatalogView` is immutable and carries `generation`,
   `snapshot_status`, and `snapshot_digest`.
2. `allowed_refs` is exactly the set of `evidence_id` values in `entries`.
3. Scalars render exact values; distributions and tables render a bounded
   deterministic summary; blobs render `rendered_value=None`.
4. Rendering is byte-stable: the same view produces the same rendered output.
5. The view contains no locator, physical path, run directory, or artifact
   filename.
6. Negative: no second function computes an allowed set. A repository search
   for a standalone allowed-reference generator returns nothing in v6 code.
7. Negative: a reference absent from the view fails validation, and the error
   names the reference.
8. Negative: a view whose `snapshot_digest` disagrees with its snapshot is
   rejected rather than used.

## Verification
```bash
pytest tests/contracts/v6/test_catalog_view.py -q
rg -n "generate_allowed_evidence_refs" kaggle_researcher/contracts/v6/ || true
```
The search must return nothing under v6 paths.

## Stop conditions
- Prompt rendering requires a field the view does not expose. Report it; do not
  widen the view to include locators.
