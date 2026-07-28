# V6-P1-008 — Publication contracts

## Status
planned

## Depends on
- V6-P1-005
- V6-P1-007

## Goal
Implement the node bundle contracts, manifests, attempts, generations, and
publication-boundary validation, without implementing the runner.

## Normative specification
- §10.5, §10.6, §10.7
- §19 (layout only)
- §29.1, §29.6 scenarios 1–3 and 7–9

## Package placement note
New v6 contracts land in `kaggle_researcher/contracts/v6/`. Legacy modules of
the same name are not modified during Phase 1.

## Allowed files
- `kaggle_researcher/contracts/v6/publication.py`
- `kaggle_researcher/contracts/v6/generation.py`
- `tests/contracts/v6/test_publication_contracts.py`
- `tests/contracts/v6/test_generations.py`

## Forbidden scope
- No runner, no scheduling, no execution of modules.
- No retention or garbage collection.
- No synthesis.

## Acceptance criteria
1. `NodeManifest` and `EvidenceFragment` are versioned contracts; recovery
   reads them rather than inferring state from filenames.
2. `NodeManifest.status` is `prepared` and is never rewritten. Publication is
   expressed by adding an immutable `PUBLISHED` marker.
3. Every state transition in the protocol is a rename; no file is rewritten in
   place.
4. `manifest_digest` is computed over the canonical serialization of the
   manifest excluding the digest field, using §11.3 canonical JSON settings.
5. `GenerationSnapshot` distinguishes `candidate` from `committed`; closure
   nodes read the candidate and external readers see the committed snapshot.
6. At most one attempt per node is active; exactly one exists for every
   successfully published node in the committed generation. A node that has
   never run has no active attempt and that is valid.
7. Commit is a single atomic manifest replacement covering the whole dependency
   closure, never a per-node edit.
8. Negative: a bundle without a valid marker contributes no evidence, and the
   catalog rebuilt from published fragments excludes it.
9. Negative: an abandoned candidate generation leaves nothing observable to
   exports, presentation, or resume.
10. Negative: a stored catalog whose generation or digest disagrees with the
    committed manifest is rebuilt or refused, never trusted.
11. Negative: a crash simulated between rename and marker creation is resolved
    by recovery with the previous generation intact.

## Verification
```bash
pytest tests/contracts/v6/test_publication_contracts.py tests/contracts/v6/test_generations.py -q
```
Crash scenarios are simulated by interrupting the protocol between steps in a
temporary directory, not by killing the process.

## Stop conditions
- A guarantee requires an atomic operation across two filesystems. Report it;
  the protocol assumes a single filesystem for rename atomicity.
