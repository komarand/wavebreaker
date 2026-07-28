# V6-P1-003 — Artifact refs and integrity

## Status
planned

## Depends on
- V6-P1-002

## Goal
Implement the three artifact reference types and their integrity rules, with
identity that includes the attempt.

## Normative specification
- §10.1, §10.2, §10.3, §10.4, §10.4.1
- §29.1

## Package placement note
`kaggle_researcher/contracts/` already exists with legacy modules of the same
names. New v6 contracts land in `kaggle_researcher/contracts/v6/` and are never
merged into the legacy modules during Phase 1. Legacy code reaches v6 contracts
only through adapters (§8), never by import from a legacy contract module.

## Allowed files
- `kaggle_researcher/contracts/v6/artifacts.py`
- `tests/contracts/v6/test_artifact_refs.py`

## Forbidden scope
- No publication protocol, no bundles, no manifests.
- No catalog or evidence types.
- No actual file writing beyond temporary fixtures in tests.

## Acceptance criteria
1. `JsonArtifactRef`, `TableArtifactRef`, and `BlobArtifactRef` exist and are
   discriminated by `kind`.
2. `artifact_id` includes run, node, and attempt identity, so two attempts of
   one node never share an artifact identity.
3. `artifact_id` is not a content address, and no code derives identity from
   `file_hash`.
4. `file_hash` is used for integrity, resume, and change detection only.
5. `TableArtifactRef` carries a schema fingerprint, row count, and column
   count, and validates schema at publication.
6. Negative: two attempts of the same node produce different `artifact_id`
   values for otherwise identical outputs.
7. Negative: a corrupted file fails integrity checking with a typed error
   naming the artifact.
8. Negative: no code path promises byte-identical float serialization across
   platforms.

## Verification
```bash
pytest tests/contracts/v6/test_artifact_refs.py -q
```

## Stop conditions
- A required integrity field cannot be produced without reading the dataset.
