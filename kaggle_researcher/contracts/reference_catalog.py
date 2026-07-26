from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from pydantic import ConfigDict, Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.evidence_manifest import EvidenceReferenceManifest
from kaggle_researcher.contracts.research import ResearchHypotheses


ReferenceNamespace = Literal[
    "evidence",
    "hypothesis",
    "risk",
    "validation_requirement",
    "safety_constraint",
    "source_claim",
]
REFERENCE_NAMESPACES = frozenset(get_args(ReferenceNamespace))
ReferenceDiagnosticCode = Literal[
    "duplicate_reference_id",
    "unknown_namespace",
    "missing_backing_evidence",
    "broken_evidence_path",
    "unknown_reference",
    "namespace_mismatch",
]
ReferenceResolutionStatus = Literal[
    "resolved",
    "unresolved",
    "ambiguous",
    "invalid_namespace",
    "namespace_mismatch",
]
ReferenceSupportKind = Literal["factual", "policy"]


class FrozenCatalogModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ReferenceCatalogDiagnostic(FrozenCatalogModel):
    code: ReferenceDiagnosticCode
    ref_id: str
    namespace: ReferenceNamespace | None = None
    evidence_ref: str | None = None
    message: str


class ReferenceCatalogEntry(FrozenCatalogModel):
    ref_id: str = Field(min_length=1)
    namespace: ReferenceNamespace
    canonical_ref: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    backing_evidence_refs: tuple[str, ...] = ()
    uncertainty_evidence_refs: tuple[str, ...] = ()
    evidence_backed: bool = False
    support_kind: ReferenceSupportKind = "factual"
    summary: str | None = None
    title: str | None = None
    source_type: str | None = None


class ReferenceResolution(FrozenCatalogModel):
    query: str
    status: ReferenceResolutionStatus
    expected_namespace: ReferenceNamespace | None = None
    entry: ReferenceCatalogEntry | None = None
    diagnostics: tuple[ReferenceCatalogDiagnostic, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved" and self.entry is not None


class ReferenceCatalog(FrozenCatalogModel):
    entries: tuple[ReferenceCatalogEntry, ...] = ()
    diagnostics: tuple[ReferenceCatalogDiagnostic, ...] = ()

    def resolve(
        self,
        ref_id: str,
        expected_namespace: str | None = None,
    ) -> ReferenceResolution:
        if expected_namespace is not None and expected_namespace not in REFERENCE_NAMESPACES:
            diagnostic = ReferenceCatalogDiagnostic(
                code="unknown_namespace",
                ref_id=ref_id,
                message=f"Unknown reference namespace: {expected_namespace!r}.",
            )
            return ReferenceResolution(
                query=ref_id,
                status="invalid_namespace",
                diagnostics=(diagnostic,),
            )

        matches = [
            entry
            for entry in self.entries
            if ref_id == entry.ref_id or ref_id in entry.aliases
        ]
        typed_namespace = expected_namespace
        if typed_namespace is not None:
            namespace_matches = [
                entry for entry in matches if entry.namespace == typed_namespace
            ]
            if matches and not namespace_matches:
                diagnostic = ReferenceCatalogDiagnostic(
                    code="namespace_mismatch",
                    ref_id=ref_id,
                    namespace=matches[0].namespace,
                    message=(
                        f"Reference resolves as {matches[0].namespace!r}, not "
                        f"{typed_namespace!r}."
                    ),
                )
                return ReferenceResolution(
                    query=ref_id,
                    status="namespace_mismatch",
                    expected_namespace=typed_namespace,
                    diagnostics=(diagnostic,),
                )
            matches = namespace_matches

        if not matches:
            diagnostic = ReferenceCatalogDiagnostic(
                code="unknown_reference",
                ref_id=ref_id,
                message="Reference is not present in the Final Strategy catalog.",
            )
            return ReferenceResolution(
                query=ref_id,
                status="unresolved",
                expected_namespace=typed_namespace,
                diagnostics=(diagnostic,),
            )
        if len(matches) > 1:
            namespaces = sorted({entry.namespace for entry in matches})
            diagnostic = ReferenceCatalogDiagnostic(
                code="duplicate_reference_id",
                ref_id=ref_id,
                message=f"Reference is ambiguous across catalog entries: {namespaces}.",
            )
            return ReferenceResolution(
                query=ref_id,
                status="ambiguous",
                expected_namespace=typed_namespace,
                diagnostics=(diagnostic,),
            )
        return ReferenceResolution(
            query=ref_id,
            status="resolved",
            expected_namespace=typed_namespace,
            entry=matches[0],
        )

    def get_namespace(self, ref_id: str) -> ReferenceNamespace | None:
        resolution = self.resolve(ref_id)
        return resolution.entry.namespace if resolution.is_resolved else None

    def get_backing_evidence_refs(self, ref_id: str) -> tuple[str, ...]:
        resolution = self.resolve(ref_id)
        if not resolution.is_resolved or resolution.entry is None:
            return ()
        return resolution.entry.backing_evidence_refs

    def is_valid_evidence_ref(self, ref: str) -> bool:
        resolution = self.resolve(ref)
        return bool(
            resolution.is_resolved
            and resolution.entry is not None
            and resolution.entry.namespace in {"evidence", "source_claim"}
            and resolution.entry.evidence_backed
        )


def build_final_strategy_reference_catalog(
    eda_evidence_pack: EdaEvidencePack,
    *,
    evidence_manifest: EvidenceReferenceManifest,
    research_hypotheses: ResearchHypotheses | None = None,
    source_claim_ids: Iterable[str] = (),
    retrieved_documents: Iterable[Any] = (),
) -> ReferenceCatalog:
    """Build the immutable Final Strategy reference catalog without inference."""

    entries: list[ReferenceCatalogEntry] = []
    diagnostics: list[ReferenceCatalogDiagnostic] = []
    evidence_refs = {
        entry.ref for entry in evidence_manifest.entries
        if entry.available
        and entry.namespace == "eda_evidence"
        and entry.reference_kind in {"direct_path", "semantic_ref"}
    }
    source_metadata: dict[str, tuple[str | None, str | None]] = {}
    for document in retrieved_documents:
        document_id = str(
            document.get("id") if isinstance(document, Mapping) else getattr(document, "id", "")
        ).strip()
        if not document_id:
            continue
        title = (
            document.get("title") if isinstance(document, Mapping) else getattr(document, "title", None)
        )
        source_type = (
            document.get("source") if isinstance(document, Mapping) else getattr(document, "source", None)
        )
        metadata = (
            str(title).strip() or None if title is not None else None,
            str(source_type).strip() or None if source_type is not None else None,
        )
        if document_id in source_metadata:
            diagnostics.append(_duplicate_diagnostic(
                document_id, "source_claim", 2,
            ))
            continue
        source_metadata[document_id] = metadata
    source_ids = {
        str(value).strip() for value in source_claim_ids if str(value).strip()
    } | set(source_metadata)
    valid_backing_refs = evidence_refs | source_ids

    for evidence_ref in sorted(evidence_refs):
        entries.append(ReferenceCatalogEntry(
            ref_id=evidence_ref,
            namespace="evidence",
            canonical_ref=evidence_ref,
            evidence_backed=True,
        ))
    for source_id in sorted(source_ids):
        title, source_type = source_metadata.get(source_id, (None, None))
        entries.append(ReferenceCatalogEntry(
            ref_id=source_id,
            namespace="source_claim",
            canonical_ref=source_id,
            evidence_backed=True,
            title=title,
            source_type=source_type,
        ))

    validated_claims = eda_evidence_pack.source_claim_validation.get("validated_claims", [])
    if isinstance(validated_claims, list):
        for claim in validated_claims:
            if not isinstance(claim, Mapping) or not str(claim.get("claim_id", "")).strip():
                continue
            claim_id = str(claim["claim_id"]).strip()
            backing = tuple(sorted({
                *[str(value) for value in claim.get("supporting_eda_refs", [])],
                *[str(value) for value in claim.get("contradicting_eda_refs", [])],
                *[str(value) for value in claim.get("source_evidence_refs", [])],
            }))
            entries.append(_object_entry(
                ref_id=claim_id,
                namespace="source_claim",
                backing=backing,
                summary=str(claim.get("claim_text") or claim.get("finding") or "") or None,
                valid_backing_refs=valid_backing_refs,
                diagnostics=diagnostics,
            ))

    producer_hypotheses: dict[str, list[Any]] = defaultdict(list)
    if research_hypotheses is not None:
        for hypothesis in research_hypotheses.hypotheses:
            producer_hypotheses[str(hypothesis.hypothesis_id)].append(hypothesis)
    for hypothesis in eda_evidence_pack.testable_hypotheses:
        producer_hypotheses[str(hypothesis.hypothesis_id)].append(hypothesis)

    hypothesis_results: dict[str, list[Any]] = defaultdict(list)
    for result in eda_evidence_pack.hypothesis_results:
        hypothesis_results[str(result.hypothesis_id)].append(result)

    for hypothesis_id in sorted(set(producer_hypotheses) | set(hypothesis_results)):
        producers = producer_hypotheses[hypothesis_id]
        results = hypothesis_results[hypothesis_id]
        if len(producers) > 1 or len(results) > 1:
            diagnostics.append(_duplicate_diagnostic(
                hypothesis_id,
                "hypothesis",
                len(producers) + len(results),
            ))
        result = results[0] if results else None
        producer = producers[0] if producers else None
        backing = (
            tuple(sorted(set(str(value) for value in result.evidence_refs)))
            if result is not None
            else tuple(sorted(set(str(value) for value in getattr(producer, "evidence_refs", ()))))
        )
        summary = (
            getattr(result, "finding", None)
            or getattr(producer, "statement", None)
            or getattr(producer, "claim", None)
        )
        uncertainty_refs: tuple[str, ...] = ()
        if result is not None and result.status in {"not_testable", "skipped"}:
            candidate = f"hypothesis_results.{hypothesis_id}"
            if candidate in evidence_refs:
                uncertainty_refs = (candidate,)
        elif result is None and producer is not None and getattr(producer, "status", None) == "untested":
            candidate = f"testable_hypotheses.{hypothesis_id}"
            if candidate in evidence_refs:
                uncertainty_refs = (candidate,)
        entries.append(_object_entry(
            ref_id=hypothesis_id,
            namespace="hypothesis",
            backing=backing,
            summary=summary,
            valid_backing_refs=valid_backing_refs,
            diagnostics=diagnostics,
            uncertainty_evidence_refs=uncertainty_refs,
        ))

    for risk in eda_evidence_pack.eda_risks:
        entries.append(_object_entry(
            ref_id=str(risk.risk_id),
            namespace="risk",
            backing=tuple(sorted(set(str(value) for value in risk.evidence_refs))),
            summary=risk.title or risk.finding,
            valid_backing_refs=valid_backing_refs,
            diagnostics=diagnostics,
        ))
    for requirement in eda_evidence_pack.validation_requirements:
        ref_id = str(requirement.validation_requirement_id)
        entries.append(_object_entry(
            ref_id=ref_id,
            namespace="validation_requirement",
            aliases=(f"validation_requirements.{ref_id}",),
            backing=tuple(sorted(set(str(value) for value in requirement.evidence_refs))),
            summary=requirement.rule,
            valid_backing_refs=valid_backing_refs,
            diagnostics=diagnostics,
        ))
    for constraint in eda_evidence_pack.safety_constraints:
        ref_id = str(constraint.safety_constraint_id)
        entries.append(_object_entry(
            ref_id=ref_id,
            namespace="safety_constraint",
            aliases=(f"safety_constraints.{ref_id}",),
            backing=tuple(sorted(set(str(value) for value in constraint.evidence_refs))),
            summary=constraint.rule,
            valid_backing_refs=valid_backing_refs,
            diagnostics=diagnostics,
            support_kind=(
                "factual"
                if constraint.evidence_origin == "dataset_measurement"
                else "policy"
            ),
        ))

    _append_duplicate_lookup_diagnostics(entries, diagnostics)
    entries.sort(key=lambda item: (item.ref_id, item.namespace, item.canonical_ref))
    diagnostics.sort(key=lambda item: (
        item.code,
        item.ref_id,
        item.namespace or "",
        item.evidence_ref or "",
    ))
    return ReferenceCatalog(entries=tuple(entries), diagnostics=tuple(diagnostics))


def _object_entry(
    *,
    ref_id: str,
    namespace: ReferenceNamespace,
    backing: tuple[str, ...],
    summary: str | None,
    valid_backing_refs: set[str],
    diagnostics: list[ReferenceCatalogDiagnostic],
    aliases: tuple[str, ...] = (),
    support_kind: ReferenceSupportKind = "factual",
    uncertainty_evidence_refs: tuple[str, ...] = (),
) -> ReferenceCatalogEntry:
    broken = tuple(value for value in backing if value not in valid_backing_refs)
    if not backing:
        diagnostics.append(ReferenceCatalogDiagnostic(
            code="missing_backing_evidence",
            ref_id=ref_id,
            namespace=namespace,
            message="Object declares no backing evidence references.",
        ))
    for evidence_ref in broken:
        diagnostics.append(ReferenceCatalogDiagnostic(
            code="broken_evidence_path",
            ref_id=ref_id,
            namespace=namespace,
            evidence_ref=evidence_ref,
            message=f"Backing evidence path {evidence_ref!r} does not resolve.",
        ))
    return ReferenceCatalogEntry(
        ref_id=ref_id,
        namespace=namespace,
        canonical_ref=ref_id,
        aliases=aliases,
        backing_evidence_refs=backing,
        uncertainty_evidence_refs=uncertainty_evidence_refs,
        evidence_backed=bool(backing) and not broken,
        support_kind=support_kind,
        summary=summary,
    )


def _append_duplicate_lookup_diagnostics(
    entries: list[ReferenceCatalogEntry],
    diagnostics: list[ReferenceCatalogDiagnostic],
) -> None:
    lookup: dict[str, list[ReferenceCatalogEntry]] = defaultdict(list)
    for entry in entries:
        for value in (entry.ref_id, *entry.aliases):
            lookup[value].append(entry)
    existing = {
        (item.code, item.ref_id, item.namespace)
        for item in diagnostics
        if item.code == "duplicate_reference_id"
    }
    for value, matches in lookup.items():
        if len(matches) < 2:
            continue
        namespaces = sorted({item.namespace for item in matches})
        key = ("duplicate_reference_id", value, None)
        if key in existing:
            continue
        diagnostics.append(ReferenceCatalogDiagnostic(
            code="duplicate_reference_id",
            ref_id=value,
            message=f"Reference ID is registered more than once in namespaces {namespaces}.",
        ))


def _duplicate_diagnostic(
    ref_id: str,
    namespace: ReferenceNamespace,
    count: int,
) -> ReferenceCatalogDiagnostic:
    return ReferenceCatalogDiagnostic(
        code="duplicate_reference_id",
        ref_id=ref_id,
        namespace=namespace,
        message=f"Reference ID has {count} producer/observation records.",
    )


__all__ = [
    "REFERENCE_NAMESPACES",
    "ReferenceCatalog",
    "ReferenceCatalogDiagnostic",
    "ReferenceCatalogEntry",
    "ReferenceNamespace",
    "ReferenceResolution",
    "ReferenceSupportKind",
    "build_final_strategy_reference_catalog",
]
