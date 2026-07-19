from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.source_registry.cache_policy import decide_cache_action
from kaggle_researcher.source_registry.errors import (
    ArtifactCacheError,
    SourceOfflineCacheMissError,
    SourceVersionError,
)
from kaggle_researcher.source_registry.fingerprints import (
    ProcessorFingerprint,
    build_embedding_fingerprint,
    build_parser_fingerprint,
    build_static_analysis_fingerprint,
    build_summary_fingerprint,
    canonical_json,
)
from kaggle_researcher.source_registry.hashing import (
    compute_content_hashes,
    normalize_text_for_hashing,
    sha256_text,
)
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.source_registry.schemas import (
    ArtifactRecord,
    CacheDecision,
    CachePolicy,
    CacheRunTelemetry,
    SourceDescriptor,
    SourceProcessingResult,
    SourceRefreshMode,
    ContentHashes,
)


class SingleFlight:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def lock(self, key: str):
        async with self._guard:
            lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                if not lock.locked() and getattr(lock, "_waiters", None) in (None, ()):
                    self._locks.pop(key, None)


_SINGLE_FLIGHT = SingleFlight()


async def process_source(
    descriptor: SourceDescriptor,
    competition_id: str,
    run_id: str,
    registry: Any,
    cache_policy: CachePolicy,
    parser: Callable[..., Any] | None = None,
    summarizer: Callable[..., Any] | None = None,
    embedder: Callable[..., Any] | None = None,
    static_analyzers: Mapping[str, Callable[..., Any]] | None = None,
    *,
    fetcher: Callable[[SourceDescriptor], Any] | None = None,
    parser_fingerprint: ProcessorFingerprint | str | None = None,
    summary_fingerprint: ProcessorFingerprint | str | None = None,
    embedding_fingerprint: ProcessorFingerprint | str | None = None,
    static_analysis_fingerprints: Mapping[str, ProcessorFingerprint | str] | None = None,
    telemetry: CacheRunTelemetry | None = None,
) -> SourceProcessingResult:
    identity = canonicalize_source_identity(
        descriptor.source_type, descriptor.external_id, descriptor.canonical_url, descriptor.provider_metadata
    )
    before_source = await registry.get_source(identity.source_id)
    source = await registry.upsert_source(
        identity, descriptor.title, descriptor.canonical_url, descriptor.provider_metadata
    )
    await registry.associate_source_with_competition(
        competition_id, source.source_id, descriptor.source_type, descriptor.discovery_query,
        descriptor.discovery_rank, {},
    )
    _inc(telemetry, "sources_discovered")
    warnings = list(identity.warnings)
    current = await registry.get_current_version(source.source_id)

    reuse_download = (
        current is not None
        and descriptor.revision_is_reliable
        and descriptor.source_revision is not None
        and current.source_revision == descriptor.source_revision
        and cache_policy.source_refresh_mode != SourceRefreshMode.ALWAYS
    )
    raw_content = descriptor.raw_content
    if raw_content is None and reuse_download:
        version = current
        created = False
        _inc(telemetry, "downloads_skipped")
    else:
        if raw_content is None:
            if cache_policy.source_refresh_mode == SourceRefreshMode.NEVER:
                if current is None:
                    raise SourceOfflineCacheMissError(f"No cached content exists for {source.source_id}")
                version = current
                created = False
                _inc(telemetry, "downloads_skipped")
            elif fetcher is None:
                if current is None:
                    raise SourceVersionError(f"No content or fetcher was provided for {source.source_id}")
                version = current
                created = False
                warnings.append("Source content refresh was unavailable; reused the current cached version.")
                _inc(telemetry, "downloads_skipped")
            else:
                raw_content = await _call(fetcher, descriptor)
                _inc(telemetry, "downloads_performed")
        if raw_content is not None:
            raw_pdf_hash = descriptor.provider_metadata.get("raw_pdf_hash")
            if descriptor.source_type == "arxiv" and raw_pdf_hash:
                hashes = ContentHashes(
                    raw_hash=str(raw_pdf_hash),
                    normalized_hash=str(raw_pdf_hash),
                    normalization_policy_version="pdf-bytes-v1",
                    normalized_size=int(descriptor.provider_metadata.get("raw_pdf_size_bytes") or 0),
                )
                raw_text = None
                content_location = descriptor.provider_metadata.get("content_location")
            else:
                content_type = _hash_content_type(descriptor.content_mime_type)
                policy_version = "binary-v1" if content_type == "binary" else "text-v1"
                hashes = compute_content_hashes(raw_content, content_type=content_type, policy_version=policy_version)
                raw_text = raw_content if isinstance(raw_content, str) else None
                content_location = None
            version, created = await registry.create_or_reuse_version(
                source.source_id, descriptor.source_revision, raw_text, content_location,
                descriptor.content_mime_type, hashes,
                {"revision_is_reliable": descriptor.revision_is_reliable, "content_url": descriptor.content_url},
            )
            if descriptor.raw_content is not None:
                _inc(
                    telemetry,
                    "downloads_skipped"
                    if descriptor.provider_metadata.get("content_from_registry_cache")
                    else "downloads_performed",
                )
    if version is None:  # type narrowing and corrupt database safeguard
        raise SourceVersionError(f"No source version is available for {source.source_id}")
    source = await registry.get_source(source.source_id) or source
    if before_source is None:
        _inc(telemetry, "sources_new")
    elif created:
        _inc(telemetry, "sources_changed")
    else:
        _inc(telemetry, "sources_reused")

    decisions: list[CacheDecision] = []
    parser_fp = _fingerprint_value(
        parser_fingerprint
        or descriptor.provider_metadata.get("parser_fingerprint")
        or build_parser_fingerprint(
        processor_name=f"{descriptor.source_type}_content_parser", processor_version="1.0",
        content_mime_type=descriptor.content_mime_type,
        )
    )
    parsed, decision = await _artifact_stage(
        registry=registry, version_id=version.version_id, source_id=source.source_id,
        artifact_type="parsed_text", fingerprint=parser_fp,
        input_hash=version.normalized_content_hash, policy=cache_policy,
        compute=lambda: _parse_content(parser, raw_content, version.raw_content, descriptor),
        metadata={"actual_input_kind": "raw_content"},
    )
    decisions.append(decision)
    _count_decision(telemetry, decision, "parses_reused", "parses_computed")
    parsed_text = _artifact_text(parsed)
    parsed_dependency_hash = _dependency_hash(parsed.output_hash, parsed.processor_fingerprint)

    summary: ArtifactRecord | None = None
    if summarizer is not None:
        summary_fp = _fingerprint_value(summary_fingerprint or build_summary_fingerprint(model="unknown"))
        summary, decision = await _artifact_stage(
            registry=registry, version_id=version.version_id, source_id=source.source_id,
            artifact_type="summary", fingerprint=summary_fp,
            input_hash=parsed_dependency_hash, policy=cache_policy,
            compute=lambda: _summarize(summarizer, parsed_text, descriptor),
            metadata={"actual_input_kind": "parsed_text"},
        )
        decisions.append(decision)
        _count_decision(telemetry, decision, "summaries_reused", "summaries_computed")

    static_records: list[ArtifactRecord] = []
    for name, analyzer in (static_analyzers or {}).items():
        fingerprint = _fingerprint_value(
            (static_analysis_fingerprints or {}).get(name)
            or build_static_analysis_fingerprint(analyzer_name=name, analyzer_version="1.0")
        )
        record, decision = await _artifact_stage(
            registry=registry, version_id=version.version_id, source_id=source.source_id,
            artifact_type=name, fingerprint=fingerprint, input_hash=parsed_dependency_hash,
            policy=cache_policy, compute=lambda analyzer=analyzer: _call(analyzer, parsed_text),
            metadata={"actual_input_kind": "parsed_text"},
        )
        static_records.append(record); decisions.append(decision)
        _count_decision(telemetry, decision, "static_analyses_reused", "static_analyses_computed")

    embedding_record = None
    if embedder is not None:
        input_artifact = summary or parsed
        input_kind = "summary" if summary is not None else "parsed_text"
        embedding_text = _artifact_text(input_artifact)
        embedding_input_hash = _dependency_hash(input_artifact.output_hash, input_artifact.processor_fingerprint)
        embedding_fp = _fingerprint_value(
            embedding_fingerprint or build_embedding_fingerprint(model="unknown", dimension=registry.embed_dim)
        )
        exact = await registry.find_embedding(
            version.version_id, input_kind, embedding_fp, embedding_input_hash
        )
        latest = exact or await registry.find_latest_embedding(version.version_id, input_kind)
        decision = decide_cache_action(
            "embedding", latest, embedding_input_hash, embedding_fp, cache_policy,
            source_id=source.source_id, version_id=version.version_id,
        )
        key = f"embedding:{version.version_id}:{embedding_fp}:{embedding_input_hash}"
        if decision.decision == "cache_hit" and exact is not None:
            embedding_record = exact
        else:
            async with _SINGLE_FLIGHT.lock(key):
                winner = await registry.find_embedding(
                    version.version_id, input_kind, embedding_fp, embedding_input_hash
                )
                if winner is not None and not cache_policy.rebuilds("embedding"):
                    embedding_record = winner
                    decision = decision.model_copy(update={
                        "decision": "cache_hit", "embedding_id": winner.embedding_id,
                        "reason": "A concurrent processor saved the compatible embedding.",
                    })
                else:
                    vector = await _call(embedder, embedding_text)
                    embedding_record = await registry.save_embedding(
                        version.version_id, input_kind, embedding_fp, embedding_input_hash,
                        [float(item) for item in vector], {"actual_input_kind": input_kind},
                    )
                    decision = decision.model_copy(update={"embedding_id": embedding_record.embedding_id})
        decisions.append(decision)
        _count_decision(telemetry, decision, "embeddings_reused", "embeddings_computed")

    artifact_ids = [str(record.artifact_id) for record in [parsed, summary, *static_records] if record]
    await registry.record_run_source(
        run_id, competition_id, source.source_id, version_id=version.version_id,
        artifact_ids=artifact_ids,
        embedding_id=embedding_record.embedding_id if embedding_record else None,
        cache_decisions={item.stage: item.decision for item in decisions},
    )
    if telemetry is not None:
        telemetry.per_source.append({
            "source_id": source.source_id, "version_id": str(version.version_id),
            "source_revision": version.source_revision, "source_status": source.source_status,
            "search_cache_decision": None,
            "download_decision": "cache_hit" if reuse_download else "content_checked",
            "parse_decision": _decision_for(decisions, "parsed_text"),
            "summary_decision": _decision_for(decisions, "summary"),
            "embedding_decision": _decision_for(decisions, "embedding"),
            "static_analysis_decisions": {
                item.stage: item.decision for item in decisions
                if item.stage not in {"parsed_text", "summary", "embedding"}
            },
            "warnings": warnings,
        })
    return SourceProcessingResult(
        source=source, version=version, parsed_artifact=parsed, summary_artifact=summary,
        static_analysis_artifacts=static_records, embedding=embedding_record,
        cache_decisions=decisions, warnings=warnings,
    )


async def process_source_documents(
    docs: list[SourceDocument], *, competition_id: str, run_id: str, registry: Any,
    cache_policy: CachePolicy, summarize_one: Callable[[SourceDocument], Any],
    embed_one: Callable[[str], Any], summary_model: str, embed_model: str,
    embed_many: Callable[[list[str]], Any] | None = None,
    telemetry: CacheRunTelemetry | None = None,
) -> tuple[list[SourceDocument], list[list[float]], list[SourceProcessingResult]]:
    from kaggle_researcher.summarizer import SYSTEM_PROMPT
    from kaggle_researcher.source_registry.static_analysis import (
        analyze_notebook_text,
        analyze_repository_text,
    )

    summary_fp = build_summary_fingerprint(model=summary_model, prompt=SYSTEM_PROMPT)
    embedding_fp = build_embedding_fingerprint(model=embed_model, dimension=registry.embed_dim)
    coordinated_embedder: Callable[[str], Any] = embed_one
    if embed_many is not None:
        coordinated_embedder = _BatchEmbedder(embed_many).submit

    async def one(doc: SourceDocument) -> tuple[SourceDocument, SourceProcessingResult]:
        descriptor = descriptor_from_document(doc)

        async def summarize_text(text: str, _descriptor: SourceDescriptor) -> Any:
            candidate = doc.model_copy(update={"content": text})
            summarized = await _call(summarize_one, candidate)
            summary_text = summarized.summary or summarized.content
            if summarized.metadata.get("summary_fallback_used"):
                return {
                    "text": summary_text,
                    "_cache_metadata": {
                        "fallback_used": True,
                        "fallback_reason": summarized.metadata.get("summary_fallback_reason"),
                        "actual_input_kind": summarized.metadata.get("summary_actual_input_kind", "parsed_text"),
                    },
                }
            return summary_text

        result = await process_source(
            descriptor, competition_id, run_id, registry, cache_policy,
            parser=_document_parser(doc),
            summarizer=summarize_text, embedder=coordinated_embedder,
            static_analyzers=(
                {"notebook_static_analysis": analyze_notebook_text}
                if doc.source == "kaggle"
                else {"repository_static_analysis": analyze_repository_text}
                if doc.source == "github"
                else {}
            ),
            summary_fingerprint=summary_fp, embedding_fingerprint=embedding_fp,
            telemetry=telemetry,
        )
        content = _artifact_text(result.parsed_artifact)
        summary_text = _artifact_text(result.summary_artifact) if result.summary_artifact else None
        updated = doc.model_copy(update={
            "id": result.source.source_id, "content": content, "summary": summary_text,
            "metadata": {**doc.metadata, "source_id": result.source.source_id,
                         "version_id": str(result.version.version_id)},
        })
        return updated, result

    pairs = await asyncio.gather(*(one(doc) for doc in docs))
    updated_docs = [pair[0] for pair in pairs]
    results = [pair[1] for pair in pairs]
    embeddings: list[list[float]] = []
    for result in results:
        if result.embedding is None:
            raise ArtifactCacheError(f"Missing embedding for {result.source.source_id}")
        vector = result.embedding.embedding
        if vector is None and hasattr(registry, "get_embedding_vector"):
            vector = await registry.get_embedding_vector(result.embedding.embedding_id)
        if vector is None:
            raise ArtifactCacheError(f"Cached embedding vector is unavailable for {result.source.source_id}")
        embeddings.append(vector)
    return updated_docs, embeddings, results


class _BatchEmbedder:
    def __init__(self, embed_many: Callable[[list[str]], Any]) -> None:
        self.embed_many = embed_many
        self.pending: list[tuple[str, asyncio.Future[list[float]]]] = []
        self.scheduled = False
        self.lock = asyncio.Lock()

    async def submit(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        async with self.lock:
            self.pending.append((text, future))
            if not self.scheduled:
                self.scheduled = True
                loop.create_task(self._flush())
        return await future

    async def _flush(self) -> None:
        await asyncio.sleep(0)
        async with self.lock:
            batch = self.pending
            self.pending = []
            self.scheduled = False
        if not batch:
            return
        try:
            vectors = list(await _call(self.embed_many, [text for text, _ in batch]))
            if len(vectors) != len(batch):
                raise ArtifactCacheError("Embedding batch result count does not match input count")
            for (_, future), vector in zip(batch, vectors, strict=True):
                if not future.cancelled():
                    future.set_result([float(item) for item in vector])
        except BaseException as exc:
            for _, future in batch:
                if not future.cancelled():
                    future.set_exception(exc)


def descriptor_from_document(doc: SourceDocument) -> SourceDescriptor:
    metadata = dict(doc.metadata)
    external_id = (
        metadata.get("arxiv_id") or metadata.get("entry_id") or metadata.get("kernel_ref")
        or metadata.get("full_name") or metadata.get("id")
    )
    if external_id is None and ":" in doc.id:
        external_id = doc.id.split(":", 1)[1]
    revision = metadata.get("source_revision") or metadata.get("version") or metadata.get("commit_sha")
    return SourceDescriptor(
        source_type=doc.source, external_id=str(external_id) if external_id else None,
        canonical_url=str(doc.url) if doc.url else None, title=doc.title,
        source_revision=str(revision) if revision is not None else None,
        revision_is_reliable=bool(metadata.get("revision_is_reliable", revision is not None)),
        content_mime_type=str(metadata.get("content_mime_type") or "text/plain"),
        provider_metadata=metadata, discovery_query=str(metadata.get("discovery_query") or ""),
        discovery_rank=metadata.get("discovery_rank"), raw_content=doc.content,
    )


def _document_parser(doc: SourceDocument) -> Callable[..., Any] | None:
    if doc.source != "arxiv" or not doc.metadata.get("content_location"):
        return None

    def parse_arxiv_content(content: Any, descriptor: SourceDescriptor) -> str:
        expected = descriptor.provider_metadata.get("parser_fingerprint")
        if descriptor.provider_metadata.get("parsed_with_fingerprint") == expected:
            return str(content)
        from pathlib import Path
        from kaggle_researcher.parsers.pdf_parser import parse_pdf

        path = Path(str(descriptor.provider_metadata["content_location"]))
        if not path.is_file():
            raise SourceVersionError(f"Cached PDF file is unavailable for {descriptor.external_id or 'source'}")
        return parse_pdf(path)

    return parse_arxiv_content


async def _artifact_stage(*, registry: Any, version_id: Any, source_id: str, artifact_type: str,
                          fingerprint: str, input_hash: str, policy: CachePolicy,
                          compute: Callable[[], Any], metadata: dict[str, Any]) -> tuple[ArtifactRecord, CacheDecision]:
    exact = await registry.find_artifact(version_id, artifact_type, fingerprint, input_hash)
    latest = exact or await registry.find_latest_artifact(version_id, artifact_type)
    decision = decide_cache_action(
        artifact_type, latest, input_hash, fingerprint, policy,
        source_id=source_id, version_id=version_id,
    )
    if decision.decision == "cache_hit" and exact is not None and _artifact_is_valid(exact):
        return exact, decision
    key = f"{artifact_type}:{version_id}:{fingerprint}:{input_hash}"
    async with _SINGLE_FLIGHT.lock(key):
        winner = await registry.find_artifact(version_id, artifact_type, fingerprint, input_hash)
        if winner is not None and _artifact_is_valid(winner) and not policy.rebuilds(artifact_type):
            return winner, decision.model_copy(update={
                "decision": "cache_hit", "artifact_id": winner.artifact_id,
                "reason": "A concurrent processor saved the compatible artifact.",
            })
        output = await _call(compute)
        output_metadata: dict[str, Any] = {}
        if isinstance(output, dict) and "_cache_metadata" in output:
            output = dict(output)
            raw_metadata = output.pop("_cache_metadata")
            if isinstance(raw_metadata, dict):
                output_metadata = raw_metadata
        payload = output if isinstance(output, (dict, list)) else {"text": str(output)}
        output_hash = sha256_text(canonical_json(payload))
        record = await registry.save_artifact(
            version_id, artifact_type, fingerprint, input_hash, output_hash, payload=payload,
            metadata={**metadata, **output_metadata},
        )
        return record, decision.model_copy(update={"artifact_id": record.artifact_id})


async def _parse_content(parser: Callable[..., Any] | None, raw_content: Any,
                         stored_content: str | None, descriptor: SourceDescriptor) -> Any:
    content = raw_content if raw_content is not None else stored_content
    if content is None:
        raise SourceVersionError("Cached source version has neither inline nor supplied content")
    if parser is not None:
        return await _call(parser, content, descriptor)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


async def _summarize(summarizer: Callable[..., Any], text: str, descriptor: SourceDescriptor) -> Any:
    try:
        value = await _call(summarizer, text, descriptor)
        return value if isinstance(value, dict) else str(value)
    except Exception as exc:
        return {
            "text": text[:800],
            "_cache_metadata": {
                "fallback_used": True,
                "fallback_reason": type(exc).__name__,
                "actual_input_kind": "parsed_text",
            },
        }


async def _call(callable_or_result: Any, *args: Any) -> Any:
    value = callable_or_result(*args) if callable(callable_or_result) else callable_or_result
    return await value if inspect.isawaitable(value) else value


def _artifact_text(record: ArtifactRecord | None) -> str:
    if record is None:
        return ""
    payload = record.payload
    if isinstance(payload, dict):
        return str(payload.get("text") or payload.get("summary") or json.dumps(payload, ensure_ascii=False))
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


def _artifact_is_valid(record: ArtifactRecord) -> bool:
    return record.output_hash == sha256_text(canonical_json(record.payload))


def _dependency_hash(output_hash: str, fingerprint: str) -> str:
    return sha256_text(canonical_json({"output_hash": output_hash, "processor_fingerprint": fingerprint}))


def _fingerprint_value(value: ProcessorFingerprint | str) -> str:
    return value.fingerprint if isinstance(value, ProcessorFingerprint) else str(value)


def _hash_content_type(mime_type: str | None) -> str:
    if mime_type and ("pdf" in mime_type or "octet-stream" in mime_type):
        return "binary"
    return "text"


def _inc(telemetry: CacheRunTelemetry | None, field: str) -> None:
    if telemetry is not None:
        setattr(telemetry, field, getattr(telemetry, field) + 1)


def _count_decision(telemetry: CacheRunTelemetry | None, decision: CacheDecision,
                    reused_field: str, computed_field: str) -> None:
    if decision.decision == "cache_hit":
        _inc(telemetry, reused_field); _inc(telemetry, "estimated_avoided_operations")
    else:
        _inc(telemetry, computed_field)


def _decision_for(decisions: list[CacheDecision], stage: str) -> str | None:
    return next((item.decision for item in decisions if item.stage == stage), None)
