from __future__ import annotations

import re
from typing import Any


MIN_QUALITY_SCORE = 0.35
MAX_GENERIC_SOURCES_IN_CONTEXT = 3
PREFER_COMPETITION_SOURCES = True


OFF_TOPIC_KEYWORDS = (
    "nlp",
    "llm",
    "loan description",
    "fairness",
    "diabetes",
    "medical risk",
    "hamiltonian neural networks",
    "survey",
)

ACADEMIC_RELEVANT_KEYWORDS = (
    "credit scoring",
    "credit default",
    "default prediction",
    "tabular",
    "risk modeling",
    "calibration",
    "gini",
    "auc",
    "temporal robustness",
    "out-of-time",
)


def score_source_quality(
    doc: dict[str, Any],
    competition_id: str,
    plan_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = str(doc.get("source") or "")
    text = _doc_text(doc)
    competition_slug = competition_id.lower().replace("_", "-")
    competition_phrase = competition_slug.replace("-", " ")
    quality_score = 1.0
    quality_notes: list[str] = []
    specificity = "generic"
    evidence_type = _evidence_type(source)

    if source == "kaggle":
        if _contains_any(text, (competition_slug, competition_phrase)):
            quality_score += 1.0
            specificity = "competition_specific"
            quality_notes.append("Kaggle source appears competition-specific.")
        else:
            quality_score += 0.25
            specificity = "domain_specific"
            quality_notes.append("Kaggle notebook source.")
    elif source == "github":
        if _contains_any(text, (competition_slug, competition_phrase)):
            quality_score += 0.55
            specificity = "competition_specific"
            quality_notes.append("GitHub repo appears competition-specific.")
        elif _contains_any(text, ("credit risk", "credit default", "home credit")):
            quality_score -= 0.2
            specificity = "domain_specific"
            quality_notes.append("Domain-related GitHub repo, but not competition-specific.")
        else:
            quality_score -= 0.35
            quality_notes.append("Generic GitHub repo.")
    elif source in {"arxiv", "papers_with_code", "papers_with_code_legacy"}:
        if _contains_any(text, ACADEMIC_RELEVANT_KEYWORDS):
            quality_score += 0.35
            specificity = "domain_specific"
            quality_notes.append("Academic paper is relevant to credit scoring/metric robustness.")
        else:
            quality_score -= 0.2
            quality_notes.append("Academic source is generic.")
    elif source == "huggingface_papers":
        if _contains_any(text, ("credit scoring", "credit default", "default prediction", "credit risk")):
            quality_score += 0.15
            specificity = "domain_specific"
            quality_notes.append("HF paper is related to credit risk/default.")
        else:
            quality_score -= 0.35
            quality_notes.append("HF paper is generic/background.")

    if _contains_any(text, OFF_TOPIC_KEYWORDS):
        quality_score -= 0.55
        specificity = "off_topic" if specificity == "generic" else specificity
        quality_notes.append("Penalized for off-topic/generic keywords.")

    if re.search(r"\b20(3\d|4\d|5\d)\b", text):
        quality_score -= 0.2
        quality_notes.append("Penalized for suspicious future date.")

    quality_score = max(0.0, min(2.0, quality_score))
    if quality_score < MIN_QUALITY_SCORE:
        specificity = "off_topic"

    return {
        "quality_score": quality_score,
        "specificity": specificity,
        "evidence_type": evidence_type,
        "quality_notes": quality_notes,
    }


def annotate_source_quality(
    docs: list[dict[str, Any]],
    competition_id: str,
    plan_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for doc in docs:
        copied = dict(doc)
        copied["source_quality"] = score_source_quality(copied, competition_id, plan_data)
        annotated.append(copied)
    return annotated


def rerank_by_source_quality(
    docs: list[Any],
    competition_id: str,
    plan_data: dict[str, Any] | None = None,
) -> list[Any]:
    annotated_docs = []
    generic_count = 0
    for doc in docs:
        as_dict = _model_to_dict(doc)
        quality = score_source_quality(as_dict, competition_id, plan_data)
        rrf_score = float(as_dict.get("rrf_score") or 0.0)
        final_score = rrf_score * quality["quality_score"]
        metadata = dict(as_dict.get("metadata") or {})
        metadata.update(
            {
                **quality,
                "final_score": final_score,
            }
        )
        updated = _copy_doc(doc, metadata)
        if quality["quality_score"] < MIN_QUALITY_SCORE:
            continue
        if quality["specificity"] == "generic":
            generic_count += 1
            if generic_count > MAX_GENERIC_SOURCES_IN_CONTEXT:
                continue
        annotated_docs.append(updated)

    return sorted(
        annotated_docs,
        key=lambda item: float(getattr(item, "metadata", {}).get("final_score", 0.0)),
        reverse=True,
    )


def source_quality_summary(docs: list[Any]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_specificity: dict[str, int] = {}
    top_sources: list[dict[str, Any]] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = getattr(doc, "source", "unknown")
        specificity = metadata.get("specificity", "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        by_specificity[specificity] = by_specificity.get(specificity, 0) + 1
        top_sources.append(
            {
                "title": getattr(doc, "title", ""),
                "source": source,
                "rrf_score": getattr(doc, "rrf_score", 0.0),
                "quality_score": metadata.get("quality_score"),
                "final_score": metadata.get("final_score"),
                "specificity": specificity,
                "evidence_type": metadata.get("evidence_type"),
                "quality_notes": metadata.get("quality_notes", []),
            }
        )
    return {
        "counts_by_source": by_source,
        "counts_by_specificity": by_specificity,
        "top_sources": sorted(
            top_sources,
            key=lambda item: float(item.get("final_score") or 0.0),
            reverse=True,
        )[:20],
    }


def _doc_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    parts = [
        str(doc.get("title") or ""),
        str(doc.get("url") or ""),
        str(doc.get("content") or ""),
        " ".join(str(value) for value in metadata.values()),
    ]
    return " ".join(parts).lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _evidence_type(source: str) -> str:
    return {
        "kaggle": "kaggle_notebook",
        "github": "github_repo",
        "arxiv": "academic_paper",
        "papers_with_code": "academic_paper",
        "papers_with_code_legacy": "academic_paper",
        "huggingface_papers": "hf_paper",
    }.get(source, "unknown")


def _model_to_dict(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "model_dump"):
        return doc.model_dump(mode="json")
    return dict(doc)


def _copy_doc(doc: Any, metadata: dict[str, Any]) -> Any:
    if hasattr(doc, "model_copy"):
        return doc.model_copy(update={"metadata": metadata})
    copied = dict(doc)
    copied["metadata"] = metadata
    return copied
