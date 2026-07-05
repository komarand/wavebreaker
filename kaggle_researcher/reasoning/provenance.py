from __future__ import annotations

from typing import Any


ALLOWED_PROVENANCE = {
    "kaggle",
    "arxiv",
    "github",
    "huggingface_papers",
    "domain_memory",
    "heuristic",
    "not_verified_on_data",
}


def normalize_provenance(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    normalized: list[str] = []
    for item in values:
        label = str(item).strip().lower()
        if label in ALLOWED_PROVENANCE and label not in normalized:
            normalized.append(label)
    return normalized


def attach_default_provenance(
    section_name: str,
    section_data: dict[str, Any] | list[Any],
    retrieved_documents: list[Any],
) -> dict[str, Any] | list[Any]:
    available_sources = {
        str(getattr(doc, "source", doc.get("source") if isinstance(doc, dict) else ""))
        for doc in retrieved_documents
    }
    return _attach(section_name, section_data, available_sources)


def provenance_summary(sections: dict[str, Any]) -> dict[str, Any]:
    counts = {label: 0 for label in sorted(ALLOWED_PROVENANCE)}
    missing: list[str] = []
    for section_name, section_data in sections.items():
        labels = _collect_provenance(section_data)
        if labels:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        else:
            missing.append(section_name)
    return {
        "claims_by_provenance": counts,
        "sections_missing_provenance": missing,
        "top_supporting_sources_by_section": {},
    }


def _attach(section_name: str, value: Any, available_sources: set[str]) -> Any:
    if isinstance(value, list):
        return [_attach(section_name, item, available_sources) for item in value]
    if not isinstance(value, dict):
        return value

    copied = {key: _attach(section_name, item, available_sources) for key, item in value.items()}
    text = " ".join(str(item) for item in copied.values()).lower()
    provenance = normalize_provenance(copied.get("provenance"))

    if "kaggle" in available_sources and any(token in text for token in ("kaggle", "notebook", "public lb", "competition")):
        provenance.append("kaggle")
    if "kaggle" in available_sources and "leakage" in section_name.lower():
        provenance.append("kaggle")
    if "arxiv" in available_sources and any(token in text for token in ("gini", "auc", "calibration", "credit scoring")):
        provenance.append("arxiv")
    if "huggingface_papers" in available_sources and any(token in text for token in ("paper", "credit scoring", "default")):
        provenance.append("huggingface_papers")
    if "github" in available_sources and any(token in text for token in ("code", "repo", "implementation")):
        provenance.append("github")
    if any(token in section_name.lower() + " " + text for token in ("leakage", "feature", "dataset", "expected_gain", "train/test")):
        provenance.append("not_verified_on_data")
    if any(token in section_name.lower() + " " + text for token in ("experiment", "roadmap", "do_not", "avoid", "priority", "expected")):
        provenance.append("heuristic")

    normalized = normalize_provenance(provenance)
    if normalized:
        copied["provenance"] = normalized
    return copied


def _collect_provenance(value: Any) -> set[str]:
    labels: set[str] = set()
    if isinstance(value, dict):
        labels.update(normalize_provenance(value.get("provenance")))
        for item in value.values():
            labels.update(_collect_provenance(item))
    elif isinstance(value, list):
        for item in value:
            labels.update(_collect_provenance(item))
    return labels
