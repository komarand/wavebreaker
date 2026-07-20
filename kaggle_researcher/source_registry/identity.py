from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from kaggle_researcher.source_registry.errors import SourceIdentityError
from kaggle_researcher.source_registry.hashing import sha256_text
from kaggle_researcher.source_registry.schemas import CanonicalSourceIdentity


IDENTITY_VERSION = "1.0"
_TYPE_ALIASES = {
    "arxiv": "arxiv",
    "arxiv_article": "arxiv",
    "paperswithcode": "papers_with_code",
    "papers_with_code": "papers_with_code",
    "papers_with_code_legacy": "papers_with_code",
    "pwc": "papers_with_code",
    "kaggle": "kaggle",
    "kaggle_notebook": "kaggle",
    "github": "github",
    "github_repository": "github",
}
_ARXIV_RE = re.compile(r"(?i)(?:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z-]+/\d{7}))(v\d+)?")


def canonicalize_source_identity(
    source_type: str,
    external_id: str | None,
    url: str | None,
    metadata: dict | None = None,
) -> CanonicalSourceIdentity:
    metadata = metadata or {}
    normalized_type = _normalize_type(source_type)
    raw_external = str(external_id or "").strip()
    raw_url = str(url or "").strip() or None
    canonical_url = _canonicalize_url(raw_url) if raw_url else None

    stable_id: str | None
    if normalized_type == "arxiv":
        stable_id = _arxiv_id(raw_external) or _arxiv_id(raw_url or "")
        if stable_id:
            canonical_url = f"https://arxiv.org/abs/{stable_id}"
    elif normalized_type == "github":
        stable_id = _github_id(raw_external) or _github_id(raw_url or "")
        if stable_id:
            canonical_url = f"https://github.com/{stable_id}"
    elif normalized_type == "kaggle":
        stable_id = _kaggle_id(raw_external) or _kaggle_id(raw_url or "")
        if stable_id:
            canonical_url = f"https://www.kaggle.com/code/{stable_id}"
    elif normalized_type == "papers_with_code":
        api_id = metadata.get("api_id") or metadata.get("paper_id") or metadata.get("id")
        stable_id = _pwc_id(str(api_id or raw_external)) or _pwc_id(raw_url or "")
        if stable_id and canonical_url is None:
            canonical_url = f"https://paperswithcode.com/paper/{stable_id}"
    else:
        stable_id = _generic_external_id(raw_external)

    if stable_id:
        return CanonicalSourceIdentity(
            source_id=f"{normalized_type}:{stable_id}",
            source_type=normalized_type,
            external_id=stable_id,
            canonical_url=canonical_url,
            identity_version=IDENTITY_VERSION,
            identity_basis="stable_external_id",
            warnings=[],
        )
    if canonical_url:
        digest = sha256_text(canonical_url)
        return CanonicalSourceIdentity(
            source_id=f"{normalized_type}:url:{digest}",
            source_type=normalized_type,
            external_id=f"url:{digest}",
            canonical_url=canonical_url,
            identity_version=IDENTITY_VERSION,
            identity_basis="canonical_url_hash",
            warnings=["No stable external ID was available; identity uses a canonical URL hash."],
        )
    raise SourceIdentityError(f"Cannot derive a non-empty identity for source type {normalized_type!r}")


def _normalize_type(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        raise SourceIdentityError("Source type is empty")
    return _TYPE_ALIASES.get(normalized, normalized)


def _arxiv_id(value: str) -> str | None:
    candidate = value.strip().replace(".pdf", "")
    match = _ARXIV_RE.search(candidate)
    return match.group(1).lower() if match else None


def _github_id(value: str) -> str | None:
    candidate = value.strip()
    if "github.com" in candidate.lower():
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        parts = [part for part in parsed.path.split("/") if part]
    else:
        candidate = re.sub(r"^github:", "", candidate, flags=re.I)
        parts = [part for part in candidate.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repository = parts[0], re.sub(r"\.git$", "", parts[1], flags=re.I)
    if not owner or not repository or owner.lower() in {"topics", "search"}:
        return None
    return f"{owner.lower()}/{repository.lower()}"


def _kaggle_id(value: str) -> str | None:
    candidate = value.strip()
    if "kaggle.com" in candidate.lower():
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0].lower() in {"code", "kernels"}:
            parts = parts[1:]
    else:
        candidate = re.sub(r"^kaggle:", "", candidate, flags=re.I)
        parts = [part for part in candidate.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0].lower()}/{parts[1].lower()}"


def _pwc_id(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    if "paperswithcode.com" in candidate.lower():
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0].lower() in {"paper", "method", "dataset"}:
            return parts[1].lower()
        return None
    candidate = re.sub(r"^(?:papers_with_code|pwc):", "", candidate, flags=re.I)
    return candidate.strip("/ ").lower() or None


def _generic_external_id(value: str) -> str | None:
    cleaned = value.strip().strip("/")
    return cleaned or None


def _canonicalize_url(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    if not host:
        raise SourceIdentityError("Source URL has no host")
    port = parsed.port
    netloc = host if port is None or (scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted((key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                             if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source"}))
    return urlunparse((scheme, netloc, path, "", query, ""))
