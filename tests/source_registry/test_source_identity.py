import pytest

from kaggle_researcher.source_registry.errors import SourceIdentityError
from kaggle_researcher.source_registry.identity import canonicalize_source_identity


def test_provider_variants_have_global_identity() -> None:
    assert canonicalize_source_identity("arxiv", "2401.12345v1", None).source_id == canonicalize_source_identity(
        "arxiv", "https://arxiv.org/pdf/2401.12345v2.pdf", None
    ).source_id
    assert canonicalize_source_identity("github", None, "https://github.com/Owner/Repo.git/").source_id == canonicalize_source_identity(
        "github", None, "https://github.com/owner/repo/tree/main"
    ).source_id
    assert canonicalize_source_identity("kaggle", "User/Book/versions/1", None).source_id == canonicalize_source_identity(
        "kaggle", "user/book/versions/9", None
    ).source_id


def test_url_fallback_is_deterministic_and_warns() -> None:
    first = canonicalize_source_identity("other", None, "HTTPS://EXAMPLE.COM/path/?utm_source=x&a=1")
    second = canonicalize_source_identity("other", None, "https://example.com/path?a=1")
    assert first.source_id == second.source_id
    assert first.identity_basis == "canonical_url_hash"
    assert first.warnings


def test_empty_identity_is_rejected() -> None:
    with pytest.raises(SourceIdentityError):
        canonicalize_source_identity("arxiv", None, None)
