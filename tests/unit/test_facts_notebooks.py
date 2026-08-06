from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kaggle_researcher.facts.notebooks as notebooks_module
from kaggle_researcher.facts.notebooks import (
    list_competition_notebooks,
    pull_notebook,
    pull_notebook_with_diagnostics,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "kernel_list.json"


class FakeKaggleApi:
    list_pages: list[list[Any]] = []
    pull_impl: Callable[[str, Path], None] | None = None
    instances: list[FakeKaggleApi] = []

    def __init__(self) -> None:
        self.authenticated = False
        self.list_calls: list[dict[str, Any]] = []
        self.pull_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def authenticate(self) -> None:
        self.authenticated = True

    def kernels_list(self, **kwargs: Any) -> list[Any]:
        self.list_calls.append(kwargs)
        page = kwargs["page"]
        if page > len(self.__class__.list_pages):
            return []
        return self.__class__.list_pages[page - 1]

    def kernels_pull(
        self,
        kernel_ref: str,
        *,
        path: str,
        metadata: bool = False,
        quiet: bool = False,
    ) -> None:
        self.pull_calls.append(
            {
                "kernel_ref": kernel_ref,
                "path": path,
                "metadata": metadata,
                "quiet": quiet,
            }
        )
        if self.__class__.pull_impl is not None:
            self.__class__.pull_impl(kernel_ref, Path(path))


@pytest.fixture(autouse=True)
def fake_kaggle_api(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeKaggleApi.list_pages = []
    FakeKaggleApi.pull_impl = None
    FakeKaggleApi.instances = []

    def create_api() -> FakeKaggleApi:
        api = FakeKaggleApi()
        api.authenticate()
        return api

    monkeypatch.setattr(notebooks_module, "create_kaggle_api", create_api)
    monkeypatch.setattr(
        notebooks_module,
        "_NOTEBOOK_REQUEST_POLICY",
        notebooks_module.KaggleRequestPolicy(
            base_delay_seconds=0,
            min_interval_seconds=0,
        ),
    )


def _load_kernels() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_pull_retries_rate_limit_before_succeeding(tmp_path: Path) -> None:
    attempts = 0

    class RateLimited(RuntimeError):
        status = 429

    def pull_after_rate_limit(kernel_ref: str, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimited("rate limited")
        (destination / "retry.ipynb").write_text("{}", encoding="utf-8")

    FakeKaggleApi.pull_impl = pull_after_rate_limit

    path = pull_notebook("author/retry", tmp_path)

    assert path == tmp_path / "retry.ipynb"
    assert attempts == 3


def test_notebook_listing_retries_transient_rate_limit() -> None:
    class RateLimited(RuntimeError):
        status = 429

    class TransientApi:
        def __init__(self) -> None:
            self.calls = 0

        def kernels_list(self, **kwargs: Any) -> list[Any]:
            self.calls += 1
            if self.calls < 3:
                raise RateLimited("rate limited")
            return []

    api = TransientApi()

    assert list_competition_notebooks("example", 10, api=api) == []
    assert api.calls == 3


def test_list_parses_string_and_float_scores_deduplicates_and_sorts() -> None:
    FakeKaggleApi.list_pages = [_load_kernels()]

    notebooks = list_competition_notebooks("example-competition", max_notebooks=10)

    assert [notebook["ref"] for notebook in notebooks] == [
        "bob/high-score",
        "alice/baseline",
        "cara/no-score",
    ]
    assert notebooks[0]["public_score"] == pytest.approx(0.8456)
    assert isinstance(notebooks[0]["public_score"], float)
    assert notebooks[1]["title"] == "Baseline updated"
    assert notebooks[1]["votes"] == 18
    assert notebooks[1]["public_score"] == pytest.approx(0.82)
    assert notebooks[2]["public_score"] is None
    assert notebooks[2]["last_run"] is None

    api = FakeKaggleApi.instances[0]
    assert api.authenticated is True
    assert api.list_calls == [
        {
            "competition": "example-competition",
            "page": 1,
            "page_size": 10,
            "sort_by": "voteCount",
            "language": "python",
            "kernel_type": "notebook",
        }
    ]


def test_list_respects_limit_after_vote_sorting() -> None:
    FakeKaggleApi.list_pages = [_load_kernels()]

    notebooks = list_competition_notebooks("example-competition", max_notebooks=2)

    assert [notebook["ref"] for notebook in notebooks] == [
        "bob/high-score",
        "alice/baseline",
    ]


def test_list_fetches_another_page_when_duplicates_leave_room() -> None:
    kernels = _load_kernels()
    FakeKaggleApi.list_pages = [
        [kernels[0], kernels[1], kernels[2]],
        [kernels[3]],
    ]

    notebooks = list_competition_notebooks("example-competition", max_notebooks=3)

    assert [notebook["ref"] for notebook in notebooks] == [
        "bob/high-score",
        "alice/baseline",
        "cara/no-score",
    ]
    assert [call["page"] for call in FakeKaggleApi.instances[0].list_calls] == [1, 2]


def test_kaggle_2_notebook_response_uses_page_tokens() -> None:
    kernels = _load_kernels()

    class PaginatedApi:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def kernels_list_with_response(self, **kwargs: Any) -> Any:
            page_token = kwargs["page_token"]
            self.calls.append(page_token)
            if page_token is None:
                return SimpleNamespace(
                    kernels=[kernels[0], kernels[1], kernels[2]],
                    next_page_token="page-2",
                )
            return SimpleNamespace(
                kernels=[kernels[3]],
                next_page_token=None,
            )

    api = PaginatedApi()

    notebooks = list_competition_notebooks(
        "example-competition",
        max_notebooks=3,
        api=api,
    )

    assert api.calls == [None, "page-2"]
    assert [notebook["ref"] for notebook in notebooks] == [
        "bob/high-score",
        "alice/baseline",
        "cara/no-score",
    ]


def test_direct_kernels_response_object_also_uses_page_tokens() -> None:
    kernels = _load_kernels()

    class DirectResponseApi:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def kernels_list(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if "page_token" not in kwargs:
                return SimpleNamespace(
                    kernels=[kernels[0], kernels[1], kernels[2]],
                    next_page_token="page-2",
                )
            return SimpleNamespace(kernels=[kernels[3]], next_page_token=None)

    api = DirectResponseApi()

    notebooks = list_competition_notebooks(
        "example-competition",
        max_notebooks=3,
        api=api,
    )

    assert [call.get("page_token") for call in api.calls] == [None, "page-2"]
    assert len(notebooks) == 3


def test_list_supports_alternate_attribute_names_and_nested_author() -> None:
    FakeKaggleApi.list_pages = [
        [
            SimpleNamespace(
                kernel_ref="dana/alternate",
                kernel_title="Alternate fields",
                author={"username": "dana"},
                vote_count="1,234",
                public_score="0.7654",
                last_run="2026-07-29T08:00:00Z",
            )
        ]
    ]

    notebooks = list_competition_notebooks("example-competition", max_notebooks=10)

    assert notebooks == [
        {
            "ref": "dana/alternate",
            "title": "Alternate fields",
            "author": "dana",
            "votes": 1234,
            "public_score": pytest.approx(0.7654),
            "last_run": "2026-07-29T08:00:00Z",
        }
    ]


def test_non_positive_limit_does_not_create_api_client() -> None:
    assert list_competition_notebooks("example-competition", max_notebooks=0) == []
    assert FakeKaggleApi.instances == []


def test_list_normalizes_required_notebook_fields_and_skips_missing_ref(
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeKaggleApi.list_pages = [
        [
            {"ref": "alice/none", "title": None, "totalVotes": None},
            {"ref": "bob/empty", "title": "", "votes": "not-a-number"},
            {"ref": "cara/blank", "title": "   ", "total_votes": -5},
            {"title": "Missing ref", "totalVotes": 100},
        ]
    ]

    notebooks = list_competition_notebooks("example-competition", max_notebooks=10)

    assert [(item["ref"], item["title"], item["votes"]) for item in notebooks] == [
        ("alice/none", "alice/none", 0),
        ("bob/empty", "bob/empty", 0),
        ("cara/blank", "cara/blank", 0),
    ]
    assert "without a ref" in caplog.text


def test_pull_notebook_returns_downloaded_ipynb_path(tmp_path: Path) -> None:
    def write_notebook(_: str, dest: Path) -> None:
        nested = dest / "downloaded"
        nested.mkdir()
        (nested / "kernel.ipynb").write_text(
            json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
            encoding="utf-8",
        )

    FakeKaggleApi.pull_impl = write_notebook

    notebook_path = pull_notebook("alice/baseline", tmp_path)

    assert notebook_path == tmp_path / "downloaded" / "kernel.ipynb"
    assert notebook_path.is_file()
    assert FakeKaggleApi.instances[0].pull_calls == [
        {
            "kernel_ref": "alice/baseline",
            "path": str(tmp_path),
            "metadata": True,
            "quiet": True,
        }
    ]


def test_pull_failure_returns_none_without_raising(tmp_path: Path) -> None:
    def fail_pull(_: str, __: Path) -> None:
        raise RuntimeError("download failed")

    FakeKaggleApi.pull_impl = fail_pull

    assert pull_notebook("alice/missing", tmp_path) is None


def test_pull_failure_reports_http_status_and_terminal_attempt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    class RateLimited(RuntimeError):
        status = 429

    def fail_pull(_: str, __: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise RateLimited("rate limited")

    FakeKaggleApi.pull_impl = fail_pull

    result = pull_notebook_with_diagnostics("alice/rate-limited", tmp_path)

    assert result.path is None
    assert result.http_status == 429
    assert (result.attempt, result.max_attempts) == (3, 3)
    assert result.error_type == "RateLimited"
    assert attempts == 3
    assert (
        "Failed to pull Kaggle notebook alice/rate-limited " "(RateLimited, HTTP 429, attempt 3/3)"
    ) in caplog.text


def test_pull_without_ipynb_returns_none(tmp_path: Path) -> None:
    def write_script(_: str, dest: Path) -> None:
        (dest / "script.py").write_text("print('script')", encoding="utf-8")

    FakeKaggleApi.pull_impl = write_script

    assert pull_notebook("alice/script", tmp_path) is None


def test_pull_empty_ref_returns_none_without_creating_destination(tmp_path: Path) -> None:
    dest = tmp_path / "not-created"

    assert pull_notebook("", dest) is None
    assert not dest.exists()
    assert FakeKaggleApi.instances == []


def test_pull_retries_without_quiet_for_older_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class OlderKaggleApi(FakeKaggleApi):
        def kernels_pull(
            self,
            kernel_ref: str,
            *,
            path: str,
            metadata: bool = False,
            **kwargs: Any,
        ) -> None:
            if "quiet" in kwargs:
                raise TypeError("quiet is unsupported")
            (Path(path) / "older.ipynb").write_text("{}", encoding="utf-8")

    OlderKaggleApi.instances = []
    monkeypatch.setattr(notebooks_module, "create_kaggle_api", OlderKaggleApi)

    notebook_path = pull_notebook("alice/older", tmp_path)

    assert notebook_path == tmp_path / "older.ipynb"
