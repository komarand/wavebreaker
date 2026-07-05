from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kaggle_researcher.agents import kaggle_agent
from kaggle_researcher.agents.kaggle_agent import (
    build_kaggle_documents,
    get_notebook_content,
    search_notebooks,
)
from kaggle_researcher.schemas import SourceDocument


class FakeKaggleApi:
    list_outputs: list[list[Any]] = []
    pull_impl: Any = None
    instances: list["FakeKaggleApi"] = []

    def __init__(self) -> None:
        self.authenticated = False
        self.kernels_list_calls: list[dict[str, Any]] = []
        self.kernels_pull_calls: list[dict[str, Any]] = []
        FakeKaggleApi.instances.append(self)

    def authenticate(self) -> None:
        self.authenticated = True

    def kernels_list(self, **kwargs: Any) -> list[Any]:
        self.kernels_list_calls.append(kwargs)
        index = len(self.kernels_list_calls) - 1
        if index >= len(FakeKaggleApi.list_outputs):
            return []
        return FakeKaggleApi.list_outputs[index]

    def kernels_pull(
        self,
        kernel_ref: str,
        path: str,
        metadata: bool = False,
        quiet: bool = False,
    ) -> None:
        self.kernels_pull_calls.append(
            {
                "kernel_ref": kernel_ref,
                "path": path,
                "metadata": metadata,
                "quiet": quiet,
            }
        )
        if FakeKaggleApi.pull_impl is not None:
            FakeKaggleApi.pull_impl(kernel_ref, Path(path))


@pytest.fixture(autouse=True)
def fake_kaggle_api(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeKaggleApi.list_outputs = []
    FakeKaggleApi.pull_impl = None
    FakeKaggleApi.instances = []
    monkeypatch.setattr(kaggle_agent, "KaggleApi", FakeKaggleApi)


def kernel(ref: str, title: str, votes: int) -> SimpleNamespace:
    return SimpleNamespace(ref=ref, title=title, totalVotes=votes)


def test_search_notebooks_uses_competition_id_when_provided() -> None:
    FakeKaggleApi.list_outputs = [[kernel("alice/good", "Good", 12)]]

    results = search_notebooks(
        ["home credit query"],
        competition_id="home-credit-credit-risk-model-stability",
        max_notebooks=10,
    )

    assert results[0]["id"] == "alice/good"
    assert results[0]["url"] == "https://www.kaggle.com/code/alice/good"
    assert results[0]["metadata"]["competition_id"] == "home-credit-credit-risk-model-stability"
    api = FakeKaggleApi.instances[0]
    assert api.authenticated is True
    assert api.kernels_list_calls == [
        {
            "competition": "home-credit-credit-risk-model-stability",
            "page_size": 10,
            "sort_by": "voteCount",
            "language": "python",
        }
    ]


def test_search_notebooks_falls_back_to_queries_when_competition_returns_nothing() -> None:
    FakeKaggleApi.list_outputs = [
        [],
        [kernel("alice/query", "Query result", 8)],
    ]

    results = search_notebooks(["fallback query"], competition_id="comp-1", max_notebooks=5)

    assert [item["id"] for item in results] == ["alice/query"]
    assert FakeKaggleApi.instances[0].kernels_list_calls[1] == {
        "search": "fallback query",
        "page_size": 5,
        "sort_by": "voteCount",
        "language": "python",
    }


def test_search_notebooks_deduplicates_by_ref_and_sorts_by_votes() -> None:
    FakeKaggleApi.list_outputs = [
        [
            kernel("alice/good", "Good", 12),
            kernel("bob/low", "Low", 1),
        ],
        [
            kernel("alice/good", "Good updated", 15),
            kernel("cara/best", "Best", 30),
        ],
    ]

    results = search_notebooks(["query one", "query two"], max_notebooks=10)

    assert [item["kernel_ref"] for item in results] == ["cara/best", "alice/good", "bob/low"]
    assert results[1]["title"] == "Good updated"
    assert results[1]["total_votes"] == 15


def test_search_notebooks_respects_max_notebooks() -> None:
    FakeKaggleApi.list_outputs = [
        [
            kernel("a/a", "A", 3),
            kernel("b/b", "B", 2),
            kernel("c/c", "C", 1),
        ]
    ]

    results = search_notebooks(["query"], max_notebooks=2)

    assert [item["kernel_ref"] for item in results] == ["a/a", "b/b"]


def test_get_notebook_content_extracts_markdown_and_code_snippets() -> None:
    long_code = "x = 1\n" * 300
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Approach\n", "Use grouped CV."]},
            {"cell_type": "code", "source": long_code},
            {"cell_type": "raw", "source": "ignored"},
        ]
    }

    def pull_impl(kernel_ref: str, path: Path) -> None:
        (path / "kernel.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    FakeKaggleApi.pull_impl = pull_impl

    content = get_notebook_content("alice/kernel")

    assert "# Approach" in content
    assert "Use grouped CV." in content
    assert "x = 1" in content
    assert "ignored" not in content
    assert len(content.split("\n\n")[1]) == 800
    assert FakeKaggleApi.instances[0].kernels_pull_calls[0]["metadata"] is True


def test_get_notebook_content_reads_py_kernel_when_ipynb_is_missing() -> None:
    def pull_impl(kernel_ref: str, path: Path) -> None:
        (path / "script.py").write_text("print('hello from kernel')", encoding="utf-8")

    FakeKaggleApi.pull_impl = pull_impl

    assert get_notebook_content("alice/script") == "print('hello from kernel')"


def test_get_notebook_content_raises_clear_error_when_no_kernel_file_downloaded() -> None:
    def pull_impl(kernel_ref: str, path: Path) -> None:
        (path / "README.md").write_text("not a notebook", encoding="utf-8")

    FakeKaggleApi.pull_impl = pull_impl

    with pytest.raises(RuntimeError, match=r"Downloaded files: \['README.md'\]"):
        get_notebook_content("alice/no-notebook")


def test_get_notebook_content_raises_clear_error_on_pull_failure() -> None:
    def pull_impl(kernel_ref: str, path: Path) -> None:
        raise ValueError("not found")

    FakeKaggleApi.pull_impl = pull_impl

    with pytest.raises(RuntimeError, match="Kaggle pull failed for missing/kernel: not found"):
        get_notebook_content("missing/kernel")


def test_build_kaggle_documents_creates_valid_source_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kaggle_agent, "get_notebook_content", lambda kernel_ref: f"content for {kernel_ref}")

    documents = build_kaggle_documents(
        [
            {"ref": "alice/kernel", "title": "Alice Kernel", "totalVotes": "7", "url": ""},
            {"kernel_ref": "alice/kernel", "title": "Duplicate", "total_votes": 1},
            {"id": "bob/kernel", "title": "Bob Kernel", "total_votes": 9, "content": "ready"},
        ],
        competition_id="comp-1",
    )

    assert len(documents) == 2
    assert all(isinstance(document, SourceDocument) for document in documents)
    assert [document.metadata["kernel_ref"] for document in documents] == ["bob/kernel", "alice/kernel"]
    assert documents[0].content == "ready"
    assert documents[0].source == "kaggle"
    assert documents[0].metadata["total_votes"] == 9
