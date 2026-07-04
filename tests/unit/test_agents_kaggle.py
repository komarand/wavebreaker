from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.agents import kaggle_agent
from kaggle_researcher.agents.kaggle_agent import (
    build_kaggle_documents,
    get_notebook_content,
    search_notebooks,
)
from kaggle_researcher.schemas import SourceDocument


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["kaggle"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_search_notebooks_deduplicates_by_ref_and_sorts_by_votes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    csv_outputs = [
        "ref,title,totalVotes,url\nalice/good,Good,12,https://www.kaggle.com/code/alice/good\nbob/low,Low,1,\n",
        "ref,title,totalVotes,url\nalice/good,Good updated,15,https://www.kaggle.com/code/alice/good\ncara/best,Best,30,\n",
    ]

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        return completed(stdout=csv_outputs[len(calls) - 1])

    monkeypatch.setattr(kaggle_agent.subprocess, "run", fake_run)

    results = search_notebooks(["query one", "query two"], max_notebooks=10)

    assert [item["kernel_ref"] for item in results] == ["cara/best", "alice/good", "bob/low"]
    assert results[1]["title"] == "Good updated"
    assert results[1]["total_votes"] == 15
    assert calls[0][:4] == ["kaggle", "kernels", "list", "--search"]
    assert "--csv" in calls[0]


def test_search_notebooks_respects_max_notebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return completed(
            stdout=(
                "ref,title,totalVotes,url\n"
                "a/a,A,3,\n"
                "b/b,B,2,\n"
                "c/c,C,1,\n"
            )
        )

    monkeypatch.setattr(kaggle_agent.subprocess, "run", fake_run)

    results = search_notebooks(["query"], max_notebooks=2)

    assert [item["kernel_ref"] for item in results] == ["a/a", "b/b"]


def test_get_notebook_content_extracts_markdown_and_code_snippets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    long_code = "x = 1\n" * 120
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Approach\n", "Use grouped CV."]},
            {"cell_type": "code", "source": long_code},
            {"cell_type": "raw", "source": "ignored"},
        ]
    }

    class FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        (tmp_path / "kernel.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
        return completed()

    monkeypatch.setattr(kaggle_agent.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(kaggle_agent.subprocess, "run", fake_run)

    content = get_notebook_content("alice/kernel")

    assert "# Approach" in content
    assert "Use grouped CV." in content
    assert "x = 1" in content
    assert "ignored" not in content
    assert len(content.split("\n\n")[1]) == 500


def test_get_notebook_content_returns_empty_on_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return completed(stderr="not found", returncode=1)

    monkeypatch.setattr(kaggle_agent.subprocess, "run", fake_run)

    assert get_notebook_content("missing/kernel") == ""


def test_get_notebook_content_returns_empty_when_download_has_no_ipynb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        (tmp_path / "README.md").write_text("not a notebook", encoding="utf-8")
        return completed()

    monkeypatch.setattr(kaggle_agent.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(kaggle_agent.subprocess, "run", fake_run)

    assert get_notebook_content("alice/no-notebook") == ""


def test_build_kaggle_documents_creates_valid_source_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kaggle_agent, "get_notebook_content", lambda kernel_ref: f"content for {kernel_ref}")

    documents = build_kaggle_documents(
        [
            {"ref": "alice/kernel", "title": "Alice Kernel", "totalVotes": "7", "url": ""},
            {"kernel_ref": "alice/kernel", "title": "Duplicate", "total_votes": 1},
            {"kernel_ref": "bob/kernel", "title": "Bob Kernel", "total_votes": 9, "content": "ready"},
        ],
        competition_id="comp-1",
    )

    assert len(documents) == 2
    assert all(isinstance(document, SourceDocument) for document in documents)
    assert [document.metadata["kernel_ref"] for document in documents] == ["bob/kernel", "alice/kernel"]
    assert documents[0].content == "ready"
    assert documents[0].source == "kaggle"
    assert documents[0].metadata["total_votes"] == 9
