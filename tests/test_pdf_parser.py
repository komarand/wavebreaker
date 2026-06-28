from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from kaggle_researcher.parsers import pdf_parser
from kaggle_researcher.parsers.pdf_parser import download_pdf, extract_tables_as_text, parse_pdf


def run(coro):
    return asyncio.run(coro)


class FakePage:
    def __init__(self, text: str, tables: list[list[list[str | None]]] | None = None) -> None:
        self.text = text
        self.tables = tables or []

    def extract_text(self) -> str:
        return self.text

    def extract_tables(self) -> list[list[list[str | None]]]:
        return self.tables


class FakePdf:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "FakePdf":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def test_extract_tables_as_text_joins_cells_with_pipes() -> None:
    page = FakePage(
        text="",
        tables=[
            [["model", "score"], ["lgbm", "0.91"]],
            [["catboost", None]],
        ],
    )

    assert extract_tables_as_text(page) == "model | score\nlgbm | 0.91\ncatboost | "


def test_parse_pdf_returns_text_for_small_fixture_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = [
        FakePage("page 0"),
        FakePage("page 1"),
        FakePage("page 2"),
        FakePage("page 3 with table", tables=[[["feature", "importance"]]]),
        FakePage("page 4"),
    ]

    class FakePdfPlumber:
        @staticmethod
        def open(path: Path) -> FakePdf:
            assert path == pdf_path
            return FakePdf(pages)

    monkeypatch.setattr(pdf_parser, "pdfplumber", FakePdfPlumber)

    text = parse_pdf(pdf_path)

    assert "page 0" in text
    assert "page 1" in text
    assert "page 2" in text
    assert "page 3 with table" in text
    assert "feature | importance" in text
    assert "page 4" in text


def test_download_pdf_writes_to_cache_with_mocked_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda request: requests.append(request)
        or httpx.Response(200, content=b"%PDF content", headers={"content-type": "application/pdf"})
    )
    real_async_client = httpx.AsyncClient

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(pdf_parser.httpx, "AsyncClient", async_client_factory)

    path = run(download_pdf("https://example.com/paper.pdf", "2301.12345/unsafe", str(tmp_path)))

    assert path is not None
    assert path.exists()
    assert path.read_bytes() == b"%PDF content"
    assert requests[0].url == "https://example.com/paper.pdf"


def test_download_pdf_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    real_async_client = httpx.AsyncClient

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(pdf_parser.httpx, "AsyncClient", async_client_factory)

    assert run(download_pdf("https://example.com/missing.pdf", "missing", str(tmp_path))) is None
