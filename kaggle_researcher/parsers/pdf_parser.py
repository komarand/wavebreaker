from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

from kaggle_researcher.logging_utils import get_logger

try:  # pragma: no cover - exercised through monkeypatches when dependency is absent.
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]


logger = get_logger(__name__)


async def download_pdf(url: str, paper_id: str, cache_dir: str) -> Path | None:
    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{_safe_filename(paper_id)}.pdf"

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to download PDF %s: %s", url, exc)
        return None

    content_type = response.headers.get("content-type", "")
    if content_type and "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        logger.warning("Downloaded content for %s does not look like a PDF", url)
        return None

    target_path.write_bytes(response.content)
    return target_path


def extract_tables_as_text(page: Any) -> str:
    try:
        tables = page.extract_tables() or []
    except Exception as exc:  # pragma: no cover - pdfplumber table internals can vary.
        logger.warning("Failed to extract PDF tables: %s", exc)
        return ""

    lines: list[str] = []
    for table in tables:
        for row in table or []:
            cells = ["" if cell is None else str(cell).strip() for cell in row or []]
            if any(cells):
                lines.append(" | ".join(cells))

    return "\n".join(lines)


def parse_pdf(pdf_path: Path, max_chars: int = 8000) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to parse PDF files")

    parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = list(pdf.pages)
        selected_indexes = _selected_page_indexes(pages)

        for page_index in selected_indexes:
            page = pages[page_index]
            text = page.extract_text() or ""
            tables_text = extract_tables_as_text(page)
            page_parts = [part.strip() for part in (text, tables_text) if part and part.strip()]
            if page_parts:
                parts.append("\n".join(page_parts))

            current_text = "\n\n".join(parts)
            if len(current_text) >= max_chars:
                return current_text[:max_chars]

    return "\n\n".join(parts)[:max_chars]


def _selected_page_indexes(pages: list[Any]) -> list[int]:
    if not pages:
        return []

    indexes: set[int] = {0}
    indexes.update(range(min(3, len(pages))))
    indexes.update(range(max(0, len(pages) - 2), len(pages)))

    for index, page in enumerate(pages):
        try:
            if page.extract_tables():
                indexes.add(index)
        except Exception:
            continue

    return sorted(indexes)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "paper"
