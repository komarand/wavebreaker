from __future__ import annotations

from pathlib import Path

from docx import Document

from kaggle_researcher.report.docx_generator import generate_report
from kaggle_researcher.schemas import RetrievedDocument


def make_source() -> RetrievedDocument:
    return RetrievedDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="High scoring notebook",
        url="https://example.com/notebook",
        content="retrieved content",
        score=0.9,
        rrf_score=0.1234,
    )


def test_generate_report_writes_openable_docx_with_headings_bullets_and_sources(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "reports" / "comp-1.docx"

    result_path = generate_report(
        competition_name="comp-1",
        roadmap_text="\n".join(
            [
                "# Roadmap",
                "Overview paragraph.",
                "## Experiments",
                "- Build baseline",
                "* Try calibration",
            ]
        ),
        sources=[make_source()],
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    document = Document(output_path)
    paragraph_texts = [paragraph.text for paragraph in document.paragraphs]
    styles_by_text = {paragraph.text: paragraph.style.name for paragraph in document.paragraphs}

    assert "comp-1" in paragraph_texts
    assert "Roadmap" in paragraph_texts
    assert "Experiments" in paragraph_texts
    assert "Build baseline" in paragraph_texts
    assert "Try calibration" in paragraph_texts
    assert "Sources" in paragraph_texts
    assert any("High scoring notebook [kaggle]" in text for text in paragraph_texts)
    assert styles_by_text["Build baseline"] == "List Bullet"


def test_generate_report_includes_empty_sources_message(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.docx"

    generate_report("comp-1", "Plain report text.", [], output_path)

    paragraph_texts = [paragraph.text for paragraph in Document(output_path).paragraphs]
    assert "No retrieved sources." in paragraph_texts
