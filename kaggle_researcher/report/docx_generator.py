from __future__ import annotations

from pathlib import Path

from docx import Document

from kaggle_researcher.schemas import RetrievedDocument


def generate_report(
    competition_name: str,
    roadmap_text: str,
    sources: list[RetrievedDocument],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    document.add_heading(competition_name, level=0)
    _add_markdown_like_text(document, roadmap_text)

    document.add_heading("Sources", level=1)
    if sources:
        for source in sources:
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.add_run(source.title).bold = True
            paragraph.add_run(f" [{source.source}]")
            if source.url is not None:
                paragraph.add_run(f" {source.url}")
            paragraph.add_run(f" rrf_score={source.rrf_score:.4f}")
    else:
        document.add_paragraph("No retrieved sources.")

    document.save(path)
    return path


def _add_markdown_like_text(document: Document, text: str) -> None:
    in_code_block = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if not stripped:
            continue

        if in_code_block:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(line)
            run.font.name = "Courier New"
        elif stripped.startswith("# "):
            document.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith("## "):
            document.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith(("- ", "* ")):
            document.add_paragraph(stripped[2:].strip(), style="List Bullet")
        else:
            document.add_paragraph(stripped)
