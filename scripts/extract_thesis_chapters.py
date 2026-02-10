from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

PDF_PATH = Path("Narmin_Alakbarli_Master_Thesis_59906.pdf")
OUT_PATH = Path("docs/thesis_chapter_notes.md")


def main() -> None:
    reader = PdfReader(str(PDF_PATH))
    pages = [page.extract_text() or "" for page in reader.pages]

    chapter_starts: list[tuple[int, str]] = []
    pattern = re.compile(r"CHAPTER\s+(\d+)\s*-\s*([A-Z][A-Z\s&]+)", re.MULTILINE)
    for idx, text in enumerate(pages, start=1):
        m = pattern.search(text)
        if m:
            chapter_starts.append((idx, f"Chapter {m.group(1)} - {m.group(2).strip().title()}"))

    chapter_starts.sort(key=lambda x: x[0])

    lines: list[str] = ["# Thesis chapter notes", ""]
    lines.append(f"- Total pages: {len(pages)}")
    lines.append("")

    for i, (start_page, title) in enumerate(chapter_starts):
        end_page = chapter_starts[i + 1][0] - 1 if i + 1 < len(chapter_starts) else len(pages)
        block = "\n".join(pages[start_page - 1 : end_page])
        block = re.sub(r"\s+", " ", block)
        snippet = block[:2200].strip()
        lines.append(f"## {title}")
        lines.append(f"- Page range: {start_page}-{end_page}")
        lines.append("")
        lines.append(snippet + "...")
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
