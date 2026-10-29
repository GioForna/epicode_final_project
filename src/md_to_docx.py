"""Converte un Markdown semplice (titoli, elenchi, tabelle, grassetto) in .docx."""
import re
import sys

from docx import Document
from docx.shared import Pt


def add_inline(paragraph, text):
    # gestisce **grassetto** e *corsivo*
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            paragraph.add_run(part[1:-1]).italic = True
        elif part:
            paragraph.add_run(part)


def convert(md_path, docx_path):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    lines = open(md_path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    p = table.cell(r, c).paragraphs[0]
                    add_inline(p, cell)
                    if r == 0:
                        for run in p.runs:
                            run.bold = True
            doc.add_paragraph()
            continue
        m = re.match(r"(#+)\s+(.*)", line)
        if m:
            level = len(m.group(1))
            doc.add_heading(m.group(2), level=0 if level == 1 else level - 1)
        elif re.match(r"\d+\.\s", line):
            add_inline(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\.\s", "", line))
        elif line.lstrip().startswith("- "):
            add_inline(doc.add_paragraph(style="List Bullet"), line.lstrip()[2:])
        else:
            add_inline(doc.add_paragraph(), line)
        i += 1
    doc.save(docx_path)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    print("Creato", sys.argv[2])
