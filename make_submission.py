"""Build the one required PDF from real deployment links and one BITS Lab image.

The builder does not create execution evidence, deploy an app, or prove that a
URL works. Open both links as an evaluator and supply an authentic lab screenshot.
"""

import argparse
import html
import re
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from common import ROOT


def validate_links(repository_url, app_url):
    repo = urlparse(repository_url)
    app = urlparse(app_url)
    if repo.scheme != "https" or repo.netloc.lower() != "github.com":
        raise ValueError("Repository URL must be an HTTPS github.com repository link.")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", repo.path):
        raise ValueError("Use a repository URL with exactly an owner and repository name.")
    if app.scheme != "https" or not (app.hostname or "").lower().endswith(".streamlit.app"):
        raise ValueError("Use the real app URL deployed on Streamlit Community Cloud (.streamlit.app).")
    if any(token in (repository_url + app_url).lower() for token in ["placeholder", "your-username", "your-app", "example."]):
        raise ValueError("Placeholder links are not accepted.")
    if repo.username or app.username or repo.query or repo.fragment:
        raise ValueError("Use clean public-facing links, without credentials or repository queries.")


def inline_markdown(text):
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#137885">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=colors.HexColor("#17343b"), spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="ReportBody", fontName="Helvetica", fontSize=9, leading=13,
        textColor=colors.HexColor("#243b40"), spaceAfter=6,
        allowWidows=0, allowOrphans=0,
    ))
    base.add(ParagraphStyle(
        name="ReportHeading", fontName="Helvetica-Bold", fontSize=12, leading=16,
        textColor=colors.HexColor("#137885"), spaceBefore=10, spaceAfter=6, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ReportSmall", parent=base["ReportBody"], fontSize=8, leading=11,
    ))
    base.add(ParagraphStyle(
        name="TableText", parent=base["ReportBody"], fontSize=7.6, leading=10,
        spaceAfter=0,
    ))
    base.add(ParagraphStyle(
        name="CodeText", fontName="Courier", fontSize=6.6, leading=9,
        backColor=colors.HexColor("#edf3f3"), borderPadding=6,
        leftIndent=6, rightIndent=6, spaceAfter=8,
    ))
    return base


def markdown_flowables(markdown, page_width, sty):
    """Render the complete project README, including tables and command blocks."""
    lines = markdown.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            output.append(Preformatted("\n".join(code), sty["CodeText"], maxLineLength=96))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append([Paragraph(inline_markdown(cell), sty["TableText"]) for cell in cells])
                index += 1
            columns = len(rows[0])
            if columns == 7:
                widths = [page_width * 0.25] + [page_width * 0.125] * 6
            elif columns == 2:
                widths = [page_width * 0.25, page_width * 0.75]
            else:
                widths = [page_width / columns] * columns
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d7e9ea")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7f7")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c1d3d5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            output.extend([table, Spacer(1, 8)])
            continue
        if line.startswith("#"):
            heading = line.lstrip("# ")
            output.append(Paragraph(inline_markdown(heading), sty["ReportHeading"]))
        elif line.startswith("- "):
            output.append(Paragraph("- " + inline_markdown(line[2:]), sty["ReportBody"]))
        else:
            paragraph = [line]
            while index + 1 < len(lines):
                following = lines[index + 1].strip()
                if not following or following.startswith(("#", "|", "```", "- ")):
                    break
                index += 1
                paragraph.append(following)
            output.append(Paragraph(inline_markdown(" ".join(paragraph)), sty["ReportBody"]))
        index += 1
    return output


def write_pdf(repository_url, app_url, screenshot, readme, output):
    sty = styles()
    page_width, page_height = A4
    margin = 18 * mm
    usable_width = page_width - 2 * margin
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=margin, leftMargin=margin,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Machine Learning Assignment 2 - Steel Plates Faults", author="",
    )
    story = [
        Paragraph("Machine Learning | Assignment 2", sty["ReportTitle"]),
        Paragraph("M.Tech (AIML) - Steel Plates Fault Classification", sty["ReportBody"]),
        Paragraph("1. GitHub Repository Link", sty["ReportHeading"]),
        Paragraph(f'<link href="{html.escape(repository_url)}" color="#137885">{html.escape(repository_url)}</link>', sty["ReportBody"]),
        Paragraph("2. Live Streamlit App Link", sty["ReportHeading"]),
        Paragraph(f'<link href="{html.escape(app_url)}" color="#137885">{html.escape(app_url)}</link>', sty["ReportBody"]),
        Paragraph("3. BITS Virtual Lab Execution Screenshot", sty["ReportHeading"]),
    ]
    with PILImage.open(screenshot) as image:
        width, height = image.size
    # One screenshot only, scaled proportionally without cropping or alteration.
    max_height = 156 * mm
    scale = min(usable_width / width, max_height / height)
    story.append(Image(str(screenshot), width=width * scale, height=height * scale, hAlign="LEFT"))
    story.extend([
        Spacer(1, 6),
        Paragraph("Execution screenshot supplied from the BITS Virtual Lab run.", sty["ReportSmall"]),
        PageBreak(),
        Paragraph("4. GitHub README Content", sty["ReportTitle"]),
    ])
    story.extend(markdown_flowables(readme, usable_width, sty))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c1d3d5"))
        canvas.line(margin, 13 * mm, page_width - margin, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52686e"))
        canvas.drawString(margin, 9 * mm, "Machine Learning Assignment 2 | PlateScope")
        canvas.drawRightString(page_width - margin, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--app-url", required=True)
    parser.add_argument("--lab-screenshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "submission" / "ML_Assignment_2.pdf")
    args = parser.parse_args()
    try:
        validate_links(args.repository_url, args.app_url)
        screenshot = args.lab_screenshot.resolve(strict=True)
        if screenshot.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            raise ValueError("Supply exactly one PNG or JPEG screenshot from BITS Virtual Lab.")
        with PILImage.open(screenshot) as image:
            image.verify()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if args.repository_url.rstrip("/") not in readme:
            raise ValueError("Update README section c with the real repository URL, then commit and push it.")
        if "Not published yet" in readme:
            raise ValueError("Remove the unpublished status from README section c before building the PDF.")
        for heading in ["## a. Problem statement", "## b. Dataset description", "## c. Github Repository Link", "## d. Models used"]:
            if heading not in readme:
                raise ValueError(f"Required README section missing: {heading}")
        from verify_project import main as verify
        verify()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_pdf(args.repository_url, args.app_url, screenshot, readme, args.output)
    except (ValueError, FileNotFoundError, OSError, AssertionError) as exc:
        parser.error(str(exc))
    print(f"Created {args.output.resolve()}")
    print("Open both links, inspect every PDF page, and SUBMIT the single PDF in the LMS.")


if __name__ == "__main__":
    main()
