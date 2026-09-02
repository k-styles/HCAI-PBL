"""The project's own documentation, served from inside the app.

Development only: every view here 404s unless DEBUG is on, so none of it is
reachable from a deployed copy and none of it has to be remembered and stripped
out before handing the project in.
"""

import html
import re
from pathlib import Path

from django.conf import settings

# The report lives beside the repository during development; a self-contained
# copy inside the repo is checked first so a clone on its own still works.
REPORT_CANDIDATES = [
    Path(settings.BASE_DIR) / "docs" / "HCAI-report.pdf",
    Path(settings.BASE_DIR).parent / "report" / "HCAI-report.pdf",
]

AI_USAGE = Path(settings.BASE_DIR) / "AI_USAGE.md"


def report_path():
    for candidate in REPORT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def render_markdown(text):
    """Enough Markdown for the one file this renders, and no more.

    Headings, tables, bullets, paragraphs, bold and inline code -- which is
    everything AI_USAGE.md actually uses. Pulling in a Markdown library to
    display a single development page would be a dependency the project has to
    carry forever for no benefit.
    """
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r'<code class="p1-code">\1</code>', s)
        return s

    out, rows, bullets, para = [], [], [], []

    def flush_table():
        if not rows:
            return
        head, body = rows[0], [r for r in rows[1:] if not set("-: |").issuperset(r)]
        cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
        out.append('<div class="p1-scroll"><table class="p1-table"><thead><tr>'
                   + "".join(f"<th>{inline(c)}</th>" for c in cells(head))
                   + "</tr></thead><tbody>"
                   + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>"
                             for r in body)
                   + "</tbody></table></div>")
        rows.clear()

    def flush_bullets():
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def flush_all():
        flush_para(); flush_bullets(); flush_table()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            flush_all()
        elif stripped.startswith("|"):
            flush_para(); flush_bullets()
            rows.append(stripped)
        elif stripped.startswith("## "):
            flush_all(); out.append(f"<h3>{inline(stripped[3:])}</h3>")
        elif stripped.startswith("# "):
            flush_all(); out.append(f"<h2>{inline(stripped[2:])}</h2>")
        elif stripped.startswith("- "):
            flush_para(); flush_table()
            bullets.append(stripped[2:])
        else:
            flush_bullets(); flush_table()
            para.append(stripped)

    flush_all()
    return "".join(out)


def ai_usage_html():
    if not AI_USAGE.exists():
        return None
    return render_markdown(AI_USAGE.read_text())
