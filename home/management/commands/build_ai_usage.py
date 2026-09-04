"""Turn AI_USAGE.md into a PDF, so the declaration can be handed in as one.

AI_USAGE.md stays the single source of truth -- it is what gets edited and what
the dev-notes page renders. This command exists so there is also a PDF without
anyone maintaining a second copy of the text by hand.

    manage.py build_ai_usage
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

TEXLIVE = "/Library/TeX/texbin/pdflatex"

SOURCE = Path(settings.BASE_DIR) / "AI_USAGE.md"
TARGET = Path(settings.BASE_DIR) / "docs" / "AI_USAGE.pdf"

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.6cm]{geometry}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{accent}{HTML}{275CB2}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\footnotesize\sffamily Use of AI tools}
\fancyhead[R]{\footnotesize\sffamily Kartik Anand, 676049}
\fancyfoot[C]{\footnotesize\thepage}
\setlength{\parskip}{0.55em}\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\setlength{\emergencystretch}{2.5em}
\makeatletter
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}{2.2ex \@plus.2ex}%
  {\normalfont\Large\sffamily\bfseries\color{accent}}}
\makeatother
\title{\vspace{-1.4cm}\sffamily\color{accent}\bfseries Use of AI Tools\\[0.25em]
  \Large\mdseries Human-Centric Artificial Intelligence, Projects 1--4}
\author{Kartik Anand \quad---\quad Matriculation 676049\\[0.3em]
  \small Hamburg University of Technology \quad SoSe 2026}
\date{\small\today}
\begin{document}\maketitle\thispagestyle{fancy}
"""

# Backslash first, so the replacements' own backslashes are not re-escaped.
SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
            ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]

# The prose uses a handful of non-ASCII characters that inputenc has no mapping
# for, and pdflatex stops dead on each one. Translating them here rather than
# rewriting AI_USAGE.md keeps the markdown readable as markdown.
UNICODE = {"\u2014": "---", "\u2013": "--", "\u2212": "$-$", "\u00a7": r"\S{}",
           "\u03a9": r"$\Omega$", "\u03bb": r"$\lambda$", "\u00d7": r"$\times$",
           "\u2248": r"$\approx$", "\u2192": r"$\rightarrow$",
           "\u2265": r"$\geq$", "\u2264": r"$\leq$",
           "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''"}


def escape(text):
    for char, replacement in SPECIALS:
        text = text.replace(char, replacement)
    for char, replacement in UNICODE.items():
        text = text.replace(char, replacement)
    # Anything still non-ASCII would stop pdflatex on a line we cannot see, so
    # it is dropped loudly rather than silently breaking the build.
    return "".join(c if ord(c) < 128 else "?" for c in text)


def inline(text):
    """Bold and inline code, applied after escaping so markers survive it."""
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(r"\texttt{\small " + escape(part[1:-1]) + "}")
        elif part.startswith("**") and part.endswith("**") and len(part) > 3:
            out.append(r"\textbf{" + escape(part[2:-2]) + "}")
        else:
            out.append(escape(part))
    return "".join(out)


def convert(markdown):
    """The subset AI_USAGE.md actually uses: headings, tables, bullets, paragraphs."""
    body, bullets, table, para = [], [], [], []

    def flush_para():
        if para:
            body.append(inline(" ".join(para)) + "\n")
            para.clear()

    def flush_bullets():
        if bullets:
            body.append(r"\begin{itemize}\setlength{\itemsep}{0.2em}")
            body.extend(r"\item " + inline(b) for b in bullets)
            body.append(r"\end{itemize}")
            bullets.clear()

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not set("-: |").issuperset(r)]
        if rows:
            columns = len(rows[0].strip("|").split("|"))
            body.append(r"\begin{tabularx}{\textwidth}{" + "X" * columns + "}\\toprule")
            for i, row in enumerate(rows):
                cells = [inline(c.strip()) for c in row.strip("|").split("|")]
                cells += [""] * (columns - len(cells))
                body.append(" & ".join(cells[:columns]) + r" \\" +
                            (r"\midrule" if i == 0 else ""))
            body.append(r"\bottomrule\end{tabularx}")
        table.clear()

    def flush_all():
        flush_para(); flush_bullets(); flush_table()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para(); flush_bullets()
            table.append(stripped)
            continue
        flush_table()
        if not stripped:
            flush_para(); flush_bullets()
        elif stripped.startswith("#"):
            flush_all()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = inline(stripped.lstrip("#").strip())
            body.append((r"\section*{%s}" if level <= 2 else r"\paragraph{%s}") % title)
        elif stripped.startswith(("- ", "* ")):
            flush_para()
            bullets.append(stripped[2:])
        else:
            flush_bullets()
            para.append(stripped)
    flush_all()
    return PREAMBLE + "\n".join(body) + "\n\\end{document}\n"


class Command(BaseCommand):
    help = "Render AI_USAGE.md to docs/AI_USAGE.pdf"

    def handle(self, *args, **options):
        if not SOURCE.exists():
            raise CommandError(f"{SOURCE} not found.")
        # TeX Live first, deliberately. A MiKTeX pdflatex earlier on PATH will
        # be found by `which` and then stop dead on geometry, prompting for a
        # file name that nothing is there to answer.
        latex = next((p for p in (TEXLIVE, shutil.which("pdflatex"))
                      if p and Path(p).exists()), None)
        if latex is None:
            raise CommandError("pdflatex not found; install TeX Live or add it to PATH.")

        TARGET.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tex = Path(tmp) / "ai_usage.tex"
            tex.write_text(convert(SOURCE.read_text()), encoding="utf-8")
            for _ in range(2):        # twice, so the page count settles
                done = subprocess.run(
                    [latex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                    cwd=tmp, capture_output=True, text=True)
            if done.returncode != 0:
                tail = "\n".join(done.stdout.splitlines()[-25:])
                raise CommandError(f"pdflatex failed:\n{tail}")
            shutil.copy(Path(tmp) / "ai_usage.pdf", TARGET)
        self.stdout.write(self.style.SUCCESS(f"Wrote {TARGET}"))
