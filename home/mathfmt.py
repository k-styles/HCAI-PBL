"""Render the short formulas used in the explanation pages as real HTML.

Not LaTeX. These pages must work with no network and no JavaScript, which rules
out MathJax and KaTeX, and vendoring a megabyte of fonts to typeset thirty
short expressions is not a trade worth making for a coursework repo.

What is here is the small part of maths typesetting those expressions need:
true subscripts and superscripts, italic variables, upright function names, and
spacing around relations. Integrals, sums, Greek letters and partial
derivatives are already single unicode characters and pass through untouched.

Notation accepted in the source strings:

    x_A          subscript, one token
    w_{c,j}      subscript, braced -- contents are rendered recursively
    e^{U_i}      superscript, braced
    R^2          superscript, one token

The convention followed is the usual one: quantities in italic, names of
functions and operators upright, so that `max` reads as an operator rather than
as m times a times x.
"""

import re

from django.utils.html import escape
from django.utils.safestring import mark_safe

# Upright because each names a function, an operator or a word, not a quantity.
# Sorted longest first so `argmax` wins over `arg` and `max`.
UPRIGHT = sorted({
    "argmin", "argmax", "softmax", "precision", "recall", "variance", "between",
    "within", "simplest", "responses", "rejected", "deferred", "chosen",
    "needed", "choose", "score", "where", "class", "differ", "every", "such",
    "that", "with", "and", "for", "not", "the", "of", "if", "is",
    "exp", "log", "max", "min", "med", "cov", "var", "std", "sign", "diag",
    "acc", "tfidf", "tf", "idf", "df", "MSE", "MAE", "RMSE", "MAD", "ALE",
    "PDP", "PD", "SE", "true", "team", "test", "emp", "opt",
}, key=len, reverse=True)

_UPRIGHT_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, UPRIGHT)) + r")(?![A-Za-z])")
_SUB_BRACED = re.compile(r"_\{([^{}]*)\}")
_SUP_BRACED = re.compile(r"\^\{([^{}]*)\}")
_SUB_ONE = re.compile(r"_([A-Za-z0-9′*]+)")
_SUP_ONE = re.compile(r"\^([A-Za-z0-9′*⁻+−]+)")
_VAR_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])")
_REL_RE = re.compile(r"\s*(=|≈|≤|≥|≠|∈|→|⁄|±|≻)\s*")

_SLOT = "\x00%d\x00"
_SLOT_RE = re.compile(r"\x00(\d+)\x00")


def render(source):
    """One formula in, safe HTML out."""
    return mark_safe(_render(escape(source), []))


def _render(text, slots):
    """Every substitution parks its HTML in `slots` and leaves a placeholder
    behind, so no generated markup is ever visible to a later pattern -- which
    is what goes wrong if you let an inserted <i> reach the relation pass."""

    def park(html):
        slots.append(html)
        return _SLOT % (len(slots) - 1)

    def script(tag, inner):
        # Braced contents can hold their own scripts, so recurse.
        return park(f"<{tag}>{_render(inner, slots)}</{tag}>")

    text = _SUB_BRACED.sub(lambda m: script("sub", m.group(1)), text)
    text = _SUP_BRACED.sub(lambda m: script("sup", m.group(1)), text)
    text = _SUB_ONE.sub(lambda m: script("sub", m.group(1)), text)
    text = _SUP_ONE.sub(lambda m: script("sup", m.group(1)), text)
    text = _UPRIGHT_RE.sub(lambda m: park(f'<span class="mf-op">{m.group(1)}</span>'), text)
    text = _VAR_RE.sub(lambda m: park(f"<i>{m.group(1)}</i>"), text)
    text = _REL_RE.sub(lambda m: park(f'<span class="mf-rel">{m.group(1)}</span>'), text)

    # Placeholders can nest, so keep expanding until none are left.
    while _SLOT_RE.search(text):
        text = _SLOT_RE.sub(lambda m: slots[int(m.group(1))], text)
    return text
