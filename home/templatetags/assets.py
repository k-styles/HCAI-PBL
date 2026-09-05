"""Static links that carry the file's own modification time.

The development server serves stylesheets with no version in the URL, so a
browser that has one cached keeps using it after an edit. The page then renders
new HTML against old CSS, which looks like a layout bug rather than a stale
file -- the confusing kind, because everything styled before the edit still
looks right.

`{% asset 'home/style.css' %}` appends the file's mtime, so the URL changes
whenever the file does and the browser refetches on its own.
"""

from pathlib import Path

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def asset(path):
    url = static(path)
    found = finders.find(path)
    if found:
        return f"{url}?v={int(Path(found).stat().st_mtime)}"
    return url


@register.filter
def maths(text):
    """Render inline formulas inside ordinary prose.

    Anything between dollar signs is typeset by home.mathfmt; the rest of the
    paragraph is escaped normally. This exists because a sentence that mentions
    x_A should not show a raw underscore -- the prose and the display formulas
    ought to look like the same notation.
    """
    from django.utils.html import escape
    from django.utils.safestring import mark_safe

    from home.mathfmt import render

    parts = str(text).split("$")
    out = []
    for i, part in enumerate(parts):
        # Odd indices are between a pair of dollars, so they are the formulas.
        out.append(render(part) if i % 2 else escape(part))
    return mark_safe("".join(out))
