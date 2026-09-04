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
