from django import template
from django.urls import reverse
from django.utils.html import format_html

from ..explain import TOPICS

register = template.Library()


@register.simple_tag
def info(slug):
    """The small "i" beside a field label."""
    topic = TOPICS.get(slug)
    if topic is None:
        return ""
    return format_html('<a class="p3-info" href="{}" title="{}">i</a>',
                       reverse("project3:explain", args=[slug]), topic.short)


@register.simple_tag
def term(slug, text=None):
    """A technical word linked in place, inside the sentence it appears in."""
    topic = TOPICS.get(slug)
    label = text or (topic.title.lower() if topic else slug)
    if topic is None:
        return label
    return format_html('<a class="p3-term" href="{}" title="{}">{}</a>',
                       reverse("project3:explain", args=[slug]), topic.short, label)


@register.filter
def dictget(mapping, key):
    """Look a dictionary up by a variable key.

    Django's template language can only do constant keys, and several tables
    here are naturally keyed by a loop variable -- the region number, the
    strategy name. The alternative is flattening every dictionary into a list of
    pairs in the view purely to satisfy the template, which moves the awkwardness
    rather than removing it.
    """
    if mapping is None:
        return None
    try:
        return mapping[key]
    except (KeyError, TypeError, IndexError):
        return mapping.get(str(key)) if hasattr(mapping, "get") else None
