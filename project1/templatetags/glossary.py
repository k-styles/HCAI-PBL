from django import template
from django.urls import reverse
from django.utils.html import format_html

from ..explain import TOPICS

register = template.Library()


@register.simple_tag
def info(slug):
    """A small "i" next to a technical word, linking to its explanation.

    Every term the interface uses should be one click from a plain answer, so
    this needs to be cheap enough to put everywhere without cluttering the
    template.
    """
    topic = TOPICS.get(slug)
    if topic is None:
        return ""
    return format_html(
        '<a class="p1-info" href="{}" title="{}">i</a>',
        reverse("project1:explain", args=[slug]), topic.short)


@register.simple_tag
def term(slug, text=None):
    """A technical word, linked in place, inside the sentence it appears in.

    The "i" beside a field label is fine for a form, but most of what the user
    reads here is prose. Making the word itself the link means help arrives at
    the moment of confusion rather than requiring a trip to a glossary and a
    trip back. Hovering shows the one-line answer without leaving the page.
    """
    topic = TOPICS.get(slug)
    label = text or (topic.title.lower() if topic else slug)
    if topic is None:
        return label
    return format_html('<a class="p1-term" href="{}" title="{}">{}</a>',
                       reverse("project1:explain", args=[slug]), topic.short, label)
