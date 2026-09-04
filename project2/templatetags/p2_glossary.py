from django import template
from django.urls import reverse
from django.utils.html import format_html

from ..explain import TOPICS

register = template.Library()


@register.simple_tag
def info(slug):
    topic = TOPICS.get(slug)
    if topic is None:
        return ""
    return format_html('<a class="p2-info" href="{}" title="{}">i</a>',
                       reverse("project2:explain", args=[slug]), topic.short)


@register.simple_tag
def term(slug, text=None):
    topic = TOPICS.get(slug)
    label = text or (topic.title.lower() if topic else slug)
    if topic is None:
        return label
    return format_html('<a class="p2-term" href="{}" title="{}">{}</a>',
                       reverse("project2:explain", args=[slug]), topic.short, label)
