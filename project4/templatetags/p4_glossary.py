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
    return format_html('<a class="p4-info" href="{}" title="{}" target="_blank">i</a>',
                       reverse("project4:explain", args=[slug]), topic.short)


@register.simple_tag
def term(slug, text=None):
    topic = TOPICS.get(slug)
    label = text or (topic.title.lower() if topic else slug)
    if topic is None:
        return label
    return format_html('<a class="p4-term" href="{}" title="{}" target="_blank">{}</a>',
                       reverse("project4:explain", args=[slug]), topic.short, label)


@register.filter
def percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""
