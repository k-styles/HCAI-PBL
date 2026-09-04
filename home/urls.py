from django.urls import path
from . import views

app_name = 'home'

urlpatterns = [
    path("", views.index, name="index"),
    path("own-work/", views.own_work, name="own_work"),
    path("notes/", views.notes, name="notes"),
    path("report.pdf", views.report, name="report"),
    path("ai-usage.pdf", views.ai_usage, name="ai_usage"),
]
