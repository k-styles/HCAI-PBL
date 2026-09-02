from django.urls import path

from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("visualize/", views.visualize, name="visualize"),
    path("train/", views.train, name="train"),
    path("problem-type/", views.retype, name="retype"),
    path("history/", views.history, name="history"),
    path("notes/", views.notes, name="notes"),
    path("notes/report.pdf", views.report, name="report"),
    path("explain/", views.explain_index, name="explain_index"),
    path("explain/<slug:slug>/", views.explain, name="explain"),
    path("reset/", views.reset, name="reset"),
]
