from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report.pdf", views.report, name="report"),
    path("start/", views.start, name="start"),
    path("task/", views.task, name="task"),
    path("debrief/", views.debrief, name="debrief"),
    path("responses/", views.responses, name="responses"),
    path("features/", views.features, name="features"),
    path("explain/", views.explain_index, name="explain_index"),
    path("explain/<slug:slug>/", views.explain, name="explain"),
]
