from django.urls import path

from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("defer/", views.defer, name="defer"),
    path("active/", views.active, name="active"),
    path("report.pdf", views.report, name="report"),
    path("explain/", views.explain_index, name="explain_index"),
    path("explain/<slug:slug>/", views.explain, name="explain"),
]
