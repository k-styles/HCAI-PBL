from django.urls import path

from . import views

app_name = "project2"

urlpatterns = [
    path("", views.index, name="index"),
    path("counterfactual/", views.counterfactual, name="counterfactual"),
    path("effects/", views.feature_effects, name="effects"),
    path("explain/", views.explain_index, name="explain_index"),
    path("explain/<slug:slug>/", views.explain, name="explain"),
]
