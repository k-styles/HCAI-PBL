from django.urls import path
from . import views

app_name = 'home'

urlpatterns = [
    path("", views.index, name="index"),
    path("own-work/", views.own_work, name="own_work"),
]
