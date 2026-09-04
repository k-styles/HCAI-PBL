"""
URL configuration for pbl project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path, re_path
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path("home/", include("home.urls")),
    path("admin/", admin.site.urls),
    path("demos/", include("demos.urls")),
    path("project1/", include("project1.urls")),
    path("project2/", include("project2.urls")),
    path("project3/", include("project3.urls")),
    path("project4/", include("project4.urls")),
    # Figures are generated per session and served from disk. Django's own
    # static serve is not built for high traffic, which is not a concern for a
    # single-user demo, and it is the only thing that works with DEBUG off.
    re_path(r"^media/(?P<path>.*)$", serve, {"document_root": settings.MEDIA_ROOT}),
]