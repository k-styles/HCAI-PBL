from django.shortcuts import render

# --- EDIT ME: one entry per group member -------------------------------
GROUP = [
    {"name": "Kartik Anand", "matriculation": "676049"},
]
# -----------------------------------------------------------------------

PROJECTS = [
    {"name": "Project 1", "url_name": "project1:index"},
]


def index(request):
    return render(request, "home/index.html", {"students": GROUP, "projects": PROJECTS})
