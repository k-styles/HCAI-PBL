from django.shortcuts import render

from .own_work import REQUIREMENTS, STATES, summary

# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 1, Task 1
#
#   "Get familiar with this app logic and edit the view to have the names
#    and matriculation numbers of all group members appear on the page.
#    Do this modification from the python, and not in the HTML!"
#
# The group lives here and reaches the page only through the context below.
# The template loops over it; nothing about the group is written in HTML.
# ══════════════════════════════════════════════════════════════════════════

# --- EDIT ME: one entry per group member ---------------------------------
GROUP = [
    {"name": "Kartik Anand", "matriculation": "676049"},
]
# -------------------------------------------------------------------------

PROJECTS = [
    {"name": "Project 1", "url_name": "project1:index",
     "subject": "Automated Machine Learning",
     "blurb": "Upload a table, look at it, and sweep a model's hyperparameter — "
              "with the split between your choices and the machine's made visible.",
     "state": "done"},
    {"name": "Project 2", "url_name": "project2:index",
     "subject": "Explainability",
     "blurb": "Penguins. Trade accuracy against how much of a model you have to read, "
              "ask what would have to change to flip a prediction, and see what each "
              "measurement actually does.",
     "state": "done"},
    {"name": "Project 3", "url_name": "project3:index",
     "subject": "Active Learning for Learning-to-Defer",
     "blurb": "News topic labelling where the system may answer or hand over to a human "
              "expert — and has to work out, from a small number of questions, when "
              "handing over is actually worth it.",
     "state": "wip"},
    {"name": "Project 4", "url_name": "project4:index",
     "subject": "Preference elicitation",
     "blurb": "Two ways of asking what films someone likes \u2014 pick one of two, or rank "
              "ten \u2014 and a study designed to find out which asks better.",
     "state": "done"},
]


def index(request):
    return render(request, "home/index.html", {"students": GROUP, "projects": PROJECTS})


def own_work(request):
    """What the briefs require us to write ourselves, and whether it is still there.

    The audit re-reads each named file and checks its marker comment survives,
    so this page reports the state of the code rather than a claim about it.
    """
    checked, counts = summary()
    return render(request, "home/own_work.html", {
        "checked": [(r, state, STATES[state]) for r, state in checked],
        "total": len(REQUIREMENTS),
        "counts": counts,
    })
