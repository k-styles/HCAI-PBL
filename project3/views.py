import uuid
from pathlib import Path

import numpy as np
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.views.decorators.clickjacking import xframe_options_sameorigin

from home.own_work import for_project

from . import data, deferral, plots, prepare
from .explain import TOPICS, groups

SESSION_KEY = "project3_choice"
DEFAULT = {"expert": "desks", "tau": 0.0}

# Tau is a difference of two probabilities, so it lives in [-1, 1]; anything
# past about +0.4 hands over nothing at all.
TAU_STOPS = [-0.30, -0.20, -0.12, -0.06, -0.02, 0.0, 0.02, 0.06, 0.12, 0.20, 0.30, 0.45]

REPORT = Path(settings.BASE_DIR) / "project3" / "report" / "project3-report.pdf"


def _cache():
    try:
        return prepare.load_app()
    except FileNotFoundError:
        return None


def _choice(request):
    saved = dict(DEFAULT)
    saved.update(request.session.get(SESSION_KEY, {}))
    return saved


def _token(request):
    if "project3_token" not in request.session:
        request.session["project3_token"] = uuid.uuid4().hex[:10]
    return request.session["project3_token"]


def _apply(request):
    choice = _choice(request)
    if request.POST.get("expert") in ("desks", "reader"):
        choice["expert"] = request.POST["expert"]
    try:
        choice["tau"] = float(request.POST.get("tau", choice["tau"]))
    except (TypeError, ValueError):
        pass
    request.session[SESSION_KEY] = choice
    return choice


def _shared(request, app, choice):
    info = app["experts"][choice["expert"]]
    return {
        "app": app,
        "choice": choice,
        "expert_key": choice["expert"],
        "expert": info,
        "experts": app["experts"],
        "tau": choice["tau"],
        "stops": TAU_STOPS,
        "machine_accuracy": app["machine_accuracy"],
        "requirements": for_project(3),
        "token": _token(request),
        "has_report": REPORT.exists(),
    }


def _missing(request):
    return render(request, "project3/missing.html", {"page": "model"})


def index(request):
    """Tasks 1 and 2: the classifier, the expert, and where each is strong."""
    app = _cache()
    if app is None:
        return _missing(request)
    if request.method == "POST":
        _apply(request)
        return redirect("project3:index")

    choice = _choice(request)
    context = _shared(request, app, choice)
    info = context["expert"]
    token = context["token"]

    regions = sorted(info["regions"], key=lambda r: -r["advantage"])
    context.update({
        "page": "model",
        "page_requirements": [r for r in for_project(3) if r.task == "Task 2"],
        "regions": regions,
        "top_terms": app["top_terms"],
        "regions_url": plots.regions(info["regions"], token),
        "helpful_regions": [r for r in regions if r["worth_deferring"]],
        "n_rows": len(app["machine_ok_test"]),
    })
    return render(request, "project3/index.html", context)


def defer(request):
    """Task 3."""
    app = _cache()
    if app is None:
        return _missing(request)
    if request.method == "POST":
        _apply(request)
        return redirect("project3:defer")

    choice = _choice(request)
    context = _shared(request, app, choice)
    info = context["expert"]
    result = app["results"][choice["expert"]]
    token = context["token"]

    machine_ok = app["machine_ok_test"]
    expert_ok = info["test_ok"]
    handed_over = result["advantage_test"] > choice["tau"]

    scores = deferral.evaluate(handed_over, machine_ok, expert_ok)
    scores["tau"] = choice["tau"]
    scores["f1"] = deferral.f1(scores)

    cases = deferral.outcomes(machine_ok, expert_ok)
    counts = {k: int((v & handed_over).sum()) for k, v in cases.items()}

    headroom = result["oracle"]["accuracy"] - app["machine_accuracy"]
    context.update({
        "page": "defer",
        "page_requirements": [],
        "scores": scores,
        "counts": counts,
        "kept": {k: int((v & ~handed_over).sum()) for k, v in cases.items()},
        "sweep": result["sweep"],
        "never": result["never"],
        "always": result["always"],
        "oracle": result["oracle"],
        "by_confidence": result["by_confidence"],
        "headroom": headroom,
        "captured": (scores["gain"] / headroom) if headroom else 0.0,
        "tradeoff_url": plots.tradeoff(result["sweep"], result, scores,
                                       app["machine_accuracy"], token),
        "outcomes_url": plots.outcomes(counts, token),
    })
    return render(request, "project3/defer.html", context)


def active(request):
    """Task 4."""
    app = _cache()
    if app is None:
        return _missing(request)
    if request.method == "POST":
        _apply(request)
        return redirect("project3:active")

    choice = _choice(request)
    context = _shared(request, app, choice)
    result = app["results"][choice["expert"]]
    token = context["token"]

    curves = result["curves"]
    full = result["best"]["accuracy"]
    rows = []
    for i, budget in enumerate(app["budgets"]):
        row = {"budget": budget}
        for key in app["strategies"]:
            row[key] = curves[key][i]
        rows.append(row)

    final = {k: curves[k][-1] for k in app["strategies"]}
    lead = final["boundary"]["accuracy"] - final["random"]["accuracy"]

    context.update({
        "page": "active",
        "page_requirements": [r for r in for_project(3) if r.task == "Task 4"],
        "rows": rows,
        "strategies": app["strategies"],
        "final": final,
        "lead": lead,
        "lead_sd": max(final["boundary"]["accuracy_sd"], final["random"]["accuracy_sd"]),
        "full_information": full,
        "budgets": app["budgets"],
        "seeds": app["seeds"],
        "curves_url": plots.learning_curves(curves, app["strategies"],
                                            app["machine_accuracy"], full, token),
    })
    return render(request, "project3/active.html", context)


def explain_index(request):
    return render(request, "project3/explain_index.html",
                  {"page": "explain", "groups": groups(), "count": len(TOPICS)})


def explain(request, slug):
    topic = TOPICS.get(slug)
    if topic is None:
        raise Http404(f"No explanation for '{slug}'.")
    return render(request, "project3/explain.html", {
        "page": "explain",
        "topic": topic,
        "related": [TOPICS[s] for s in topic.related if s in TOPICS],
    })


@xframe_options_sameorigin
def report(request):
    """The brief requires the PDF report to be reachable from the interface."""
    if not REPORT.exists():
        raise Http404("The project 3 report has not been built yet.")
    return FileResponse(open(REPORT, "rb"), content_type="application/pdf",
                        filename=REPORT.name)
