import uuid

from django.shortcuts import redirect, render

from django.http import Http404

from home.own_work import for_project

from . import counterfactuals, data, effects, models_lab, plots, treeview
from .explain import TOPICS, groups

SESSION_KEY = "project2_choice"
DEFAULT = {"kind": "tree", "lam": 0.02}

# The slider's stops. Lambda is a rate of accuracy per unit of complexity, so
# the interesting range is small: at 0.1 a leaf has to buy ten points of
# accuracy to be worth having, which almost nothing does.
LAMBDA_STOPS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5]


def _choice(request):
    saved = dict(DEFAULT)
    saved.update(request.session.get(SESSION_KEY, {}))
    return saved


def _token(request):
    if "project2_token" not in request.session:
        request.session["project2_token"] = uuid.uuid4().hex[:10]
    return request.session["project2_token"]


def _selected(choice):
    """The model the current slider position picks, plus its context."""
    family = models_lab.family(choice["kind"])
    hull, bands = models_lab.frontier(family)
    winner = models_lab.choose(family, choice["lam"])
    return family, hull, bands, winner


def _shared(request, choice, family, hull, bands, winner):
    """The control strip every region shows, so the link between them is visible."""
    return {
        "penguins": data.load(),
        "choice": choice,
        "kind": choice["kind"],
        "lam": choice["lam"],
        "stops": LAMBDA_STOPS,
        "family": family,
        "hull": hull,
        "hull_omegas": [c.omega for c in hull],
        # objective() takes an argument, which a template cannot pass
        "objective_value": winner.objective(choice["lam"]),
        "bands": bands,
        "winner": winner,
        "omega_name": "number of leaves" if choice["kind"] == "tree"
                      else "features with a non-zero coefficient",
        "requirements": for_project(2),
        "token": _token(request),
    }


def _apply(request):
    """Read the model controls out of a POST and remember them."""
    choice = _choice(request)
    kind = request.POST.get("kind")
    if kind in ("tree", "logistic"):
        choice["kind"] = kind
    try:
        choice["lam"] = float(request.POST.get("lam", choice["lam"]))
    except (TypeError, ValueError):
        pass
    request.session[SESSION_KEY] = choice
    return choice


def index(request):
    """Tasks 1 to 3: the model, its accuracy, its complexity, and the slider."""
    if request.method == "POST":
        _apply(request)
        return redirect("project2:index")

    choice = _choice(request)
    family, hull, bands, winner = _selected(choice)
    token = _token(request)

    context = _shared(request, choice, family, hull, bands, winner)
    context.update({
        "page": "model",
        # Task 3's requirement is about logistic regression's complexity measure,
        # so it only belongs on the page when that is the model being shown.
        "page_requirements": [r for r in for_project(2) if r.task == "Task 3"]
                             if choice["kind"] == "logistic" else [],
        "frontier_url": plots.frontier(family, hull, winner, choice["lam"],
                                       context["omega_name"], token),
        "objective_url": plots.objective(family, hull, choice["lam"], token),
    })

    if choice["kind"] == "tree":
        context["tree_url"] = plots.tree_diagram(winner.estimator, context["penguins"], token)
        context["rules"] = treeview.flatten(treeview.as_rules(winner.estimator,
                                                              context["penguins"]))
    else:
        context["coefficients"] = _coefficient_table(winner, context["penguins"])
    return render(request, "project2/index.html", context)


def _coefficient_table(winner, penguins):
    """Which features the sparse model kept, and which it discarded."""
    model = winner.estimator
    rows = []
    for feature in penguins.features:
        columns = [i for i, source in enumerate(penguins.origin) if source == feature]
        weights = model.coef_[:, columns]
        strength = float(abs(weights).max())
        rows.append({
            "feature": data.PRETTY[feature],
            "used": strength > 1e-8,
            "strength": strength,
            "per_class": [{"label": c, "value": float(weights[i].flat[abs(weights[i]).argmax()])}
                          for i, c in enumerate(model.classes_)],
        })
    rows.sort(key=lambda r: -r["strength"])
    return rows


def counterfactual(request):
    """Task 4."""
    if request.method == "POST":
        _apply(request)
        request.session["project2_cf"] = {
            "example": request.POST.get("example", "0"),
            "target": request.POST.get("target", ""),
        }
        return redirect("project2:counterfactual")

    choice = _choice(request)
    family, hull, bands, winner = _selected(choice)
    penguins = data.load()
    saved = request.session.get("project2_cf", {})

    try:
        index_of = max(0, min(len(penguins.frame) - 1, int(saved.get("example", 0))))
    except (TypeError, ValueError):
        index_of = 0

    row = penguins.frame.iloc[index_of].to_dict()
    predicted = winner.estimator.predict(penguins.X[[index_of]])[0]
    target = saved.get("target") or next(c for c in penguins.classes if c != predicted)
    if target not in penguins.classes:
        target = next(c for c in penguins.classes if c != predicted)

    found, log = counterfactuals.search(winner.estimator, penguins, row, target, k=5)

    context = _shared(request, choice, family, hull, bands, winner)
    context.update({
        "page": "counterfactual",
        "page_requirements": [r for r in for_project(2) if r.task == "Section 2"],
        "example_index": index_of,
        "example": row,
        "example_rows": _example_choices(penguins),
        "actual": row[data.TARGET],
        "predicted": predicted,
        "target": target,
        "found": found,
        "log": log,
        "mads": data.numeric_mads(penguins),
        "already": predicted == target,
    })
    return render(request, "project2/counterfactual.html", context)


def _example_choices(penguins):
    out = []
    for i, row in penguins.frame.iterrows():
        out.append({"index": i,
                    "label": f"#{i} · {row[data.TARGET]} · bill {row['bill_length_mm']:.1f} mm"
                             f" · flipper {row['flipper_length_mm']:.0f} mm"})
    return out


def feature_effects(request):
    """Task 5."""
    if request.method == "POST":
        _apply(request)
        request.session["project2_feature"] = request.POST.get("feature", data.NUMERIC[0])
        return redirect("project2:effects")

    choice = _choice(request)
    family, hull, bands, winner = _selected(choice)
    penguins = data.load()

    feature = request.session.get("project2_feature", data.NUMERIC[0])
    if feature not in data.NUMERIC:
        feature = data.NUMERIC[0]

    curve = effects.curves_for(winner.estimator, penguins, feature, choice["kind"])

    context = _shared(request, choice, family, hull, bands, winner)
    context.update({
        "page": "effects",
        "page_requirements": [r for r in for_project(2) if r.task == "Task 5"],
        "feature": feature,
        "numeric": [(f, data.PRETTY[f]) for f in data.NUMERIC],
        "curve": curve,
        "effects_url": plots.effect_curves(curve, _token(request)),
    })
    return render(request, "project2/effects.html", context)


def explain_index(request):
    return render(request, "project2/explain_index.html",
                  {"page": "explain", "groups": groups(), "count": len(TOPICS)})


def explain(request, slug):
    topic = TOPICS.get(slug)
    if topic is None:
        raise Http404(f"No explanation for '{slug}'.")
    return render(request, "project2/explain.html", {
        "page": "explain",
        "topic": topic,
        "related": [TOPICS[s] for s in topic.related if s in TOPICS],
    })
