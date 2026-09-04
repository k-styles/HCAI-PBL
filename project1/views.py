import os
import time
import uuid

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.views.decorators.clickjacking import xframe_options_sameorigin

from . import data, learning, plots
from .explain import TOPICS, groups
from .forms import TrainingForm, UploadForm
from .models import TrainingRun

SESSION_KEY = "project1_dataset"
UPLOAD_DIR = "project1"


def _storage_path(filename):
    return os.path.join(settings.MEDIA_ROOT, UPLOAD_DIR, filename)


def _remembered(request):
    """Reload whatever the user uploaded earlier in this session.

    Only the filename lives in the session; the rows are re-read from disk each
    time.  Keeping a parsed DataFrame in the session would serialise the whole
    dataset on every single request, and these files are small enough that
    re-reading costs nothing.
    """
    note = request.session.get(SESSION_KEY)
    if not note:
        return None, None

    path = _storage_path(note["file"])
    if not os.path.exists(path):
        request.session.pop(SESSION_KEY, None)
        return None, None
    try:
        return data.load_csv(path, kind=note.get("kind")), note
    except data.DatasetError:
        request.session.pop(SESSION_KEY, None)
        return None, None


def _require_dataset(request):
    dataset, note = _remembered(request)
    if dataset is None:
        messages.info(request, "Upload a dataset first.")
    return dataset, note


def index(request):
    dataset, note = _remembered(request)

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.cleaned_data["file"]
            token = uuid.uuid4().hex[:12]
            filename = f"{token}.csv"
            os.makedirs(_storage_path(""), exist_ok=True)
            with open(_storage_path(filename), "wb") as target:
                for chunk in upload.chunks():
                    target.write(chunk)

            chosen_kind = form.cleaned_data["kind"] or None
            try:
                dataset = data.load_csv(_storage_path(filename), kind=chosen_kind)
            except data.DatasetError as problem:
                os.remove(_storage_path(filename))
                messages.error(request, str(problem))
                return render(request, "project1/index.html",
                              {"form": form, "dataset": None, "page": "data"})

            request.session[SESSION_KEY] = {
                "file": filename, "token": token,
                "name": upload.name, "kind": chosen_kind,
            }
            messages.success(request, f"Loaded {upload.name}: {dataset.n_rows} rows, "
                                      f"{len(dataset.features)} features.")
            return redirect("project1:visualize")
    else:
        form = UploadForm(initial={"kind": (note or {}).get("kind") or ""})

    return render(request, "project1/index.html",
                  {"form": form, "dataset": dataset, "note": note,
                   "name": (note or {}).get("name"), "page": "data"})


def visualize(request):
    dataset, note = _require_dataset(request)
    if dataset is None:
        return redirect("project1:index")

    features = dataset.features
    # On a regression problem the target is a number like any other column, and
    # plotting a feature straight against it is the plainest view of the thing
    # being predicted. For classes that plot would be a row of stripes, so the
    # axes stay on the features and colour does the work instead.
    axes = features + ([dataset.target] if dataset.kind == "regression" else [])

    x_name = request.GET.get("x") or features[0]
    y_name = request.GET.get("y") or axes[-1]
    if x_name not in axes:
        x_name = features[0]
    if y_name not in axes:
        y_name = axes[-1]
    if x_name == y_name and len(axes) > 1:
        y_name = next(a for a in axes if a != x_name)

    # A pair grid of every column would be unreadable on a wide table, so it is
    # capped at the features that say the most about the target on their own.
    relevance = data.feature_relevance(dataset)
    pair_names = [s["name"] for s in relevance["scores"][:6]]
    pair_names = [f for f in features if f in pair_names]

    token = note["token"]
    pca_url, pca_share = (plots.projection(dataset, token) if len(features) > 2
                          else (None, None))
    context = {
        "page": "look",
        "stamp": int(time.time()),
        "dataset": dataset,
        "name": note["name"],
        "note": note,
        "features": axes,
        "x_name": x_name,
        "y_name": y_name,
        "scatter_url": plots.scatter(dataset, token, x_name, y_name),
        "dist_url": plots.distributions(dataset, token),
        "corr_url": plots.correlations(dataset, token),
        "pair_url": plots.pair_grid(dataset, token, pair_names),
        "pair_names": pair_names,
        "pair_capped": len(features) > len(pair_names),
        "pca_url": pca_url,
        "pca_share": pca_share,
        "preview": data.preview(dataset),
        "relevance": relevance,
        "summary": dataset.describe(),
        "counts": ([{"label": str(c), "n": int((dataset.y == c).sum())} for c in dataset.classes()]
                   if dataset.kind == "classification" else None),
        "target_stats": (None if dataset.kind == "classification" else {
            "mean": dataset.y.mean(), "std": dataset.y.std(),
            "min": dataset.y.min(), "max": dataset.y.max()}),
    }
    return render(request, "project1/visualize.html", context)


def train(request):
    dataset, note = _require_dataset(request)
    if dataset is None:
        return redirect("project1:index")

    token = note["token"]
    result = leaderboard = None
    curve_url = fit_url = None
    automated = "automl" in request.POST

    if request.method == "POST" and automated:
        form = TrainingForm(dataset.kind)
        result, leaderboard = learning.automl(dataset)
        TrainingRun.record(note["name"], result, 0.25, 0, automated=True)
        messages.info(request, "Ran without asking you anything. Scroll down for what it decided.")

    elif request.method == "POST":
        form = TrainingForm(dataset.kind, request.POST)
        if form.is_valid():
            answers = form.cleaned_data
            try:
                result = learning.run(
                    dataset,
                    answers["algorithm"],
                    answers["score"],
                    answers["test_percent"] / 100,
                    answers["n_folds"],
                    answers["rule"],
                    answers["seed"],
                    grid=answers["values"],
                )
            except ValueError as problem:
                messages.error(request, str(problem))
            else:
                TrainingRun.record(note["name"], result, answers["test_percent"] / 100,
                                   answers["seed"])
    else:
        form = TrainingForm(dataset.kind)

    if result is not None:
        curve_url = plots.validation_curve(result, token)
        if dataset.kind == "regression":
            fit_url = plots.predicted_against_actual(result.truth, result.predicted,
                                                     token, dataset.target)

    return render(request, "project1/train.html", {
        "page": "train",
        "stamp": int(time.time()),
        "dataset": dataset,
        "name": note["name"],
        "form": form,
        "result": result,
        "leaderboard": leaderboard,
        "automated": automated,
        "curve_url": curve_url,
        "fit_url": fit_url,
        "hints": {a.key: {"param": a.param, "help": a.param_help,
                          "grid": ", ".join(f"{v:g}" for v in a.grid),
                          "about": a.about, "param_about": a.param_about,
                          "about_short": TOPICS[a.about].short,
                          "param_short": TOPICS[a.param_about].short}
                  for a in learning.algorithms_for(dataset.kind)},
        "score_hints": {s.key: {"about": s.about, "short": TOPICS[s.about].short}
                        for s in learning.scores_for(dataset.kind)},
    })


def retype(request):
    """Change how the target is interpreted, without re-uploading anything.

    The file is already on disk and the problem type is only a note in the
    session, so asking the user to upload the same file again to change one
    dropdown would be ceremony. If the new reading is impossible, nothing
    changes and the dataset they had stays loaded.
    """
    dataset, note = _remembered(request)
    if dataset is None:
        return redirect("project1:index")

    wanted = request.POST.get("kind") or None
    try:
        switched = data.load_csv(_storage_path(note["file"]), kind=wanted)
    except data.DatasetError as problem:
        messages.error(request, str(problem))
        return redirect("project1:visualize")

    note["kind"] = wanted
    request.session[SESSION_KEY] = note
    messages.success(
        request,
        f"Now reading '{switched.target}' as {switched.kind}"
        + ("." if wanted else " (detected automatically)."))
    return redirect("project1:visualize")


def history(request):
    dataset, note = _remembered(request)
    return render(request, "project1/history.html", {
        "page": "history",
        "runs": TrainingRun.objects.all()[:40],
        "dataset": dataset,
        "name": (note or {}).get("name"),
    })


def explain_index(request):
    return render(request, "project1/explain_index.html",
                  {"page": "explain", "groups": groups(), "count": len(TOPICS)})


def explain(request, slug):
    topic = TOPICS.get(slug)
    if topic is None:
        raise Http404(f"No explanation for '{slug}'.")
    return render(request, "project1/explain.html", {
        "page": "explain",
        "topic": topic,
        "related": [TOPICS[s] for s in topic.related if s in TOPICS],
    })
def reset(request):
    note = request.session.pop(SESSION_KEY, None)
    if note:
        for leftover in ("csv", "scatter.png", "dist.png", "corr.png", "sweep.png", "fit.png"):
            stale = _storage_path(f"{note['token']}-{leftover}" if leftover != "csv" else note["file"])
            if os.path.exists(stale):
                os.remove(stale)
    return redirect("project1:index")
