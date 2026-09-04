import secrets
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.clickjacking import xframe_options_sameorigin

from home.own_work import for_project

from . import data, preference, study
from .explain import TOPICS, groups
from .models import Participant, Response

REPORT = Path(settings.BASE_DIR) / "project4" / "report" / "project4-report.pdf"
SESSION_KEY = "project4_participant"


def _participant(request):
    token = request.session.get(SESSION_KEY)
    if not token:
        return None
    return Participant.objects.filter(token=token).first()


def _shell(request, page, **extra):
    return {"page": page,
            "has_report": REPORT.exists(),
            "page_requirements": extra.pop("requirements", []),
            **extra}


def index(request):
    """The landing page the brief asks for: the PDF on one side, the study on
    the other."""
    pool = data.pool()
    live = _participant(request)
    return render(request, "project4/index.html", _shell(
        request, "landing",
        requirements=for_project(4),
        films=len(pool["films"]),
        dims=pool["X"].shape[1],
        genres=pool["genres"],
        dropped=pool["dropped"],
        threshold=pool["threshold"],
        rank_size=study.RANK_SIZE,
        minutes=study.ELICIT_SECONDS // 60,
        holdout=study.HOLDOUT_PAIRS,
        arms=Participant.ARMS,
        completed=Participant.objects.filter(finished__isnull=False).count(),
        live=live))


def features(request):
    """Task 1 in the open: every dimension, why it is there, what was left out."""
    pool = data.pool()
    X = pool["X"]
    rows = []
    for j, name in enumerate(pool["names"]):
        column = X[:, j]
        rows.append({"name": name, "label": pool["labels"][name],
                     "genre": name in pool["genres"],
                     "prevalence": float(pool["prevalence"].get(name, float("nan"))),
                     "informative": float(2 * pool["prevalence"].get(name, 0)
                                          * (1 - pool["prevalence"].get(name, 0)))
                                    if name in pool["genres"] else None,
                     "mean": float(column.mean()), "sd": float(column.std())})
    dropped = [{"name": g, "prevalence": float(pool["prevalence"][g]),
                "informative": float(2 * pool["prevalence"][g] * (1 - pool["prevalence"][g]))}
               for g in pool["dropped"]]
    return render(request, "project4/features.html", _shell(
        request, "features",
        requirements=[r for r in for_project(4) if r.task == "Task 1"],
        rows=rows, dropped=dropped, threshold=pool["threshold"],
        films=len(pool["films"]), dims=X.shape[1],
        scaling=pool["scaling"], min_votes=data.MIN_VOTES,
        sample=data.cards(range(3))))


def start(request):
    """Consent, one stratification question, then random assignment."""
    if request.method != "POST":
        return render(request, "project4/consent.html", _shell(
            request, "consent",
            minutes=study.ELICIT_SECONDS // 60,
            rank_size=study.RANK_SIZE, holdout=study.HOLDOUT_PAIRS,
            frequencies=Participant.FREQUENCY))

    if not request.POST.get("consent"):
        return render(request, "project4/consent.html", _shell(
            request, "consent", error="The study cannot begin without consent.",
            minutes=study.ELICIT_SECONDS // 60,
            rank_size=study.RANK_SIZE, holdout=study.HOLDOUT_PAIRS,
            frequencies=Participant.FREQUENCY))

    frequency = request.POST.get("frequency", "")
    forced = request.POST.get("arm")
    if forced not in (Participant.PAIR, Participant.RANK):
        # Simple randomisation. The report explains why this is blocked by
        # frequency in the real study and why that cannot be done here without
        # a running allocation table.
        forced = secrets.choice([Participant.PAIR, Participant.RANK])

    participant = Participant.objects.create(
        token=secrets.token_hex(6),
        arm=forced,
        frequency=frequency if frequency in dict(Participant.FREQUENCY) else "",
        seed=secrets.randbelow(2 ** 31),
        preview=bool(request.POST.get("preview")))
    request.session[SESSION_KEY] = participant.token
    request.session["project4_shown"] = time.time()
    return redirect("project4:task")


def task(request):
    participant = _participant(request)
    if participant is None:
        return redirect("project4:index")

    if request.method == "POST":
        # Design 1 posts which film was clicked plus the pair it came from;
        # Design 2 posts the whole list already in order. Both become the same
        # thing -- a ranking, best first -- which is the point of Task 2.
        chosen = request.POST.get("chosen")
        if chosen is not None:
            items = [int(v) for v in request.POST.getlist("items")]
            order = [int(chosen)] + [i for i in items if i != int(chosen)]
        else:
            order = [int(v) for v in request.POST.getlist("order") if v.strip()]
        phase = request.POST.get("phase")
        position = int(request.POST.get("position", 0))
        shown = request.session.get("project4_shown") or time.time()
        elapsed = max(0.0, min(time.time() - shown, 600.0))

        expected = 2 if (phase == Response.HOLDOUT
                         or participant.arm == Participant.PAIR) else study.RANK_SIZE
        if len(order) == expected and len(set(order)) == expected:
            repeat = request.POST.get("repeat_of")
            study.record(participant, phase, position, order, elapsed,
                         int(repeat) if repeat not in (None, "", "None") else None)
            if request.POST.get("stop"):
                # Ending elicitation early is a right, not a bug; spend the
                # remaining budget so the state machine moves on.
                participant.elicitation_seconds = study.budget(participant)
                participant.save(update_fields=["elicitation_seconds"])
        request.session["project4_shown"] = time.time()
        return redirect("project4:task")

    step = study.next_task(participant)
    if step is None:
        if participant.finished is None:
            participant.finished = timezone.now()
            participant.save(update_fields=["finished"])
        return redirect("project4:debrief")

    request.session["project4_shown"] = time.time()
    context = _shell(request, "study",
                     participant=participant, step=step,
                     cards=data.cards(step["items"]),
                     holdout=step["phase"] == Response.HOLDOUT,
                     total=step.get("total"),
                     done=participant.elicitations.count(),
                     budget_seconds=study.budget(participant),
                     seconds_left=int(step.get("seconds_left") or 0))
    if step["phase"] == Response.HOLDOUT or participant.arm == Participant.PAIR:
        return render(request, "project4/pair.html", context)
    return render(request, "project4/rank.html", context)


def debrief(request):
    participant = _participant(request)
    if participant is None:
        return redirect("project4:index")
    result = study.analyse(participant)
    if request.GET.get("clear"):
        request.session.pop(SESSION_KEY, None)
        return redirect("project4:index")
    return render(request, "project4/debrief.html", _shell(
        request, "debrief",
        participant=participant, result=result,
        profile=study.profile(result["w"]),
        picks=study.recommend(result["w"]),
        baseline=preference.score_holdout(
            np.zeros(data.pool()["X"].shape[1]), data.pool()["X"],
            [(r.order[0], r.order[1]) for r in participant.holdouts
             if r.repeat_of is None])))


def responses(request):
    """What the study has collected so far. Not part of the participant's
    journey -- it is the view the person running the study needs, and the one
    that shows the data is really being written down."""
    rows = []
    for p in Participant.objects.all()[:60]:
        result = study.analyse(p) if p.responses.exists() else None
        rows.append({"p": p, "result": result})
    return render(request, "project4/responses.html", _shell(
        request, "responses", rows=rows,
        arms=dict(Participant.ARMS),
        total=Participant.objects.count()))


def explain_index(request):
    return render(request, "project4/explain_index.html",
                  _shell(request, "explain", groups=groups(), count=len(TOPICS)))


def explain(request, slug):
    topic = TOPICS.get(slug)
    if topic is None:
        raise Http404(f"No explanation for '{slug}'.")
    return render(request, "project4/explain.html", _shell(
        request, "explain", topic=topic,
        related=[TOPICS[s] for s in topic.related if s in TOPICS]))


@xframe_options_sameorigin
def report(request):
    """The brief: "I can download a PDF providing explanations about the
    implemented method (Tasks 1 and 2) and the design of the user study"."""
    if not REPORT.exists():
        raise Http404("The project 4 report has not been built yet.")
    return FileResponse(open(REPORT, "rb"), content_type="application/pdf",
                        filename=REPORT.name)
