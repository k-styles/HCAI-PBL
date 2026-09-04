"""The protocol: what a participant is shown, in what order, and when it ends.

Kept apart from the views so that the design decisions in the report have one
obvious home in the code, and so the numbers below can be read without wading
through request handling.
"""

import numpy as np

from . import data, preference
from .models import Participant, Response

# Both arms get the same wall-clock budget rather than the same number of
# tasks. Comparing "40 pairs against 8 rankings" would just be comparing two
# arbitrary numbers; what a real recommender is buying is the participant's
# attention, and eight minutes of it is the same cost either way. This is the
# single most important control in the design.
ELICIT_SECONDS = 8 * 60
PREVIEW_SECONDS = 75          # so the interface can be walked through quickly

# Enough that quitting immediately still leaves something to fit.
MIN_TASKS = {Participant.PAIR: 4, Participant.RANK: 1}

RANK_SIZE = 10                # Design 2, fixed by the brief
HOLDOUT_PAIRS = 10            # identical in both arms
REPEATS = 2                   # of those pairs, shown twice, for consistency


def budget(participant):
    return PREVIEW_SECONDS if participant.preview else ELICIT_SECONDS


def remaining(participant):
    return max(0.0, budget(participant) - participant.elicitation_seconds)


def _rng(participant, stream):
    # Deriving every draw from the participant's stored seed means the held-out
    # block is fixed for them: reloading the page cannot reroll it into an
    # easier one, and the whole session is reproducible from one integer.
    return np.random.default_rng([participant.seed, stream])


def holdout_block(participant):
    """The test set. Same protocol for both arms, which is what makes the
    primary comparison meaningful, and disjoint from nothing -- films may recur
    from elicitation, but these particular pairs were never asked before."""
    rng = _rng(participant, 1)
    pairs = [data.sample(rng, 2) for _ in range(HOLDOUT_PAIRS)]
    block = [{"pair": p, "repeat_of": None} for p in pairs]
    for slot in rng.choice(HOLDOUT_PAIRS, size=REPEATS, replace=False):
        # Presented again, sides swapped so it is not visually identical.
        block.append({"pair": list(reversed(pairs[int(slot)])),
                      "repeat_of": int(slot)})
    tail = block[HOLDOUT_PAIRS:]
    rng.shuffle(tail)
    return block[:HOLDOUT_PAIRS] + tail


def next_task(participant):
    """Where the participant is. Returns None when the study is over."""
    done = participant.elicitations.count()
    if remaining(participant) > 0 or done < MIN_TASKS[participant.arm]:
        rng = _rng(participant, 100 + done)
        size = 2 if participant.arm == Participant.PAIR else RANK_SIZE
        return {"phase": Response.ELICIT, "position": done,
                "items": data.sample(rng, size),
                "seconds_left": remaining(participant)}

    block = holdout_block(participant)
    position = participant.holdouts.count()
    if position < len(block):
        step = block[position]
        return {"phase": Response.HOLDOUT, "position": position,
                "items": step["pair"], "repeat_of": step["repeat_of"],
                "total": len(block)}
    return None


def record(participant, phase, position, order, seconds, repeat_of=None):
    Response.objects.update_or_create(
        participant=participant, phase=phase, position=position,
        defaults={"order": [int(i) for i in order],
                  "seconds": float(seconds), "repeat_of": repeat_of})
    if phase == Response.ELICIT:
        participant.elicitation_seconds += float(seconds)
        participant.save(update_fields=["elicitation_seconds"])


def analyse(participant):
    """Fit w on this participant's elicitation responses and score it on their
    held-out pairs -- the per-participant version of the study's outcome."""
    X = data.pool()["X"]
    elicited = [r.order for r in participant.elicitations]
    w, info = preference.fit(X, elicited)

    holdouts = list(participant.holdouts)
    comparisons = [(r.order[0], r.order[1]) for r in holdouts if r.repeat_of is None]
    score = preference.score_holdout(w, X, comparisons)

    # Test-retest: did the repeated pairs get the same answer? Two flips is the
    # pre-registered exclusion criterion, so it is computed here rather than
    # left to whoever analyses the data later.
    first = {r.position: r.order[0] for r in holdouts if r.repeat_of is None}
    agree = [first.get(r.repeat_of) == r.order[0]
             for r in holdouts if r.repeat_of is not None]

    return {"w": w, "fit": info, "score": score,
            "tasks": len(elicited),
            "seconds": participant.elicitation_seconds,
            "consistency": agree,
            "consistent": all(agree) if agree else None,
            "excluded": bool(agree) and not any(agree)}


def profile(w, top=6):
    """w as something a person can read: the features pushing hardest either
    way. Shown on the debrief because a participant who has just spent eight
    minutes is owed an answer, not a thank-you page."""
    pool = data.pool()
    order = np.argsort(-np.abs(w))
    scale = float(np.abs(w).max()) or 1.0
    rows = []
    for j in order[:top]:
        rows.append({"name": pool["labels"][pool["names"][j]],
                     "weight": float(w[j]),
                     "width": abs(float(w[j])) / scale * 100,
                     "likes": bool(w[j] > 0)})
    return rows


def recommend(w, count=5):
    """The point of the exercise: what would we now put in front of this user."""
    pool = data.pool()
    utility = pool["X"] @ w
    return data.cards(np.argsort(-utility)[:count])
