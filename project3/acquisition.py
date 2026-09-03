"""Active learning (Task 4): which documents to spend expert queries on.

Task 3 assumed every expert answer was already available. Here none of them are,
and each one has to be asked for. The question is which few hundred documents to
ask about, out of 120,000, in order to learn when handing over is worth it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import deferral

# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 3, Task 4
#
#   "Choose an active learning strategy for querying the expert. The goal is to
#    efficiently learn when deferral is beneficial. Justify your choice and
#    report the results using metrics of your choice."
#
# THE CHOICE: query the documents where the DEFERRAL DECISION is closest to
# flipping -- smallest |m_exp(x) - m_clf(x)| -- rather than where the classifier
# is least sure of its answer.
#
# WHY, and why the obvious alternative is wrong here. Ordinary active learning
# asks where the classifier is uncertain, because the classifier is what is
# being learned. That instinct does not transfer. Here the classifier is already
# trained and fixed, and its correctness is known for free on every training row
# from the out-of-fold pass -- m_clf costs zero expert queries. The only thing an
# expert query buys is information about m_exp, and only where it changes the
# ANSWER to the question "who should handle this document".
#
# A document the classifier finds baffling is worth a query only if it is also
# unclear whether the expert would do better. If the expert is plainly out of
# their depth there too, both agree, the document is hopeless, and asking about
# it teaches nothing about where to hand over. Conversely a document the
# classifier is quite sure about can still be worth asking about, if the expert
# is likely to be surer still. Classifier entropy targets the wrong quantity,
# and `entropy_sampling` below is implemented so that this can be shown rather
# than asserted.
#
# THE COLD START, which the boundary rule cannot handle on its own. With no
# expert answers there is no m_exp, so there is no boundary to sit near, and
# worse, an early estimate fitted on a handful of points will confidently point
# the search at the wrong place and keep it there -- it would be choosing where
# to look using the very belief it is trying to correct. So the first batch is
# drawn stratified across the k-means regions instead, which guarantees every
# part of the input space is represented before any targeting begins.
# ══════════════════════════════════════════════════════════════════════════


def stratified_start(clusters, size, rng):
    """A first batch spread evenly over the regions, as near as their sizes allow."""
    labels = np.unique(clusters)
    per_region = max(1, size // len(labels))
    picked = []
    for label in labels:
        members = np.flatnonzero(clusters == label)
        take = min(per_region, len(members))
        picked.extend(rng.choice(members, take, replace=False))
    picked = np.array(picked)

    if len(picked) < size:                       # top up from whatever is left
        rest = np.setdiff1d(np.arange(len(clusters)), picked)
        picked = np.concatenate([picked, rng.choice(rest, size - len(picked), replace=False)])
    return picked[:size]


def boundary_sampling(features, asked, machine_correctness, expert_head, size, rng):
    """Our strategy: closest to flipping the decision."""
    if expert_head is None:                      # nothing fitted yet
        return random_sampling(features, asked, size, rng)
    advantage = expert_head.predict_proba(features)[:, 1] - machine_correctness
    return _take_smallest(np.abs(advantage), asked, size)


def entropy_sampling(features, asked, proba, size, rng):
    """The instinct this problem invites: ask about whatever the classifier finds
    hardest. Present as a baseline precisely because it should lose."""
    entropy = deferral.confidence_features(proba)[:, 2]
    return _take_smallest(-entropy, asked, size)          # largest entropy first


def random_sampling(features, asked, size, rng):
    available = np.flatnonzero(~asked)
    return rng.choice(available, min(size, len(available)), replace=False)


def stratified_sampling(clusters, asked, size, rng):
    available = np.flatnonzero(~asked)
    local = stratified_start(clusters[available], min(size, len(available)), rng)
    return available[local]


def _take_smallest(score, asked, size):
    score = score.copy()
    score[asked] = np.inf
    return np.argsort(score)[:size]


STRATEGIES = {
    "boundary": "Closest to flipping the decision — our choice",
    "entropy": "Whatever the classifier finds hardest — the wrong instinct",
    "random": "Uniformly at random — the honest control",
    "stratified": "Spread evenly across regions — coverage without targeting",
}


def run(strategy, features, clusters, proba, classifier_ok, expert_ok,
        eval_features, eval_classifier_ok, eval_expert_ok,
        budgets, classifier_head, first_batch=120, seed=0, tau=0.0):
    """Query the expert under one strategy, measuring the team as the budget grows.

    Returns one row per budget: what the team scored on the held-out set having
    seen only that many expert answers.
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    asked = np.zeros(n, dtype=bool)

    # m_clf is fixed for the whole experiment -- it costs no expert queries and
    # does not change as the budget grows -- so it is fitted once by the caller
    # rather than 48 times over the strategy and seed grid.
    machine_correctness = classifier_head.predict_proba(features)[:, 1]

    expert_head = None
    history = []

    for budget in budgets:
        wanted = budget - int(asked.sum())
        while wanted > 0:
            if not asked.any():
                batch = stratified_start(clusters, min(first_batch, wanted), rng)
            elif strategy == "boundary":
                batch = boundary_sampling(features, asked, machine_correctness,
                                          expert_head, wanted, rng)
            elif strategy == "entropy":
                batch = entropy_sampling(features, asked, proba, wanted, rng)
            elif strategy == "stratified":
                batch = stratified_sampling(clusters, asked, wanted, rng)
            else:
                batch = random_sampling(features, asked, wanted, rng)

            asked[batch] = True
            wanted = budget - int(asked.sum())

            queried = np.flatnonzero(asked)
            if len(np.unique(expert_ok[queried])) < 2:
                # Every answer so far agrees; a classifier needs both outcomes.
                expert_head = None
            else:
                expert_head = LogisticRegression(max_iter=3000, random_state=seed) \
                    .fit(features[queried], expert_ok[queried])

        if expert_head is None:
            scores = deferral.evaluate(np.zeros(len(eval_features), bool),
                                       eval_classifier_ok, eval_expert_ok)
        else:
            heads = deferral.Heads(classifier_head, expert_head, int(asked.sum()))
            scores = deferral.evaluate(heads.advantage(eval_features) > tau,
                                       eval_classifier_ok, eval_expert_ok)

        scores["budget"] = int(budget)
        scores["strategy"] = strategy
        scores["f1"] = deferral.f1(scores)
        history.append(scores)

    return history
