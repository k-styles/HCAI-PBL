"""Simulated experts (Task 2).

A simulated expert is a function from a document to an answer, which is right
some of the time. The whole project rests on how that "some of the time" is
defined, so the reasoning is written out below rather than buried.
"""

from dataclasses import dataclass

import numpy as np

from . import data

# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 3, Task 2
#
#   "Design and implement at least one simulated expert. Analyze its strengths
#    and weaknesses on the dataset and report its accuracy on the test set."
#   and, earlier in the brief: "These experts should not be perfect. Instead,
#   they should exhibit expertise in specific regions of the input space."
#
# THE INVARIANT THAT MATTERS: an expert's competence is a function of the INPUT
# ONLY, never of the true label.
#
# This is not fussiness. Task 4 has to discover the competence profile by
# querying the expert, and at query time the true label is precisely what is
# unknown. An expert defined as "reliable whenever the true topic is Sports"
# cannot be discovered by any honest procedure, because predicting its
# competence would require already knowing the answer -- so Task 4 would be
# vacuous by construction. Defining competence over regions of the input space,
# as the brief words it, keeps the profile learnable from the document alone.
#
# The one place the true label IS used is choosing WHICH wrong answer to give
# when the expert errs. That is a property of the simulation, not of the
# competence: it does not change what Task 4 has to learn, because nothing
# downstream predicts the content of a wrong answer, only whether an answer is
# right. A real human who misreads an article also produces a specific wrong
# answer; the simulation is entitled to know the truth in order to avoid
# accidentally producing the right one.
# ══════════════════════════════════════════════════════════════════════════

CLUSTERS = 12
SPECIALISMS = 5
P_SPECIALIST = 0.95
P_OTHERWISE = 0.40


@dataclass
class ExpertProfile:
    key: str
    label: str
    description: str
    competence: np.ndarray      # P(correct) per row, from the input alone
    detail: dict                # whatever the interface needs to explain it


def desk_expert(clusters_train, clusters_eval, seed=0):
    """Expert A: specialist desks over k-means regions of the TF-IDF space.

    k-means partitions the documents into regions; a fixed, recorded subset of
    those regions is the expert's beat. Inside their beat they are reliable, and
    outside it they are guessing not much better than a coin between four
    options.

    Chosen over a smooth competence function because "regions of the input
    space" is the brief's own phrasing, and because a region is something that
    can be shown to a reader: the interface lists each cluster's top terms
    alongside the expert's accuracy there, which turns Task 2's "analyse its
    strengths and weaknesses" into something concrete rather than a table of
    numbers with no interpretation.
    """
    rng = np.random.default_rng(seed)
    specialist = np.zeros(CLUSTERS, dtype=bool)
    specialist[rng.choice(CLUSTERS, SPECIALISMS, replace=False)] = True

    return {
        "specialist": specialist,
        "train": np.where(specialist[clusters_train], P_SPECIALIST, P_OTHERWISE),
        "eval": np.where(specialist[clusters_eval], P_SPECIALIST, P_OTHERWISE),
    }


def surface_features(texts):
    """Three things about a document that have nothing to do with its topic:
    how long it is, how numeric it is, and how much of it is capitalised."""
    lengths = np.array([len(t) for t in texts], dtype=float)
    digits = np.array([sum(c.isdigit() for c in t) / max(len(t), 1) for t in texts])
    caps = np.array([
        sum(w[:1].isupper() for w in t.split()) / max(len(t.split()), 1) for t in texts])
    return np.c_[lengths, digits, caps]


def headline_expert(features_train, features_eval, reference):
    """Expert B: competence smooth in surface features, deliberately unaligned
    with topic.

    Included as a contrast, not as a better idea. Expert A's competence follows
    regions that correlate with subject matter, so it should be comparatively
    easy to learn from the same TF-IDF features the classifier uses. Expert B's
    does not, so Task 4 should find it harder. Reporting both means the active
    learning results describe a method rather than one lucky competence
    structure.
    """
    mean, scale = reference.mean(axis=0), reference.std(axis=0) + 1e-9
    weights = np.array([0.9, -1.4, 0.7])          # long, non-numeric, name-heavy

    def competence(features):
        z = (features - mean) / scale
        return P_OTHERWISE + (P_SPECIALIST - P_OTHERWISE) / (1 + np.exp(-(z @ weights)))

    return {"train": competence(features_train), "eval": competence(features_eval),
            "weights": weights}


def answer(competence, truth, topics, seed):
    """Sample the expert's answers given its per-row probability of being right.

    A wrong answer is drawn uniformly from the other three topics. Structured
    confusion (World mistaken for Business more often than for Sports) would be
    marginally more realistic and would change nothing that is measured: no part
    of the pipeline predicts which wrong answer appears, only whether an answer
    is right. Simplicity wins.
    """
    rng = np.random.default_rng(seed)
    topics = np.asarray(topics)
    right = rng.random(len(truth)) < competence

    given = np.array(truth, dtype=object).copy()
    wrong_positions = np.flatnonzero(~right)
    for i in wrong_positions:
        options = topics[topics != truth[i]]
        given[i] = options[rng.integers(0, len(options))]
    return given.astype(str), right


def profile_by_region(clusters, expert_ok, classifier_ok, n_clusters=CLUSTERS):
    """The per-region comparison that makes the analysis an analysis.

    Being a specialist is not the same as being worth consulting: the expert can
    be reliable in a region where the classifier is more reliable still, and
    deferring there loses accuracy. Reporting expert accuracy alone would hide
    that, so both are reported side by side along with the difference.
    """
    rows = []
    for k in range(n_clusters):
        mask = clusters == k
        if not mask.any():
            continue
        expert = float(expert_ok[mask].mean())
        machine = float(classifier_ok[mask].mean())
        rows.append({
            "cluster": int(k),
            "n": int(mask.sum()),
            "expert": expert,
            "classifier": machine,
            "advantage": expert - machine,
            "worth_deferring": expert > machine,
        })
    return rows
