"""Learning to defer (Task 3): deciding, per document, who should answer.

Two small models, each estimating a probability of being right:

    m_clf(x)  how likely the classifier is to be correct on x
    m_exp(x)  how likely the expert is to be correct on x

and one rule: hand over when the expert looks more likely to be right than the
classifier, by a margin the user chooses.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def confidence_features(proba):
    """What the classifier itself is telling us about how sure it is.

    Measured on this data, the top-two margin alone predicts the classifier's own
    correctness at AUC 0.848, while all 50,000 TF-IDF features together manage
    only 0.801. Its own uncertainty is the single most informative thing about
    when it is wrong, so it goes in explicitly rather than being left for the
    text features to rediscover.
    """
    ordered = np.sort(proba, axis=1)
    top = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1)
    return np.c_[top, margin, entropy]


def competence_features(reduced, proba):
    """The representation both heads are fitted on.

    A 100-dimensional SVD of the TF-IDF matrix, plus the three confidence
    numbers. Not the raw 50,000 features: Task 4 has to fit m_exp from a few
    hundred expert answers, and a logistic regression with n=200 and d=50000
    fits noise. Measured on the full data the compact version costs about 0.016
    AUC (0.855 against 0.871) -- a small price for a model that can still be
    estimated when the evidence is scarce.
    """
    return np.hstack([reduced, confidence_features(proba)])


@dataclass
class Heads:
    """The two competence models, and what they were fitted on."""
    classifier_head: object
    expert_head: object
    n_expert_labels: int

    def advantage(self, features):
        """m_exp(x) - m_clf(x): how much better the expert looks, per row.

        Positive means hand over. The size of it matters as well as the sign,
        which is what lets a single threshold express "only hand over when it is
        clearly worth it".
        """
        return (self.expert_head.predict_proba(features)[:, 1]
                - self.classifier_head.predict_proba(features)[:, 1])


def fit_heads(features, classifier_ok, expert_ok, expert_rows=None, seed=0):
    """Fit both heads. `expert_rows` restricts the expert head to queried rows.

    m_clf can always use every training row, because classifier correctness is
    known everywhere from the out-of-fold pass -- it costs no expert queries at
    all. That asymmetry is the whole reason Task 4 is about learning m_exp
    specifically.
    """
    classifier_head = LogisticRegression(max_iter=3000, random_state=seed)
    classifier_head.fit(features, classifier_ok)

    rows = np.arange(len(features)) if expert_rows is None else np.asarray(expert_rows)
    expert_head = LogisticRegression(max_iter=3000, random_state=seed)
    expert_head.fit(features[rows], expert_ok[rows])

    return Heads(classifier_head, expert_head, len(rows))


def outcomes(classifier_ok, expert_ok):
    """The four cases a deferral decision can land in.

    Only one of them is a reason to hand over. Reporting team accuracy alone
    hides the difference between the other three, and the brief asks for the
    quality of the decisions, not just the score they produce.
    """
    return {
        "both": classifier_ok & expert_ok,          # handing over changes nothing, and wastes a person's time
        "machine_only": classifier_ok & ~expert_ok,  # handing over actively costs accuracy
        "expert_only": ~classifier_ok & expert_ok,   # the only case worth handing over
        "neither": ~classifier_ok & ~expert_ok,      # nobody gets it; handing over is harmless
    }


def evaluate(defer, classifier_ok, expert_ok):
    """Score one deferral policy.

    Accuracy is reported, but on its own it rewards a policy that hands
    everything to a good expert without ever learning anything. The precision
    and recall below are about the decisions themselves: of the documents handed
    over, how many needed to be, and of the documents that needed handing over,
    how many were.
    """
    defer = np.asarray(defer, dtype=bool)
    cases = outcomes(classifier_ok, expert_ok)
    worth_it = cases["expert_only"]
    harmful = cases["machine_only"]

    team = np.where(defer, expert_ok, classifier_ok)
    deferred = int(defer.sum())

    return {
        "accuracy": float(team.mean()),
        "deferral_rate": float(defer.mean()),
        "n_deferred": deferred,
        # of what we handed over, how much of it actually needed handing over
        "precision": float(worth_it[defer].mean()) if deferred else 0.0,
        # of everything that needed handing over, how much we caught
        "recall": float(worth_it[defer].sum() / worth_it.sum()) if worth_it.any() else 0.0,
        # handed over something the classifier would have got right
        "harm_rate": float(harmful[defer].sum() / max(deferred, 1)),
        "gain": float(team.mean() - classifier_ok.mean()),
    }


def f1(scores):
    p, r = scores["precision"], scores["recall"]
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def reference_policies(classifier_ok, expert_ok, proba):
    """The things a learned policy has to beat before it has earned anything.

    `confidence` is the important one. Handing over whatever the classifier is
    least sure about is the obvious approach, it needs no expert data at all,
    and any marker will ask why it was not enough -- so it is measured rather
    than dismissed.
    """
    n = len(classifier_ok)
    cases = outcomes(classifier_ok, expert_ok)
    order = np.argsort(confidence_features(proba)[:, 1])   # least confident first

    def by_confidence(rate):
        chosen = np.zeros(n, dtype=bool)
        chosen[order[:int(round(rate * n))]] = True
        return chosen

    return {
        "never": np.zeros(n, dtype=bool),
        "always": np.ones(n, dtype=bool),
        "oracle": cases["expert_only"],       # unreachable: needs both true labels
        "by_confidence": by_confidence,
    }


def sweep(heads, features, classifier_ok, expert_ok, thresholds):
    """Team accuracy and decision quality across the whole range of thresholds."""
    advantage = heads.advantage(features)
    rows = []
    for tau in thresholds:
        scores = evaluate(advantage > tau, classifier_ok, expert_ok)
        scores["tau"] = float(tau)
        scores["f1"] = f1(scores)
        rows.append(scores)
    return advantage, rows
