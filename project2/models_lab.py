"""Fitting a family of models at varying complexity, and picking one from it.

Equation (1) of the brief trades training loss against a complexity measure.
Task 2 turns that trade into something the user holds: a slider on lambda,
selecting the model that maximises

    acc_test - lambda * Omega(f).
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import data

LEAF_BUDGETS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
# Finely spaced at the low end: that is where features drop out one at a time,
# and a coarse grid there would skip whole values of Omega the slider could use.
PENALTIES = [0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.014,
             0.017, 0.020, 0.025, 0.03, 0.04, 0.06, 0.09, 0.15, 0.3, 0.6, 1.5, 5.0]


@dataclass
class Candidate:
    setting: str                 # what was varied, in words
    omega: float                 # the complexity measure
    accuracy: float              # on the shared test split
    train_accuracy: float
    estimator: object = field(repr=False, default=None)
    detail: str = ""             # e.g. which features survived

    def objective(self, lam):
        return self.accuracy - lam * self.omega


def _accuracy(estimator, X, y):
    return float((estimator.predict(X) == y).mean())


@lru_cache(maxsize=1)
def tree_family():
    """Decision trees of increasing size. Omega is the number of leaves, which
    the brief fixes for this model."""
    p = data.load()
    out = []
    for budget in LEAF_BUDGETS:
        tree = DecisionTreeClassifier(max_leaf_nodes=budget, random_state=0)
        tree.fit(p.X[p.train], p.y[p.train])
        leaves = int(tree.get_n_leaves())
        if out and leaves == out[-1].omega:
            continue                      # the budget stopped binding; same tree
        out.append(Candidate(
            setting=f"at most {budget} leaves",
            omega=leaves,
            accuracy=_accuracy(tree, p.X[p.test], p.y[p.test]),
            train_accuracy=_accuracy(tree, p.X[p.train], p.y[p.train]),
            estimator=tree,
            detail=f"{leaves} leaves, depth {tree.get_depth()}"))
    return out


# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 2, Task 3
#
#   "Repeat the same with logistic regression. Choose a suitable complexity
#    measure Omega."
#
# Omega is the number of ORIGINAL features carrying a non-zero coefficient in
# any class, under an L1 penalty.
#
# Why this one. Omega has to measure the same thing the leaf count measures:
# how much of the model a person must read before they can say what it does.
# For a tree that is the number of questions it can ask. For a linear model it
# is the number of features you have to look up before you can evaluate it --
# a coefficient of zero costs the reader nothing. L1 is what makes that count
# move, since an L2 penalty shrinks coefficients without ever setting them to
# zero and would leave Omega pinned at the full feature count.
#
# Original features, not encoded columns: "island" is one thing a person has
# to go and find out, whether it expands into two dummy columns or ten. The
# reader's effort is what is being counted, not the matrix width.
# ══════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def logistic_family():
    p = data.load()
    scaler = StandardScaler().fit(p.X[p.train])
    out = []
    for C in PENALTIES:
        # l1_ratio=1 is plain L1; scikit-learn 1.8 deprecated penalty="l1".
        # saga is stochastic: without a fixed seed the fitted coefficients, and
        # therefore Omega, differ between runs and the page contradicts itself.
        model = LogisticRegression(l1_ratio=1, C=C, solver="saga", max_iter=20000,
                                   random_state=0)
        model.fit(scaler.transform(p.X[p.train]), p.y[p.train])

        used = {source for source, column in zip(p.origin, model.coef_.T)
                if np.abs(column).max() > 1e-8}
        omega = len(used)
        if out and omega == out[-1].omega:
            continue

        wrapped = _Scaled(scaler, model)
        names = ", ".join(data.PRETTY[f] for f in p.features if f in used) or "none"
        out.append(Candidate(
            setting=f"C = {C:g}",
            omega=omega,
            accuracy=_accuracy(wrapped, p.X[p.test], p.y[p.test]),
            train_accuracy=_accuracy(wrapped, p.X[p.train], p.y[p.train]),
            estimator=wrapped,
            detail=f"{omega} of {len(p.features)} features used: {names}"))
    return out


class _Scaled:
    """Keeps the standardiser with the model so callers can pass raw rows."""

    def __init__(self, scaler, model):
        self.scaler, self.model = scaler, model
        self.classes_ = model.classes_

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def scale_(self):
        return self.scaler.scale_


def family(kind):
    return tree_family() if kind == "tree" else logistic_family()


def frontier(candidates):
    """Which candidates can ever be selected, and over which range of lambda.

    For a fixed candidate, acc - lambda*Omega is a straight line in lambda with
    slope -Omega. Maximising over candidates is therefore taking the upper
    envelope of a family of lines, and only the candidates whose line touches
    that envelope are ever chosen. Plotted as (Omega, accuracy), that envelope
    is the upper concave hull: anything strictly below it loses at every
    lambda to something simpler, something more accurate, or a blend of the
    two, so no setting of the slider will ever produce it.

    Knowing this turns the slider from a dial you drag blindly into a short
    list -- there are only so many models it can give you, and the lambda at
    which the answer changes is where two lines cross.
    """
    best = {}
    for c in candidates:
        if c.omega not in best or c.accuracy > best[c.omega].accuracy:
            best[c.omega] = c

    rising = []
    for c in sorted(best.values(), key=lambda c: c.omega):
        if not rising or c.accuracy > rising[-1].accuracy:
            rising.append(c)          # more complex but no better: never worth it

    def slope(a, b):
        return (b.accuracy - a.accuracy) / (b.omega - a.omega)

    hull = []
    for c in rising:
        while len(hull) >= 2 and slope(hull[-2], hull[-1]) <= slope(hull[-1], c):
            hull.pop()                # the middle point sits below the chord
        hull.append(c)

    bands = []
    for i, c in enumerate(hull):
        upper = slope(hull[i - 1], c) if i > 0 else None      # None means "and above"
        lower = slope(c, hull[i + 1]) if i + 1 < len(hull) else 0.0
        bands.append({"candidate": c, "low": lower, "high": upper})
    bands.reverse()                   # most accurate first, i.e. lambda ascending
    return hull, bands


def choose(candidates, lam):
    """The model the slider currently points at."""
    return max(candidates, key=lambda c: (c.objective(lam), -c.omega))
