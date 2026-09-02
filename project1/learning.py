"""Model selection: fold construction, the hyperparameter sweep, and the rule
that picks a winner from it."""

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class Algorithm:
    key: str
    label: str
    kind: str
    param: str
    param_help: str
    grid: list
    cast: Callable
    simpler: str            # "low" or "high": which end of the grid is the plainer model
    build: Callable
    about: str = ""            # glossary entry for the model
    param_about: str = ""      # glossary entry for its hyperparameter
    ceiling: Callable = None   # largest usable value given the rows available to fit on


def _scaled(estimator):
    """Distance- and penalty-based methods need comparable feature units;
    petal width in cm would otherwise count for less than sepal length."""
    return make_pipeline(StandardScaler(), estimator)


ALGORITHMS = {a.key: a for a in [
    Algorithm("knn_c", "k-nearest neighbours", "classification",
              "k", "How many neighbours vote. Larger k smooths the decision boundary.",
              [1, 2, 3, 5, 7, 9, 13, 19, 27], int, "high",
              lambda v: _scaled(KNeighborsClassifier(n_neighbors=int(v))),
              about="knn", param_about="hyperparameter-k",
              ceiling=lambda rows: rows),

    Algorithm("tree_c", "Decision tree", "classification",
              "max depth", "How many questions deep the tree may go. Deeper trees fit more detail.",
              [1, 2, 3, 4, 5, 6, 8, 10, 14], int, "low",
              lambda v: DecisionTreeClassifier(max_depth=int(v), random_state=0),
              about="tree", param_about="hyperparameter-depth"),

    Algorithm("logreg", "Logistic regression", "classification",
              "C", "Inverse penalty strength. Small C pulls the coefficients towards zero.",
              [0.001, 0.01, 0.1, 1, 10, 100, 1000], float, "low",
              lambda v: _scaled(LogisticRegression(C=float(v), max_iter=5000)),
              about="logistic-regression", param_about="hyperparameter-c"),

    Algorithm("svm", "SVM (RBF kernel)", "classification",
              "C", "How much the margin may be violated. Large C fits the training points harder.",
              [0.01, 0.1, 1, 10, 100, 1000], float, "low",
              lambda v: _scaled(SVC(C=float(v), kernel="rbf")),
              about="svm", param_about="hyperparameter-c"),

    Algorithm("ridge", "Ridge regression", "regression",
              "alpha", "Penalty on the coefficient norm. This is the lambda from lecture 1.",
              [0.001, 0.01, 0.1, 1, 10, 100, 1000], float, "high",
              lambda v: _scaled(Ridge(alpha=float(v))),
              about="ridge", param_about="hyperparameter-alpha"),

    Algorithm("knn_r", "k-nearest neighbours", "regression",
              "k", "How many neighbours are averaged. Larger k gives a flatter fit.",
              [1, 2, 3, 5, 7, 9, 13, 19, 27], int, "high",
              lambda v: _scaled(KNeighborsRegressor(n_neighbors=int(v))),
              about="knn", param_about="hyperparameter-k",
              ceiling=lambda rows: rows),

    Algorithm("tree_r", "Decision tree", "regression",
              "max depth", "How many questions deep the tree may go.",
              [1, 2, 3, 4, 5, 6, 8, 10, 14], int, "low",
              lambda v: DecisionTreeRegressor(max_depth=int(v), random_state=0),
              about="tree", param_about="hyperparameter-depth"),
]}


@dataclass
class Score:
    key: str
    label: str
    kind: str
    higher_is_better: bool
    fn: Callable
    about: str = ""

    def __call__(self, truth, prediction):
        return float(self.fn(truth, prediction))


SCORES = {s.key: s for s in [
    Score("accuracy", "Accuracy", "classification", True, accuracy_score, "accuracy"),
    Score("f1_macro", "Macro F1", "classification", True,
          lambda t, p: f1_score(t, p, average="macro", zero_division=0), "macro-f1"),
    Score("r2", "R squared", "regression", True, r2_score, "r-squared"),
    Score("mse", "Mean squared error", "regression", False, mean_squared_error, "mse"),
    Score("mae", "Mean absolute error", "regression", False, mean_absolute_error, "mae"),
]}


def algorithms_for(kind):
    return [a for a in ALGORITHMS.values() if a.kind == kind]


def scores_for(kind):
    return [s for s in SCORES.values() if s.kind == kind]


def hold_out(y, test_fraction, rng, stratify):
    """Split row positions into a training and a test part.

    When stratifying, the share is taken class by class and every class is
    forced to contribute at least one test row, so a rare class cannot vanish
    from the evaluation entirely.
    """
    idx = np.arange(len(y))
    if not stratify:
        shuffled = rng.permutation(idx)
        cut = int(round(len(idx) * test_fraction))
        cut = min(max(cut, 1), len(idx) - 1)
        return shuffled[cut:], shuffled[:cut]

    test = []
    for label in np.unique(y):
        members = rng.permutation(idx[y == label])
        cut = min(max(1, int(round(len(members) * test_fraction))), len(members) - 1)
        test.extend(members[:cut])
    test = np.array(sorted(test))
    return np.setdiff1d(idx, test), test


def make_folds(y, n_folds, rng, stratify):
    """Deal shuffled row positions round-robin into n_folds groups.

    Dealing one class at a time keeps each fold close to the class proportions
    of the whole training set.  With iris and five folds, a plain shuffle can
    hand one fold twice as many of a species as another, and then the
    validation curve is partly measuring the shuffle rather than the model.
    """
    idx = np.arange(len(y))
    groups = [idx[y == label] for label in np.unique(y)] if stratify else [idx]

    folds = [[] for _ in range(n_folds)]
    for group in groups:
        for offset, position in enumerate(rng.permutation(group)):
            folds[offset % n_folds].append(position)
    return [np.array(sorted(f)) for f in folds]


@dataclass
class SweepRow:
    value: float
    mean: float
    se: float
    train_mean: float
    fold_scores: list
    errors: dict = field(default_factory=dict)
    chosen: bool = False
    best_mean: bool = False
    within_one_se: bool = False


def error_breakdown(truth, predicted, labels):
    """What the mistakes actually were, not just how many.

    For classes this is the confusion table: rows are the truth, columns are the
    guess. For numbers there is no such table, so the equivalent question --
    where is it wrong and by how much -- is answered with the spread of the
    misses and the direction of the bias.
    """
    if labels is not None:
        counts = confusion_matrix(truth, predicted, labels=labels)
        return {
            "kind": "classes",
            "labels": [str(l) for l in labels],
            "rows": [
                {"label": str(label), "total": int(row.sum()),
                 "correct": int(row[i]), "missed": int(row.sum() - row[i]),
                 "cells": [{"n": int(n), "diagonal": i == j} for j, n in enumerate(row)]}
                for i, (label, row) in enumerate(zip(labels, counts))],
        }

    residual = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    off_by = np.abs(residual)
    return {
        "kind": "numbers",
        "typical": float(np.median(off_by)),
        "worst": float(off_by.max()),
        "within_half": float(np.mean(off_by <= np.median(off_by)) * 100),
        "over": int((residual > 0).sum()),
        "under": int((residual < 0).sum()),
        "bias": float(residual.mean()),
    }


def sweep(algo, score, X, y, folds, labels=None):
    """Cross-validate every candidate value of the hyperparameter.

    Three things come out of each fit and all three are kept. The held-out score
    decides the winner. The score on the fold's own training rows is what makes
    overfitting visible. And the predictions themselves are collected across the
    folds, so every candidate -- not only the one that wins -- can be asked what
    kind of mistakes it made. Those predictions are free: the models have
    already been fitted, and no fold ever scores itself.
    """
    rows = []
    for value in algo.grid:
        held, fitted = [], []
        out_of_fold_truth, out_of_fold_guess = [], []

        for f, validation in enumerate(folds):
            inner = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
            model = algo.build(value).fit(X[inner], y[inner])
            guess = model.predict(X[validation])

            held.append(score(y[validation], guess))
            fitted.append(score(y[inner], model.predict(X[inner])))
            out_of_fold_truth.append(y[validation])
            out_of_fold_guess.append(guess)

        spread = np.std(held, ddof=1) / np.sqrt(len(held)) if len(held) > 1 else 0.0
        rows.append(SweepRow(
            value=algo.cast(value), mean=float(np.mean(held)), se=float(spread),
            train_mean=float(np.mean(fitted)), fold_scores=[round(s, 4) for s in held],
            errors=error_breakdown(np.concatenate(out_of_fold_truth),
                                   np.concatenate(out_of_fold_guess), labels)))
    return rows


def select(rows, score, algo, rule):
    """Pick one value out of the sweep.

    "best" takes the top mean.  "one_se" takes the plainest model whose mean is
    still within one standard error of the top: those values are not
    distinguishable from the winner given five noisy numbers, so preferring the
    smaller tree or the stronger penalty among them costs almost nothing on
    this sample and is less likely to be an artefact of one particular shuffle.
    """
    peak = (max if score.higher_is_better else min)(rows, key=lambda r: r.mean)
    peak.best_mean = True

    if score.higher_is_better:
        cutoff = peak.mean - peak.se
        tied = [r for r in rows if r.mean >= cutoff]
    else:
        cutoff = peak.mean + peak.se
        tied = [r for r in rows if r.mean <= cutoff]
    for r in tied:
        r.within_one_se = True

    if rule == "one_se":
        order = (lambda r: r.value) if algo.simpler == "low" else (lambda r: -r.value)
        winner = min(tied, key=order)
    else:
        winner = peak
    winner.chosen = True
    return winner, peak


def parse_grid(algo, text):
    """Read the comma-separated hyperparameter values the user typed."""
    if not text or not text.strip():
        return list(algo.grid)

    values = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = algo.cast(float(token))
        except ValueError:
            raise ValueError(f"'{token}' is not a number.")
        if value <= 0:
            raise ValueError(f"{algo.param} must be positive, got {token}.")
        if value not in values:
            values.append(value)

    if len(values) < 2:
        raise ValueError("Give at least two values, otherwise there is nothing to compare.")
    return sorted(values)


@dataclass
class Result:
    algo: Algorithm
    score: Score
    rule: str
    rows: list
    winner: SweepRow
    peak: SweepRow
    test_scores: list
    n_train: int
    n_test: int
    n_folds: int
    truth: object = None
    predicted: object = None
    dropped: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    matrix: list = field(default_factory=list)

    @property
    def headline(self):
        return self.test_scores[0]["value"]

    @property
    def overruled(self):
        return self.winner.value != self.peak.value


def run(dataset, algo, score, test_fraction, n_folds, rule, seed, grid=None):
    """The whole pipeline of lecture 1, in order, with the test set kept back.

    The hyperparameter is chosen by cross-validating inside the training rows.
    The test rows are touched once, at the end.  Selecting on the test set and
    then quoting that same number as the result -- as the lecture demo does --
    reports a score that has already been optimised against.
    """
    algo = ALGORITHMS[algo] if isinstance(algo, str) else algo
    score = SCORES[score] if isinstance(score, str) else score
    if grid:
        algo = replace(algo, grid=grid)

    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy()
    stratify = dataset.kind == "classification"
    rng = np.random.default_rng(seed)

    train, test = hold_out(y, test_fraction, rng, stratify)
    if stratify:
        smallest = min(int((y[train] == label).sum()) for label in np.unique(y[train]))
        if smallest < n_folds:
            raise ValueError(
                f"{n_folds} folds needs {n_folds} training rows per class, but the smallest "
                f"class only has {smallest}. Use fewer folds or a smaller test set.")
    elif len(train) < n_folds:
        raise ValueError(f"{n_folds} folds needs at least {n_folds} training rows.")

    folds = make_folds(y[train], n_folds, rng, stratify)

    # A fold's model only ever sees the other folds, so a candidate that needs
    # more rows than that (k neighbours out of too few points) cannot be tried.
    dropped = []
    if algo.ceiling is not None:
        available = len(train) - max(len(f) for f in folds)
        usable = [v for v in algo.grid if v <= algo.ceiling(available)]
        dropped = [v for v in algo.grid if v not in usable]
        if not usable:
            raise ValueError(
                f"Every value of {algo.param} you asked for needs more than the {available} "
                f"rows a fold can train on. Try smaller values or fewer folds.")
        algo = replace(algo, grid=usable)

    labels = sorted(np.unique(y).tolist(), key=str) if stratify else None
    rows = sweep(algo, score, X[train], y[train], folds, labels=labels)
    winner, peak = select(rows, score, algo, rule)

    model = algo.build(winner.value).fit(X[train], y[train])
    predicted = model.predict(X[test])

    reported = [{"label": s.label, "value": s(y[test], predicted), "primary": s.key == score.key}
                for s in scores_for(dataset.kind)]
    reported.sort(key=lambda s: not s["primary"])

    result = Result(algo=algo, score=score, rule=rule, rows=rows, winner=winner, peak=peak,
                    test_scores=reported, n_train=len(train), n_test=len(test), n_folds=n_folds,
                    truth=y[test], predicted=predicted, dropped=dropped)

    if dataset.kind == "classification":
        counts = confusion_matrix(y[test], predicted, labels=labels)
        result.labels = [str(l) for l in labels]
        result.matrix = [
            {"label": str(label), "total": int(row.sum()),
             "cells": [{"n": int(n), "diagonal": i == j} for j, n in enumerate(row)]}
            for i, (label, row) in enumerate(zip(labels, counts))]
    return result


def automl(dataset, seed=0, test_fraction=0.25, n_folds=5):
    """What a fully automated pipeline does: try every algorithm for this
    problem type, sweep each one's default grid, keep whichever wins.

    The user is asked nothing.  That is the point of the comparison on the
    training page -- every choice lecture 1 attributes to a human is still
    being made here, just not by the human.
    """
    score = scores_for(dataset.kind)[0]
    attempts = []
    for algo in algorithms_for(dataset.kind):
        outcome = run(dataset, algo, score, test_fraction, n_folds, "best", seed)
        attempts.append(outcome)

    pick = (max if score.higher_is_better else min)(attempts, key=lambda r: r.winner.mean)
    leaderboard = [{"label": r.algo.label, "param": r.algo.param, "value": r.winner.value,
                    "cv": r.winner.mean, "test": r.headline, "won": r is pick}
                   for r in attempts]
    leaderboard.sort(key=lambda row: -row["cv"] if score.higher_is_better else row["cv"])
    return pick, leaderboard
