"""Feature effect curves: what does one measurement actually do to the model?

Both methods answer the same question and disagree in an informative way. The
PDP asks what happens if every penguin in the data had this bill length. The
ALE asks what happens to penguins that actually have roughly this bill length.
When the features are correlated -- and here flipper length and body mass very
much are -- the first question involves penguins that could not exist.
"""

import numpy as np

from . import data

# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 2, Task 5
#
#   "The code for the computation of the PDP and ALE values should be written
#    by you, i.e., do not use a library for them."
#
# Everything below is computed from the fitted model's own predicted
# probabilities and numpy arithmetic. There is no sklearn.inspection, no
# partial_dependence, no PartialDependenceDisplay, no ALE package: the imports
# at the top of this file are numpy and this project's own data module, and
# that is the whole of it.
# ══════════════════════════════════════════════════════════════════════════


def _grid(values, points):
    """Evaluation points spread by quantile, so they follow the data rather
    than the axis: a long empty tail gets few points, a dense region many."""
    qs = np.linspace(0, 1, points)
    grid = np.unique(np.quantile(values, qs))
    return grid


def _with_feature_set_to(penguins, column, value):
    """A copy of the whole encoded dataset with one column overwritten."""
    X = penguins.X.copy()
    X[:, column] = value
    return X


def partial_dependence(model, penguins, feature, points=40):
    """PDP: average the model's probability over the data, having forced the
    feature to each value in turn.

        PDP_c(v) = (1/n) * sum_i  P(class c | x_i with feature := v)

    The average is over every row, which is exactly why it can be misleading:
    setting flipper length to 230 mm on an Adelie produces a penguin that does
    not exist, and the model is asked about it anyway.
    """
    column = penguins.columns.index(feature)
    values = penguins.X[:, column]
    grid = _grid(values, points)

    curves = np.zeros((len(grid), len(model.classes_)))
    for i, v in enumerate(grid):
        probabilities = model.predict_proba(_with_feature_set_to(penguins, column, v))
        curves[i] = probabilities.mean(axis=0)
    return grid, curves


def _bins(values, count):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, count + 1)))
    return edges


def accumulated_local_effects(model, penguins, feature, bins=20, exact=False,
                              substeps=8):
    """ALE, following the lecture's definition

        ALE_c(v) = integral over z up to v of  E[ d P(c) / dz ]  dz  -  C,

    with C chosen so the curve averages to zero over the data.

    The expectation is conditional: inside each bin only the rows whose own
    value of the feature falls in that bin contribute. That is the whole point
    of ALE over the PDP -- the model is never asked about a combination that
    does not occur.

    Two ways of getting the integrand, which is the question the brief asks:

    `exact=False`  -- the derivative is replaced by a difference across the bin
        edges. Required for a decision tree: a tree is piecewise constant, so
        its derivative is zero almost everywhere and undefined on the split
        points. There is no derivative to evaluate, and differencing across an
        interval is the only thing that recovers the steps.

    `exact=True`   -- the derivative is written down in closed form and
        integrated over the bin numerically. Available for logistic regression,
        which is smooth. See `_softmax_gradient` for the expression.
    """
    column = penguins.columns.index(feature)
    values = penguins.X[:, column]
    edges = _bins(values, bins)
    n_classes = len(model.classes_)

    local = np.zeros((len(edges) - 1, n_classes))
    counts = np.zeros(len(edges) - 1)

    for k in range(len(edges) - 1):
        low, high = edges[k], edges[k + 1]
        inside = (values > low) & (values <= high) if k else (values >= low) & (values <= high)
        counts[k] = inside.sum()
        if counts[k] == 0:
            continue

        rows = penguins.X[inside]
        if exact:
            # Integrate the closed-form derivative across the bin.
            ts = np.linspace(low, high, substeps)
            slopes = np.zeros((len(ts), n_classes))
            for j, t in enumerate(ts):
                probe = rows.copy()
                probe[:, column] = t
                slopes[j] = _softmax_gradient(model, probe, column).mean(axis=0)
            local[k] = np.trapezoid(slopes, ts, axis=0)
        else:
            upper, lower = rows.copy(), rows.copy()
            upper[:, column] = high
            lower[:, column] = low
            local[k] = (model.predict_proba(upper)
                        - model.predict_proba(lower)).mean(axis=0)

    accumulated = np.vstack([np.zeros(n_classes), np.cumsum(local, axis=0)])

    # Centre: weight each edge by how much data sits around it, so the curve
    # averages to zero over the dataset rather than over the axis.
    weight = np.zeros(len(edges))
    weight[:-1] += counts / 2
    weight[1:] += counts / 2
    weight /= weight.sum()
    accumulated -= (accumulated * weight[:, None]).sum(axis=0)
    return edges, accumulated


def _softmax_gradient(model, X, column):
    """d P(class c) / d x_j in closed form, for the logistic model.

    The scores are linear in the standardised inputs, s = W z + b with
    z = (x - mean) / scale, so d s_c / d x_j = W[c, j] / scale[j]. Pushing that
    through the softmax gives

        d p_c / d x_j = p_c * ( ds_c - sum_c' p_c' ds_c' ),

    i.e. a class gains probability only to the extent its own score rises
    faster than the average rise, weighted by how likely the classes are. This
    is why an ALE curve for one class must fall when the others rise.
    """
    p = model.predict_proba(X)
    ds = model.coef_[:, column] / model.scale_[column]        # one per class
    weighted = p @ ds
    return p * (ds[None, :] - weighted[:, None])


def curves_for(model, penguins, feature, kind, bins=20):
    """Both curves for one feature, plus which ALE route was taken and why."""
    grid, pdp = partial_dependence(model, penguins, feature)
    smooth = kind == "logistic"
    edges, ale = accumulated_local_effects(model, penguins, feature,
                                           bins=bins, exact=smooth)
    return {
        "feature": feature,
        "pretty": data.PRETTY[feature],
        "classes": list(model.classes_),
        "pdp_grid": grid, "pdp": pdp,
        "ale_edges": edges, "ale": ale,
        "exact": smooth,
        "derivative": ("closed form \u2014 logistic regression is smooth, so the softmax "
                       "derivative is written down and integrated over each bin"
                       if smooth else
                       "finite differences \u2014 a decision tree is piecewise constant, so "
                       "its derivative is zero almost everywhere and undefined at the "
                       "splits; there is nothing to evaluate pointwise"),
    }
