"""Counterfactual explanations: what would have had to be different?

The brief's recipe, followed directly -- sample around x, keep the draws the
model assigns to the desired class, rank them by MAD-weighted L1 distance, show
the closest few.
"""

import numpy as np

from . import data

# Each round widens the search: more draws, wider numeric noise, more willing to
# change a categorical. Starting wide would find far-away counterfactuals that a
# tighter search would have beaten, so the first round is deliberately timid.
ROUNDS = [
    {"draws": 2000, "spread": 0.5, "flip": 0.05},
    {"draws": 4000, "spread": 1.0, "flip": 0.15},
    {"draws": 8000, "spread": 2.0, "flip": 0.30},
    {"draws": 16000, "spread": 4.0, "flip": 0.50},
]


def distance(a: dict, b: dict, mads: dict) -> float:
    """MAD-weighted L1 (lecture 3).

    Dividing by each feature's median absolute deviation puts the four numeric
    features on one scale: 200 g of body mass and 2 mm of bill length are then
    comparable amounts of "change", which they are not in raw units.

    A categorical feature has no magnitude -- an island is not 0.4 of another
    island -- so a change counts as 1 and no change as 0. That leaves it in the
    same order as a typical numeric change, which is the intent.
    """
    total = 0.0
    for f in data.NUMERIC:
        scale = mads[f] or 1.0
        total += abs(float(a[f]) - float(b[f])) / scale
    for f in data.CATEGORICAL:
        total += 0.0 if str(a[f]) == str(b[f]) else 1.0
    return total


# ══════════════════════════════════════════════════════════════════════════
# OWN WORK REQUIRED -- Project 2, Section 2
#
#   "Pay attention to the type of data. If it is decimal, you can proceed as
#    above, but what about binary or categorical data? Think of an appropriate
#    way to noise these kinds of features as well."
#
# Numeric features: Gaussian noise with standard deviation spread * MAD(feature).
# Scaling by the feature's own MAD means one `spread` setting means the same
# thing to body mass in grams and bill depth in millimetres, and it is the same
# scale the ranking distance uses, so searching and ranking agree.
#
# Categorical features: there is nothing between "Biscoe" and "Dream", so a
# perturbation has to be a jump, not a nudge. Each categorical is left alone
# with probability 1 - flip, and otherwise redrawn from the values actually
# observed in the data. Two consequences are deliberate:
#
#   - most draws leave the categoricals untouched, so the search spends its
#     effort where small changes are possible, and the counterfactuals that
#     come back tend to be "change your measurements", not "be a different
#     penguin on a different island in a different year";
#   - redrawing from the empirical distribution rather than uniformly keeps
#     rare combinations rare, so we do not rank a proposal highly that no
#     penguin would ever be.
#
# Sampled values are clipped to the range actually observed. A counterfactual
# with a negative body mass is not an answer to "what would I have to change".
#
# One more decision, and the one that matters most in practice. Perturbing all
# four measurements on every draw means every counterfactual that comes back
# has all four changed, and "alter everything slightly" is close to useless as
# advice. So each draw first picks how many numeric features it is allowed to
# touch -- uniformly from one to all of them -- and leaves the rest exactly as
# they were. Sparse proposals then exist to be found, and because changing
# fewer features costs less distance, they win the ranking when they work at
# all. The L1 distance rewards sparsity; without this it never gets the chance.
# ══════════════════════════════════════════════════════════════════════════
def propose(x: dict, penguins, mads, draws, spread, flip, rng):
    """`draws` perturbed copies of x, as a frame-like dict of arrays."""
    out = {}

    # Per draw, how many of the numeric features may move, and which ones.
    budget = rng.integers(1, len(data.NUMERIC) + 1, size=draws)
    order = np.argsort(rng.random((draws, len(data.NUMERIC))), axis=1)
    allowed = order < budget[:, None]

    for j, f in enumerate(data.NUMERIC):
        column = penguins.frame[f].to_numpy(dtype=float)
        noise = rng.normal(0.0, spread * (mads[f] or 1.0), size=draws)
        moved = np.clip(float(x[f]) + noise, column.min(), column.max())
        out[f] = np.where(allowed[:, j], moved, float(x[f]))

    for f in data.CATEGORICAL:
        observed = penguins.frame[f].astype(str).to_numpy()
        redrawn = rng.choice(observed, size=draws)          # empirical distribution
        keep = rng.random(draws) >= flip
        out[f] = np.where(keep, str(x[f]), redrawn)
    return out


def _encode_batch(penguins, batch, draws):
    rows = np.zeros((draws, len(penguins.columns)))
    for i, (column, source) in enumerate(zip(penguins.columns, penguins.origin)):
        if source in data.NUMERIC:
            rows[:, i] = batch[source]
        else:
            level = column.split("=", 1)[1]
            rows[:, i] = (batch[source] == level).astype(float)
    return rows


def search(model, penguins, x: dict, target: str, k=5, seed=0):
    """Closest k points the model calls `target`, plus a note on the effort."""
    mads = data.numeric_mads(penguins)
    rng = np.random.default_rng(seed)
    found, log = [], []

    for attempt, settings in enumerate(ROUNDS, start=1):
        batch = propose(x, penguins, mads, settings["draws"], settings["spread"],
                        settings["flip"], rng)
        predictions = model.predict(_encode_batch(penguins, batch, settings["draws"]))
        hits = np.flatnonzero(predictions == target)
        log.append({"round": attempt, "draws": settings["draws"],
                    "spread": settings["spread"], "flip": settings["flip"],
                    "hits": int(hits.size)})

        for i in hits:
            candidate = {f: batch[f][i] for f in data.NUMERIC}
            candidate.update({f: str(batch[f][i]) for f in data.CATEGORICAL})
            found.append((distance(x, candidate, mads), candidate))

        if len(found) >= k:
            break

    found.sort(key=lambda pair: pair[0])
    best = []
    for d, candidate in found[:k]:
        best.append({
            "distance": d,
            "row": candidate,
            "changes": _changes(x, candidate, mads),
        })
    return best, log


def _changes(x, candidate, mads):
    """Only the features that actually moved, described in the user's units."""
    out = []
    for f in data.NUMERIC:
        delta = float(candidate[f]) - float(x[f])
        if abs(delta) / (mads[f] or 1.0) < 0.02:
            continue                                   # visually identical
        out.append({"feature": data.PRETTY[f], "kind": "number",
                    "was": round(float(x[f]), 1), "now": round(float(candidate[f]), 1),
                    "delta": round(delta, 1), "up": delta > 0})
    for f in data.CATEGORICAL:
        if str(candidate[f]) != str(x[f]):
            out.append({"feature": data.PRETTY[f], "kind": "category",
                        "was": str(x[f]), "now": str(candidate[f]),
                        "delta": None, "up": None})
    return out
