"""Bradley-Terry, its extension to full rankings, and fitting w from responses.

Nothing here comes out of a library: the likelihood, its gradient, the MAP
optimiser and the Fisher information are all written out, because the whole
point of Task 2 is the derivation.
"""

import numpy as np

# The prior. With ~25 responses in 20 dimensions the unpenalised likelihood is
# often separable -- some direction in w can be pushed to infinity without ever
# contradicting the data -- so the MLE does not exist. sigma = 1 says a one
# standard deviation move in a feature is worth about one unit of utility,
# which is roughly a 73/27 preference. That is a weak claim, and it is the
# thing that keeps the fit finite.
PRIOR_SIGMA = 1.0


# ---------------------------------------------------------------------------
# OWN WORK REQUIRED -- Project 4, Task 2
#
#   "Propose an extension of the Bradley-Terry model capable of modeling a
#    ranking of i1 > i2 > ... > in instead of a single pairwise comparison.
#    Explain and justify your proposed formulation."
#
# Bradley-Terry over utilities U(x) = w'x is
#
#     P(i > j) = exp(U_i) / (exp(U_i) + exp(U_j)) = sigmoid(w'(x_i - x_j)).
#
# Read a ranking as the participant choosing their favourite, then their
# favourite of what is left, and so on. Each of those choices is Bradley-Terry
# widened from two items to however many remain, which gives
#
#     P(i1 > i2 > ... > in) = prod_{k=1}^{n-1} exp(U_ik) / sum_{l>=k} exp(U_il)
#
# -- the Plackett-Luce model. Three reasons this is the right extension and not
# just one that happens to work.
#
# 1. At n = 2 the product has a single factor and is exactly Bradley-Terry, so
#    the two interfaces in this study are not fitting two different models.
#
# 2. Its pairwise marginals are the Bradley-Terry probabilities: the chance that
#    i appears somewhere before j in a Plackett-Luce ranking is
#    exp(U_i)/(exp(U_i)+exp(U_j)), independent of the other eight films in the
#    set. This is what makes the study legitimate at all. Design 1 and Design 2
#    are estimating the same w, so scoring both against the same held-out
#    pairwise comparisons is a fair test rather than a category error. It is
#    checked numerically in `check_marginals`.
#
# 3. The obvious alternative -- split the ranking into all n(n-1)/2 implied
#    pairs and multiply their Bradley-Terry probabilities -- is wrong twice
#    over. Those pairs are not independent (i1 > i3 is partly implied by
#    i1 > i2 and i2 > i3), so it counts the same evidence repeatedly and
#    reports far more confidence than it has; and the resulting product is not
#    a distribution over the n! orderings, it does not sum to one, so calling
#    it a likelihood is a mistake. Plackett-Luce does sum to one by
#    construction, being a product of properly normalised choices.
#
# What it inherits from Bradley-Terry, and what a reader should be told: Luce's
# choice axiom. The relative odds of i over j do not change when other films
# join or leave the set. Real people violate this -- put two near-identical
# sequels in a set of ten and both get pushed down -- and that is a limitation
# of the model, not of the implementation. The study design in the report
# treats it as a threat to validity rather than pretending it away.
# ---------------------------------------------------------------------------


def ranking_log_likelihood(w, X, rankings):
    """log P of each ranking under Plackett-Luce. `rankings` are index lists
    already in preference order, best first."""
    total = 0.0
    for order in rankings:
        u = X[list(order)] @ w
        # Stage k normalises over the tail u[k:]; a reversed cumulative
        # log-sum-exp gives all n-1 denominators in one pass.
        shift = u.max()
        tail = np.cumsum(np.exp(u - shift)[::-1])[::-1]
        total += float(np.sum(u[:-1] - (np.log(tail[:-1]) + shift)))
    return total


def ranking_gradient(w, X, rankings):
    """d/dw of the above. At stage k the gradient is the chosen film's features
    minus the softmax-weighted average over the films still available."""
    g = np.zeros_like(w)
    for order in rankings:
        rows = X[list(order)]
        u = rows @ w
        for k in range(len(order) - 1):
            v = u[k:]
            p = np.exp(v - v.max())
            p /= p.sum()
            g += rows[k] - p @ rows[k:]
    return g


def fit(X, rankings, sigma=PRIOR_SIGMA, steps=3000, tol=1e-8):
    """MAP estimate of w. Gradient ascent on the penalised log-likelihood with
    backtracking, which is enough for a problem this size and avoids pulling in
    an optimiser whose failure modes we would then have to explain."""
    w = np.zeros(X.shape[1])
    if not rankings:
        return w, {"steps": 0, "objective": 0.0, "converged": True}

    penalty = 1.0 / (2 * sigma ** 2)

    def objective(v):
        return ranking_log_likelihood(v, X, rankings) - penalty * float(v @ v)

    value = objective(w)
    rate = 0.5
    taken = 0
    norm = float("inf")
    for _ in range(steps):
        g = ranking_gradient(w, X, rankings) - 2 * penalty * w
        norm = float(np.linalg.norm(g))
        if norm < tol:
            break
        while rate > 1e-12:
            candidate = w + rate * g
            proposed = objective(candidate)
            if proposed >= value:
                w, value = candidate, proposed
                rate *= 1.6
                taken += 1
                break
            rate *= 0.5
        else:
            break
    # The objective is strictly concave (concave log-likelihood plus a strictly
    # concave penalty), so a small gradient really does mean the optimum.
    return w, {"steps": taken, "objective": value,
               "gradient": norm, "converged": norm < 1e-4}


def pair_probability(w, X, i, j):
    """Bradley-Terry, and by point 2 above also the Plackett-Luce marginal."""
    return float(1 / (1 + np.exp(-(X[i] - X[j]) @ w)))


def score_holdout(w, X, comparisons):
    """How well a fitted w predicts comparisons it was not fitted on.

    This is the study's primary outcome. w is latent, so 'how close is the
    estimate' is not a measurable quantity; how well it predicts fresh choices
    by the same person is."""
    if not comparisons:
        return None
    hits, loss = 0.0, 0.0
    for chosen, rejected in comparisons:
        p = pair_probability(w, X, chosen, rejected)
        p = min(max(p, 1e-9), 1 - 1e-9)
        # An untrained w gives every pair exactly 0.5. Calling that a miss
        # would flatter any fit at all; it is a coin toss and scores as one.
        hits += 1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0)
        loss -= np.log(p)
    return {"n": len(comparisons),
            "accuracy": hits / len(comparisons),
            "log_loss": loss / len(comparisons)}


# --- the two checks the report leans on ------------------------------------

def check_marginals(w, X, items, trials=200_000, seed=0):
    """Sample rankings from Plackett-Luce and confirm that P(i before j)
    matches Bradley-Terry. This is claim 2, verified rather than asserted."""
    rng = np.random.default_rng(seed)
    u = X[list(items)] @ w
    n = len(items)
    before = np.zeros((n, n))
    weights = np.exp(u - u.max())
    for _ in range(trials):
        # Gumbel-max sampling draws an exact Plackett-Luce ranking in one shot:
        # perturb each log-utility with a Gumbel and sort.
        order = np.argsort(-(np.log(weights) + rng.gumbel(size=n)))
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        before += (rank[:, None] < rank[None, :])
    empirical = before / trials
    predicted = 1 / (1 + np.exp(-(u[:, None] - u[None, :])))
    off = ~np.eye(n, dtype=bool)
    return {"max_error": float(np.abs(empirical - predicted)[off].max()),
            "empirical": empirical, "predicted": predicted}


def information_per_task(w, X, indices, kind):
    """Fisher information contributed by one task, as a scalar (the trace).

    Used in the report to say how much a rank-of-ten is worth against a pair.
    For a softmax choice among a set S the information is
    sum_S p_s x_s x_s' - (sum_S p_s x_s)(sum_S p_s x_s)', and a Plackett-Luce
    ranking contributes one such term per stage. A pairwise comparison is the
    n = 2 case."""
    rows = X[list(indices)]
    u = rows @ w
    stages = range(len(indices) - 1) if kind == "rank" else [0]
    total = np.zeros((X.shape[1], X.shape[1]))
    for k in stages:
        r = rows[k:]
        p = np.exp(u[k:] - u[k:].max())
        p /= p.sum()
        mean = p @ r
        total += (r * p[:, None]).T @ r - np.outer(mean, mean)
    return total
