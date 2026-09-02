"""Figures. Django cannot show a matplotlib canvas directly, so each of these
writes a PNG into MEDIA_ROOT and hands back the URL to put in an <img>."""

import os

import matplotlib
matplotlib.use("Agg")           # no display in a server thread; must precede pyplot

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from django.conf import settings

PALETTE = ["#275CB2", "#D96A29", "#2E9E7C", "#9B4F96", "#C43D5C", "#6E8B3D", "#B08A2E"]
GRID = {"color": "#DDE3EC", "linewidth": 0.8}


def _open(width=7.5, height=4.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9AA7BC")
    return fig, ax


def _save(fig, token, name):
    filename = f"{token}-{name}.png"
    folder = os.path.join(settings.MEDIA_ROOT, "project1")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"{settings.MEDIA_URL}project1/{filename}"


def _column(dataset, name):
    return dataset.y if name == dataset.target else dataset.X[name]


def scatter(dataset, token, x_name, y_name):
    """Two columns against each other.

    On a classification problem both axes are features and the class is carried
    by colour. On a regression problem the target may itself be an axis, which
    is the more direct way to see how one feature drives the outcome; colour is
    then dropped, since it would only repeat an axis.
    """
    fig, ax = _open()
    x, y = _column(dataset, x_name), _column(dataset, y_name)
    against_target = dataset.target in (x_name, y_name)

    if dataset.kind == "classification":
        for i, label in enumerate(dataset.classes()):
            mask = (dataset.y == label).to_numpy()
            ax.scatter(x[mask], y[mask], s=46, alpha=0.85, label=str(label),
                       color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.7)
        ax.legend(title=dataset.target, frameon=False, loc="best")
    elif against_target:
        ax.scatter(x, y, s=46, alpha=0.8, color=PALETTE[0],
                   edgecolor="white", linewidth=0.5)
    else:
        dots = ax.scatter(x, y, c=dataset.y, s=46, alpha=0.9, cmap="viridis",
                          edgecolor="white", linewidth=0.5)
        fig.colorbar(dots, ax=ax, label=dataset.target)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{y_name} against {x_name}", pad=12, fontsize=13)
    return _save(fig, token, "scatter")


def distributions(dataset, token):
    """One panel per feature, split by class so overlap is visible.

    On a regression problem the target gets a panel of its own: a skewed or
    long-tailed outcome is worth knowing about before choosing a score, since
    squared error will chase the tail.
    """
    names = dataset.features + ([dataset.target] if dataset.kind == "regression" else [])
    cols = min(2, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.9 * rows), dpi=120)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        ax.set_facecolor("#FBFCFE")
        ax.grid(True, axis="y", **GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        column = _column(dataset, name)
        edges = np.histogram_bin_edges(column, bins="auto")
        if name == dataset.target:
            ax.hist(column, bins=edges, color=PALETTE[2], alpha=0.85)
            ax.set_title(f"{name}  (target)", fontsize=11)
            continue
        if dataset.kind == "classification":
            for i, label in enumerate(dataset.classes()):
                mask = (dataset.y == label).to_numpy()
                ax.hist(column[mask], bins=edges, alpha=0.62, label=str(label),
                        color=PALETTE[i % len(PALETTE)])
        else:
            ax.hist(column, bins=edges, color=PALETTE[0], alpha=0.85)
        ax.set_title(name, fontsize=11)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    if dataset.kind == "classification":
        handles = [Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=PALETTE[i % len(PALETTE)], label=str(l))
                   for i, l in enumerate(dataset.classes())]
        fig.legend(handles=handles, title=dataset.target, frameon=False,
                   loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _save(fig, token, "dist")


def correlations(dataset, token):
    """How much the features duplicate each other, and the target if it is numeric."""
    frame = dataset.X.copy()
    if dataset.kind == "regression":
        frame[dataset.target] = dataset.y
    matrix = frame.corr().to_numpy()
    names = list(frame.columns)

    size = max(4.2, 0.72 * len(names) + 2.2)
    fig, ax = plt.subplots(figsize=(size, size * 0.86), dpi=120)
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(value) > 0.55 else "#2A3242")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Pearson correlation")
    ax.set_title("Feature correlations", pad=12, fontsize=13)
    fig.tight_layout()
    return _save(fig, token, "corr")


def validation_curve(result, token):
    """The sweep: held-out score with its one-standard-error band, against the
    score on the rows the model was fitted on. The gap is the overfitting."""
    fig, ax = _open(width=7.8, height=4.6)
    values = [r.value for r in result.rows]
    mean = np.array([r.mean for r in result.rows])
    se = np.array([r.se for r in result.rows])
    train = np.array([r.train_mean for r in result.rows])

    positions = np.arange(len(values))
    ax.fill_between(positions, mean - se, mean + se, color=PALETTE[0], alpha=0.16,
                    label="±1 standard error")
    ax.plot(positions, mean, "-o", color=PALETTE[0], markersize=6, label="cross-validation")
    ax.plot(positions, train, "--s", color=PALETTE[1], markersize=5, alpha=0.9,
            label="training rows")

    chosen = positions[[r.chosen for r in result.rows].index(True)]
    ax.axvline(chosen, color="#2A3242", linestyle=":", linewidth=1.4)
    ax.annotate(f"chosen: {result.algo.param} = {result.winner.value:g}",
                xy=(chosen, mean[chosen]), xytext=(6, 12), textcoords="offset points",
                fontsize=10, color="#2A3242",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F2F5FA", edgecolor="#C6D0E0"))

    ax.set_xticks(positions, [f"{v:g}" for v in values])
    ax.set_xlabel(result.algo.param)
    ax.set_ylabel(result.score.label)
    ax.set_title(f"{result.algo.label}: {result.n_folds}-fold sweep over {result.algo.param}",
                 pad=12, fontsize=13)
    ax.legend(frameon=False)
    return _save(fig, token, "sweep")


def predicted_against_actual(truth, predicted, token, target_name):
    """Regression diagnostic: where the model is wrong, and in which direction."""
    fig, ax = _open(width=5.6, height=5.0)
    ax.scatter(truth, predicted, s=42, alpha=0.75, color=PALETTE[0],
               edgecolor="white", linewidth=0.6)
    span = [min(truth.min(), predicted.min()), max(truth.max(), predicted.max())]
    ax.plot(span, span, "--", color="#6B7688", linewidth=1.3, label="perfect prediction")
    ax.set_xlabel(f"actual {target_name}")
    ax.set_ylabel(f"predicted {target_name}")
    ax.set_title("Test set predictions", pad=12, fontsize=13)
    ax.legend(frameon=False)
    return _save(fig, token, "fit")


def pair_grid(dataset, token, names):
    """Every pair of the chosen features at once.

    The single scatter above answers "do these two separate the classes"; this
    answers "which two should I have picked". Histograms sit on the diagonal so
    each feature's own distribution is in the same picture as its interactions.
    """
    n = len(names)
    size = 1.65
    fig, axes = plt.subplots(n, n, figsize=(size * n + 1.2, size * n + 0.9), dpi=110)
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    classes = dataset.classes() if dataset.kind == "classification" else []
    masks = [(dataset.y == c).to_numpy() for c in classes]

    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            ax.set_facecolor("#FBFCFE")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=7, length=2)

            xs, ys = _column(dataset, names[col]), _column(dataset, names[row])
            if row == col:
                edges = np.histogram_bin_edges(xs, bins=12)
                if masks:
                    for i, mask in enumerate(masks):
                        ax.hist(xs[mask], bins=edges, alpha=0.6,
                                color=PALETTE[i % len(PALETTE)])
                else:
                    ax.hist(xs, bins=edges, color=PALETTE[0], alpha=0.85)
            elif masks:
                for i, mask in enumerate(masks):
                    ax.scatter(xs[mask], ys[mask], s=8, alpha=0.75,
                               color=PALETTE[i % len(PALETTE)], linewidth=0)
            else:
                ax.scatter(xs, ys, s=8, alpha=0.7, c=dataset.y, cmap="viridis", linewidth=0)

            if row != n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[col], fontsize=8.5)
            if col != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[row], fontsize=8.5)

    if classes:
        handles = [Line2D([], [], marker="s", linestyle="", markersize=8,
                          color=PALETTE[i % len(PALETTE)], label=str(c))
                   for i, c in enumerate(classes)]
        fig.legend(handles=handles, title=dataset.target, frameon=False,
                   loc="lower center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Every pair of the most informative features", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return _save(fig, token, "pairs")


def projection(dataset, token):
    """The data flattened onto the two directions it varies in most.

    Columns are standardised first, otherwise a feature that happens to be
    measured in larger units defines the axes on its own. Written out as an SVD
    of the centred matrix rather than pulled from a library, since that is all
    principal components are.
    """
    X = dataset.X.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    spread = X.std(axis=0)
    spread[spread == 0] = 1.0
    X = X / spread

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    share = (S ** 2) / np.sum(S ** 2)

    fig, ax = _open(width=6.6, height=5.0)
    if dataset.kind == "classification":
        for i, label in enumerate(dataset.classes()):
            mask = (dataset.y == label).to_numpy()
            ax.scatter(coords[mask, 0], coords[mask, 1], s=42, alpha=0.85, label=str(label),
                       color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.6)
        ax.legend(title=dataset.target, frameon=False, loc="best")
    else:
        dots = ax.scatter(coords[:, 0], coords[:, 1], c=dataset.y, s=42, alpha=0.88,
                          cmap="viridis", edgecolor="white", linewidth=0.4)
        fig.colorbar(dots, ax=ax, label=dataset.target)

    ax.set_xlabel(f"first component  ({share[0] * 100:.1f}% of the variance)")
    ax.set_ylabel(f"second component  ({share[1] * 100:.1f}% of the variance)")
    ax.set_title("All features projected onto two directions", pad=12, fontsize=13)
    return _save(fig, token, "pca"), [round(float(s) * 100, 1) for s in share[:4]]
