"""Figures for project 2. Same conventions as project 1: Agg backend, written
into MEDIA_ROOT, filenames keyed so two tabs cannot overwrite each other."""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

from django.conf import settings

from . import data

SPECIES = {"Adelie": "#275CB2", "Chinstrap": "#D96A29", "Gentoo": "#2E9E7C"}
GRID = {"color": "#DDE3EC", "linewidth": 0.8}


def _open(width=7.2, height=4.4):
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


def _save(fig, name):
    folder = os.path.join(settings.MEDIA_ROOT, "project2")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"{settings.MEDIA_URL}project2/{name}"


def frontier(candidates, hull, selected, lam, omega_name, token):
    """Accuracy against complexity, with the reachable models marked.

    The line is the upper concave hull. A model below it loses to something on
    it at every lambda, so no position of the slider will ever produce it --
    which is worth seeing, because several of them look perfectly reasonable in
    the table.
    """
    fig, ax = _open(width=7.4, height=4.6)
    on_hull = {id(c) for c in hull}

    dominated = [c for c in candidates if id(c) not in on_hull]
    if dominated:
        ax.scatter([c.omega for c in dominated], [c.accuracy for c in dominated],
                   s=64, color="#C6D0E0", edgecolor="white", linewidth=1.2,
                   zorder=2, label="never selectable")

    ax.plot([c.omega for c in hull], [c.accuracy for c in hull], "-",
            color="#9AB4DC", linewidth=1.6, zorder=1)
    ax.scatter([c.omega for c in hull], [c.accuracy for c in hull], s=76,
               color="#275CB2", edgecolor="white", linewidth=1.2, zorder=3,
               label="reachable by some λ")
    ax.scatter([selected.omega], [selected.accuracy], s=250, marker="*",
               color="#D96A29", edgecolor="white", linewidth=1.0, zorder=4,
               label=f"selected at λ = {lam:g}")

    ax.set_xlabel(omega_name + "   —   Ω(f)")
    ax.set_ylabel("accuracy on the test set")
    ax.set_title("What the slider can and cannot reach", pad=12, fontsize=13)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    return _save(fig, f"{token}-frontier.png")


def objective(candidates, hull, lam, token):
    """acc − λΩ for every candidate, as λ runs from 0 upwards.

    Each model is a straight line; the winner is whichever is highest. Drawing
    them together is the clearest way to show why most never win: their line
    spends its whole life underneath someone else's.
    """
    fig, ax = _open(width=7.4, height=4.2)
    top = max(c.accuracy for c in candidates)
    span = np.linspace(0, min(0.4, max(0.05, top / max(1, min(c.omega for c in hull) + 1))), 200)
    on_hull = {id(c) for c in hull}

    for c in candidates:
        winner = id(c) in on_hull
        ax.plot(span, [c.objective(l) for l in span],
                color="#275CB2" if winner else "#CBD5E5",
                linewidth=1.8 if winner else 1.0,
                zorder=3 if winner else 1)

    envelope = [max(c.objective(l) for c in candidates) for l in span]
    ax.plot(span, envelope, "--", color="#D96A29", linewidth=1.6, zorder=4,
            label="the best available")
    ax.axvline(lam, color="#2A3242", linestyle=":", linewidth=1.4, zorder=5)
    ax.annotate(f"λ = {lam:g}", xy=(lam, ax.get_ylim()[0]), xytext=(5, 8),
                textcoords="offset points", fontsize=10, color="#2A3242")

    ax.set_xlabel("λ")
    ax.set_ylabel("accuracy − λ · Ω(f)")
    ax.set_title("One line per model; the slider picks the highest", pad=12, fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    return _save(fig, f"{token}-objective.png")


def tree_diagram(estimator, penguins, token):
    leaves = estimator.get_n_leaves()
    width = max(7.5, min(22, 1.9 * leaves))
    fig, ax = plt.subplots(figsize=(width, max(4.0, 1.5 * estimator.get_depth() + 1.5)),
                           dpi=110)
    fig.patch.set_facecolor("white")
    names = [c.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ")
             for c in penguins.columns]
    plot_tree(estimator, feature_names=names, class_names=list(estimator.classes_),
              filled=True, rounded=True, impurity=False, proportion=False,
              fontsize=9, ax=ax)
    ax.set_title(f"{leaves} leaves", fontsize=12, color="#16233A")
    return _save(fig, f"{token}-tree.png")


def effect_curves(curve, token):
    """PDP and ALE for one feature, three species per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=120)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#FBFCFE")
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for i, species in enumerate(curve["classes"]):
        axes[0].plot(curve["pdp_grid"], curve["pdp"][:, i], "-", linewidth=2.0,
                     color=SPECIES.get(species, "#666"), label=species)
        axes[1].plot(curve["ale_edges"], curve["ale"][:, i], "-", linewidth=2.0,
                     color=SPECIES.get(species, "#666"), label=species)

    axes[0].set_title("Partial dependence", fontsize=12.5, pad=10)
    axes[0].set_ylabel("predicted probability")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Accumulated local effects", fontsize=12.5, pad=10)
    axes[1].set_ylabel("change in probability")
    axes[1].axhline(0, color="#9AA7BC", linewidth=0.9)

    for ax in axes:
        ax.set_xlabel(curve["pretty"])
        ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout()
    return _save(fig, f"{token}-effects.png")
