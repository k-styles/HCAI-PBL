"""Figures for project 3. Agg backend, written into MEDIA_ROOT, filenames keyed
so two browser tabs cannot overwrite each other's plots."""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from django.conf import settings

MACHINE = "#275CB2"
EXPERT = "#D96A29"
GOOD = "#2E9E7C"
GREY = "#9AA7BC"
GRID = {"color": "#DDE3EC", "linewidth": 0.8}


def _open(width=7.4, height=4.4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    return fig, ax


def _save(fig, name):
    folder = os.path.join(settings.MEDIA_ROOT, "project3")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"{settings.MEDIA_URL}project3/{name}"


def regions(rows, token):
    """Who is better where. The point of the figure is the crossings: being the
    expert's speciality does not automatically mean it is worth handing over."""
    fig, ax = _open(width=8.2, height=4.6)
    order = sorted(rows, key=lambda r: -r["advantage"])
    x = np.arange(len(order))

    ax.bar(x - 0.2, [r["classifier"] for r in order], 0.4, label="classifier",
           color=MACHINE, edgecolor="white")
    ax.bar(x + 0.2, [r["expert"] for r in order], 0.4, label="expert",
           color=EXPERT, edgecolor="white")

    for i, r in enumerate(order):
        if r["worth_deferring"]:
            ax.plot(i, max(r["expert"], r["classifier"]) + 0.035, marker="v",
                    color=GOOD, markersize=7)

    ax.set_xticks(x, [f"{r['cluster']}\n(n={r['n']})" for r in order], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("region of the input space, ordered by how much the expert helps")
    ax.set_ylabel("accuracy within the region")
    ax.set_title("Green markers: regions where handing over actually pays",
                 pad=12, fontsize=12.5)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    return _save(fig, f"{token}-regions.png")


def tradeoff(sweep, refs, chosen, machine_accuracy, token):
    """Team accuracy against how much of the expert's time is spent."""
    fig, ax = _open(width=7.6, height=4.6)
    rates = [r["deferral_rate"] for r in sweep]
    accs = [r["accuracy"] for r in sweep]

    ax.plot(rates, accs, "-", color=MACHINE, linewidth=2.0, label="learned policy")
    ax.scatter([chosen["deferral_rate"]], [chosen["accuracy"]], s=190, marker="*",
               color=EXPERT, edgecolor="white", zorder=5,
               label=f"current threshold (τ = {chosen['tau']:+.2f})")

    ax.axhline(machine_accuracy, color=GREY, linestyle="--", linewidth=1.2)
    ax.annotate("classifier alone, nobody disturbed", xy=(0.02, machine_accuracy),
                xytext=(0, 6), textcoords="offset points", fontsize=9, color="#5A677E")

    ax.axhline(refs["oracle"]["accuracy"], color=GOOD, linestyle=":", linewidth=1.4)
    ax.annotate("oracle — knows who is right, unreachable",
                xy=(0.02, refs["oracle"]["accuracy"]), xytext=(0, 6),
                textcoords="offset points", fontsize=9, color="#1E6650")

    ax.scatter([refs["by_confidence"]["deferral_rate"]], [refs["by_confidence"]["accuracy"]],
               s=95, marker="s", color="#9B4F96", edgecolor="white", zorder=4,
               label="hand over what the classifier is least sure of")

    ax.set_xlabel("share of articles handed to the expert")
    ax.set_ylabel("team accuracy")
    ax.set_title("What each level of interruption buys", pad=12, fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    return _save(fig, f"{token}-tradeoff.png")


def outcomes(counts, token):
    """The four cases, as a single stacked bar of where the deferred work went."""
    fig, ax = _open(width=7.6, height=2.6)
    order = [("expert_only", "worth handing over", GOOD),
             ("both", "both would be right — wasted", "#B9CBE8"),
             ("neither", "neither is right — pointless", GREY),
             ("machine_only", "classifier was right — harmful", "#C43D5C")]
    left = 0
    total = max(sum(counts[k] for k, _, _ in order), 1)
    for key, label, colour in order:
        width = counts[key] / total
        if width <= 0:
            continue
        ax.barh([0], [width], left=left, color=colour, edgecolor="white", height=0.55,
                label=f"{label} ({counts[key]})")
        left += width

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("of everything handed to the expert")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.42), ncol=2)
    return _save(fig, f"{token}-outcomes.png")


def learning_curves(curves, labels, machine_accuracy, full_information, token):
    """Team accuracy against the number of questions asked of the expert."""
    fig, ax = _open(width=7.8, height=4.6)
    colours = {"boundary": MACHINE, "entropy": EXPERT, "random": GREY, "stratified": "#9B4F96"}

    for key, rows in curves.items():
        budgets = [r["budget"] for r in rows]
        mean = np.array([r["accuracy"] for r in rows])
        sd = np.array([r["accuracy_sd"] for r in rows])
        ax.fill_between(budgets, mean - sd, mean + sd, color=colours[key], alpha=0.13)
        ax.plot(budgets, mean, "-o", markersize=4.5, linewidth=2.0 if key == "boundary" else 1.4,
                color=colours[key], label=labels[key].split(" — ")[0])

    ax.axhline(full_information, color=GOOD, linestyle=":", linewidth=1.4)
    ax.annotate("every expert answer known", xy=(budgets[0], full_information),
                xytext=(0, 5), textcoords="offset points", fontsize=9, color="#1E6650")
    ax.axhline(machine_accuracy, color="#B9C2D2", linestyle="--", linewidth=1.1)
    ax.annotate("classifier alone", xy=(budgets[0], machine_accuracy), xytext=(0, -13),
                textcoords="offset points", fontsize=9, color="#7C89A2")

    ax.set_xscale("log")
    ax.set_xticks(budgets, [str(b) for b in budgets], fontsize=9)
    ax.set_xlabel("questions asked of the expert (log scale)")
    ax.set_ylabel("team accuracy")
    ax.set_title("Does choosing the questions beat asking at random?", pad=12, fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    return _save(fig, f"{token}-curves.png")
