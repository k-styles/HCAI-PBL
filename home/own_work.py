"""What the briefs require us to write ourselves, and where each one lives.

Kept in one place because it has two readers. The project pages show the
relevant entries as a reminder while the code is being written, and the summary
page at /home/own-work/ is what a marker would want to see. Keeping two copies
of this list in sync by hand would guarantee they drift.

Each entry names the file that satisfies it and the marker comment inside that
file, and `audit()` checks the marker is really there -- so deleting or renaming
a marked block shows up as a failure on the page instead of quietly passing.
"""

from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

MARKER = "OWN WORK REQUIRED"


@dataclass
class Requirement:
    project: int
    task: str
    quote: str                  # the brief's own words, verbatim
    kind: str                   # implement | choose | placement
    where: str                  # file that satisfies it, relative to BASE_DIR
    answer: str                 # how it is satisfied

    @property
    def label(self):
        return {"implement": "Must be implemented from scratch",
                "choose": "A design choice the brief leaves to us",
                "placement": "Must be done in a particular place"}[self.kind]


REQUIREMENTS = [
    Requirement(
        project=1, task="Task 1", kind="placement",
        quote="Do this modification from the python, and not in the HTML!",
        where="home/views.py",
        answer="Group members live in the GROUP list in home/views.py and reach the "
               "template only through the view's context. The template loops over "
               "them and hardcodes nothing."),

    Requirement(
        project=2, task="Task 3", kind="choose",
        quote="Repeat the same with logistic regression. Choose a suitable complexity "
              "measure Ω.",
        where="project2/models_lab.py",
        answer="Ω is the number of features with a non-zero coefficient under an L1 "
               "penalty. Leaves count the questions a tree asks; non-zero coefficients "
               "count the features a linear model consults. Both measure how much has "
               "to be read to understand the model, which is what equation (1) is "
               "penalising. Task 4 calling the slider \"sparsity\" points the same way."),

    Requirement(
        project=2, task="Section 2", kind="choose",
        quote="Pay attention to the type of data. If it is decimal, you can proceed as "
              "above, but what about binary or categorical data? Think of an appropriate "
              "way to noise these kinds of features as well.",
        where="project2/counterfactuals.py",
        answer="Numeric features get Gaussian noise scaled by each feature's own MAD, so "
               "the perturbation is in the same units the distance is measured in. "
               "Categorical features cannot be nudged — there is no value between two "
               "islands — so each is resampled from the empirical distribution with a "
               "flip probability, which leaves most draws unchanged and keeps the "
               "proposals realistic."),

    Requirement(
        project=2, task="Task 5", kind="implement",
        quote="The code for the computation of the PDP and ALE values should be written "
              "by you, i.e., do not use a library for them.",
        where="project2/effects.py",
        answer="PDP and ALE are computed directly from the fitted model's predicted "
               "probabilities and numpy arithmetic. No sklearn.inspection, no PartialDependenceDisplay, "
               "no ALE package — the imports at the top of the file are the whole story."),

    Requirement(
        project=3, task="Task 2", kind="implement",
        quote="Design and implement at least one simulated expert. Analyze its strengths "
              "and weaknesses on the dataset and report its accuracy on the test set.",
        where="project3/expert.py",
        answer="The expert's competence is a function of the input text, not of the true "
               "label. That distinction is the whole game: an expert who is reliable "
               "wherever the label happens to be Sports could never be discovered by "
               "Task 4, because at query time the label is exactly what is unknown."),

    Requirement(
        project=3, task="Task 4", kind="choose",
        quote="Choose an active learning strategy for querying the expert. The goal is to "
              "efficiently learn when deferral is beneficial. Justify your choice and "
              "report the results using metrics of your choice.",
        where="project3/acquisition.py",
        answer="The acquisition function targets uncertainty in the deferral decision "
               "rather than in the classifier's own prediction, which is the "
               "instinct to resist here — the classifier's hardest documents are not "
               "the ones where handing over helps most."),

    Requirement(
        project=2, task="Task 5", kind="choose",
        quote="For ALE, you will need partial derivatives. For which model can you "
              "compute them exactly? For which do you have to use a discretization?",
        where="project2/effects.py",
        answer="Logistic regression is differentiable, so the softmax derivative is used "
               "in closed form. A decision tree is piecewise constant: its derivative is "
               "zero almost everywhere and undefined on the split points, so it has to "
               "go through finite differences across bin edges. Both paths are "
               "implemented and the page says which one produced the curve."),

    Requirement(
        project=4, task="Task 1", kind="choose",
        quote="Choose a feature representation that you believe is appropriate for a "
              "movie recommender system. Justify your choice of features and write a "
              "method to extract them from the dataset.",
        where="project4/data.py",
        answer="20 dimensions, chosen by two rules that have nothing to do with which "
               "columns describe a film best. A genre earns a dimension only if two random "
               "films actually differ on it \u2014 probability 2p(1-p), and requiring one "
               "informative pair in ten puts the cut at p >= 0.053. And a feature the "
               "participant cannot see on the card cannot have caused their choice, so "
               "budget, gross and Facebook likes are out. Director and cast are excluded "
               "not for being weak but for being unidentifiable: 2,398 directors, and two "
               "random films share one well under once in a thousand times, so essentially "
               "every response would multiply those weights by zero."),

    Requirement(
        project=4, task="Task 2", kind="implement",
        quote="Propose an extension of the Bradley-Terry model capable of modeling a "
              "ranking of i1 \u227b i2 \u227b \u00b7\u00b7\u00b7 \u227b in instead of a single "
              "pairwise comparison. Explain and justify your proposed formulation.",
        where="project4/preference.py",
        answer="Plackett-Luce: read the ranking as choosing a favourite, then a favourite "
               "of what remains, each stage being Bradley-Terry widened past two items. It "
               "is exactly Bradley-Terry at n = 2, and its pairwise marginals are the "
               "Bradley-Terry probabilities \u2014 which is what lets both interfaces be "
               "scored against the same held-out comparisons instead of quietly measuring "
               "two different quantities. Checked numerically in check_marginals rather "
               "than asserted. The tempting alternative, multiplying all n(n-1)/2 implied "
               "pairs, is rejected because it counts dependent comparisons repeatedly and "
               "does not sum to one over the n! orderings."),
]


def for_project(number):
    return [r for r in REQUIREMENTS if r.project == number]


# Three outcomes, and the difference matters. "pending" is honest work not yet
# done on a project still in progress; "broken" means a marked block was deleted
# or moved and something that was satisfied no longer is. Collapsing them into
# one failure state would make an unfinished project look like a regression.
STATES = {"ok": "ok", "pending": "not written yet", "broken": "marker missing"}


def audit():
    """Check every claimed file exists and still carries its marker comment."""
    report = []
    for r in REQUIREMENTS:
        path = Path(settings.BASE_DIR) / r.where
        if not path.exists():
            state = "pending"
        elif MARKER not in path.read_text(errors="ignore"):
            state = "broken"
        else:
            state = "ok"
        report.append((r, state))
    return report


def summary():
    checked = audit()
    counts = {"ok": 0, "pending": 0, "broken": 0}
    for _, state in checked:
        counts[state] += 1
    return checked, counts
