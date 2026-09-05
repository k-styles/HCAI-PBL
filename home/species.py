"""Photographs of the species being classified, with their attribution.

A column of numbers labelled "Chinstrap" tells you nothing about what a
Chinstrap is, and the identifying feature is often exactly what the
measurements are circling around -- a Gentoo's white eye-patch and orange bill,
the black line under a Chinstrap's chin. Seeing the animal makes the
classification problem concrete in a way no scatter plot does.

Everything here is from Wikimedia Commons under CC0, CC BY or CC BY-SA. Those
licences require attribution, so the credits travel with the images and are
rendered under every photograph rather than being buried in a file nobody opens.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings

CREDITS = Path(settings.BASE_DIR) / "home" / "static" / "home" / "species" / "credits.json"

# Which photograph belongs to which label, per dataset. The keys are the values
# that actually appear in the target column.
DATASETS = {
    "penguins": {
        "Adelie": ("adelie", "Pygoscelis adeliae",
                   "The plain one: all-black head, and a white ring around the eye. "
                   "Shortest bill of the three."),
        "Chinstrap": ("chinstrap", "Pygoscelis antarcticus",
                      "Named for the thin black line under the chin. Bill is long "
                      "but shallow, which is what separates it from Adelie."),
        "Gentoo": ("gentoo", "Pygoscelis papua",
                   "White patch over the eye and a bright orange bill. Much the "
                   "largest, with the longest flippers -- which is why it is the "
                   "easiest of the three for a model to pick out."),
    },
    "iris": {
        "Iris-setosa": ("setosa", "Iris setosa",
                        "Small petals, and the only one of the three that separates "
                        "cleanly from the others on petal length alone."),
        "Iris-versicolor": ("versicolor", "Iris versicolor",
                            "The middle species on almost every measurement, which is "
                            "why most of the errors involve it."),
        "Iris-virginica": ("virginica", "Iris virginica",
                           "The largest petals. Overlaps versicolor enough that no "
                           "straight line separates them perfectly."),
    },
}

# Copies of these datasets disagree about spelling -- iris columns turn up as
# SepalLengthCm, sepal_length and sepal.length, and its labels as "Setosa",
# "setosa" and "Iris-setosa". Rather than enumerating every variant, names are
# reduced to letters only and matched on that.
def _key(text):
    return re.sub(r"[^a-z]", "", str(text).lower())


LABEL_KEYS = {
    "penguins": {"adelie": "Adelie", "chinstrap": "Chinstrap", "gentoo": "Gentoo"},
    "iris": {"setosa": "Iris-setosa", "irissetosa": "Iris-setosa",
             "versicolor": "Iris-versicolor", "irisversicolor": "Iris-versicolor",
             "virginica": "Iris-virginica", "irisvirginica": "Iris-virginica"},
}

# A dataset is recognised by the columns it must have, again as reduced keys.
SIGNATURES = {
    "penguins": [{"billlengthmm", "flipperlengthmm"}],
    "iris": [{"sepallength", "petallength"}, {"sepallengthcm", "petallengthcm"}],
}


@lru_cache(maxsize=1)
def _credits():
    try:
        return json.loads(CREDITS.read_text())
    except (OSError, ValueError):
        return {}


def detect(columns):
    """Which of the datasets we hold photographs for is this, if any."""
    keys = {_key(c) for c in columns}
    for name, options in SIGNATURES.items():
        if any(needed <= keys for needed in options):
            return name
    return None


def for_dataset(name, labels, counts=None):
    """Cards for the labels present, or None when we have no photographs.

    `labels` are the values in the target column, whatever they are spelled
    like; `counts` optionally maps each to how many rows carry it.
    """
    table = DATASETS.get(name)
    lookup = LABEL_KEYS.get(name, {})
    if not table:
        return None

    credits = _credits()
    cards = []
    for raw in labels:
        canonical = lookup.get(_key(raw))
        entry = table.get(canonical) if canonical else None
        if entry is None:
            continue
        slug, latin, note = entry
        credit = credits.get(slug)
        if not credit:
            continue
        cards.append({
            "label": str(raw), "latin": latin, "note": note,
            "image": credit["file"],
            "author": credit["author"], "licence": credit["licence"],
            "licence_url": credit["licence_url"], "source": credit["source"],
            "count": (counts or {}).get(raw) or (counts or {}).get(str(raw)),
        })
    return cards or None
