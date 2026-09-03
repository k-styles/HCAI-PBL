"""The Palmer Penguins dataset, prepared once and shared by every model.

One split is made and reused everywhere. Task 2 compares models by their test
accuracy, so a model must not be able to look better simply by having been
handed an easier test set.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV = Path(__file__).resolve().parent / "data" / "penguins.csv"

TARGET = "species"
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["island", "sex", "year"]

PRETTY = {
    "bill_length_mm": "bill length (mm)",
    "bill_depth_mm": "bill depth (mm)",
    "flipper_length_mm": "flipper length (mm)",
    "body_mass_g": "body mass (g)",
    "island": "island",
    "sex": "sex",
    "year": "year",
}

SPLIT_SEED = 0
TEST_FRACTION = 0.3


@dataclass
class Penguins:
    frame: pd.DataFrame          # cleaned, human-readable, categoricals as text
    X: np.ndarray                # one-hot encoded, model-ready
    y: np.ndarray                # species labels as text
    columns: list                # names of the encoded columns
    origin: list                 # which original feature each encoded column came from
    train: np.ndarray            # row positions
    test: np.ndarray
    dropped: int

    @property
    def classes(self):
        return sorted(set(self.y))

    @property
    def features(self):
        return NUMERIC + CATEGORICAL

    def categories(self, feature):
        return sorted(self.frame[feature].astype(str).unique())

    def encode(self, row: dict) -> np.ndarray:
        """Turn one human-readable row into the model's encoded vector."""
        vector = np.zeros(len(self.columns))
        for i, (column, source) in enumerate(zip(self.columns, self.origin)):
            if source in NUMERIC:
                vector[i] = float(row[source])
            else:
                vector[i] = 1.0 if str(row[source]) == column.split("=", 1)[1] else 0.0
        return vector

    def decode(self, vector: np.ndarray) -> dict:
        """The inverse, for showing a generated point back to the user."""
        row = {}
        for feature in NUMERIC:
            row[feature] = float(vector[self.columns.index(feature)])
        for feature in CATEGORICAL:
            options = [(i, c) for i, (c, s) in enumerate(zip(self.columns, self.origin))
                       if s == feature]
            best = max(options, key=lambda pair: vector[pair[0]])
            row[feature] = best[1].split("=", 1)[1]
        return row


@lru_cache(maxsize=1)
def load() -> Penguins:
    raw = pd.read_csv(CSV)
    frame = raw.dropna().reset_index(drop=True)
    frame["year"] = frame["year"].astype(int).astype(str)

    columns, origin, blocks = [], [], []
    for feature in NUMERIC:
        columns.append(feature)
        origin.append(feature)
        blocks.append(frame[[feature]].to_numpy(dtype=float))
    for feature in CATEGORICAL:
        for level in sorted(frame[feature].astype(str).unique()):
            columns.append(f"{feature}={level}")
            origin.append(feature)
        blocks.append(pd.get_dummies(frame[feature].astype(str), prefix=feature,
                                     prefix_sep="=").to_numpy(dtype=float))

    X = np.hstack(blocks)
    y = frame[TARGET].to_numpy()

    positions = np.arange(len(frame))
    train, test = train_test_split(positions, test_size=TEST_FRACTION,
                                   random_state=SPLIT_SEED, stratify=y)

    return Penguins(frame=frame, X=X, y=y, columns=columns, origin=origin,
                    train=np.sort(train), test=np.sort(test),
                    dropped=len(raw) - len(frame))


def mad(values) -> float:
    """Median absolute deviation -- the scale the counterfactual distance uses."""
    values = np.asarray(values, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def numeric_mads(penguins) -> dict:
    return {f: mad(penguins.frame[f]) for f in NUMERIC}
