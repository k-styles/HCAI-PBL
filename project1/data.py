"""Reading a user-supplied CSV and deciding what it actually contains."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DatasetError(Exception):
    pass


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    kind: str                              # "classification" or "regression"
    detected_kind: str                     # what we inferred, before any override
    notes: list = field(default_factory=list)

    @property
    def features(self):
        return list(self.X.columns)

    @property
    def target(self):
        return self.y.name

    @property
    def n_rows(self):
        return len(self.X)

    def classes(self):
        return sorted(self.y.unique(), key=str)

    def describe(self):
        """Rows of (feature, mean, std, min, median, max) for the summary table."""
        out = []
        for name in self.X.columns:
            col = self.X[name]
            out.append({
                "name": name,
                "mean": col.mean(),
                "std": col.std(),
                "min": col.min(),
                "median": col.median(),
                "max": col.max(),
            })
        return out


def looks_like_row_number(col: pd.Series) -> bool:
    """True for a column that only numbers the rows: integer, unique, consecutive.

    Matching on the name alone ("id") is too eager -- a patient id can be a
    genuine feature.  Requiring the values to form an unbroken run is what
    actually distinguishes bookkeeping from data.
    """
    if not pd.api.types.is_integer_dtype(col):
        return False
    v = np.sort(col.to_numpy())
    if len(np.unique(v)) != len(v) or len(v) < 10:
        return False
    return bool((np.diff(v) == 1).all())


def infer_kind(y: pd.Series) -> str:
    """Classification or regression, from the target column alone.

    Anything non-numeric is a label.  A numeric target is only treated as
    labels if the values are whole numbers *and* there are few enough distinct
    ones to plausibly be classes.  A flat "10 or fewer" cutoff gets this wrong
    in both directions, so the cutoff grows with the number of rows: three
    distinct values out of 20 is a class, three out of 20000 is a coincidence
    worth flagging but not enough to call it a class -- hence the floor of 10.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    v = y.dropna().to_numpy()
    if v.size == 0:
        raise DatasetError("The target column is empty.")
    if not np.all(np.isclose(v, np.round(v))):
        return "regression"

    cutoff = min(20, max(10, int(np.sqrt(v.size))))
    return "classification" if len(np.unique(v)) <= cutoff else "regression"


def load_csv(source, kind=None) -> Dataset:
    """Last column is the target, first row is the header (as the brief specifies)."""
    try:
        frame = pd.read_csv(source, skipinitialspace=True)
    except Exception as exc:
        raise DatasetError(f"Could not parse the file as CSV: {exc}")

    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.shape[1] < 2:
        raise DatasetError("Need at least one feature column and one target column.")

    if len(frame) < 10:
        raise DatasetError(f"Only {len(frame)} rows -- too few to train and test on.")

    notes = []
    y = frame.iloc[:, -1]
    X = frame.iloc[:, :-1]

    missing_target = y.isna()
    if missing_target.any():
        notes.append(f"Dropped {int(missing_target.sum())} row(s) with no value for '{y.name}'.")
        X, y = X[~missing_target], y[~missing_target]

    for name in list(X.columns):
        if looks_like_row_number(X[name]):
            X = X.drop(columns=name)
            notes.append(f"Ignored '{name}': it just numbers the rows.")

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = X.drop(columns=non_numeric)
        notes.append("Ignored non-numeric feature(s): " + ", ".join(non_numeric) + ".")

    if X.shape[1] == 0:
        raise DatasetError("No usable numeric features are left after cleaning.")

    incomplete = X.isna().any(axis=1)
    if incomplete.any():
        notes.append(f"Dropped {int(incomplete.sum())} row(s) with missing feature values.")
        X, y = X[~incomplete], y[~incomplete]

    if len(X) < 10:
        raise DatasetError(f"Only {len(X)} complete rows are left -- too few to train and test on.")

    detected = infer_kind(y)
    chosen = kind or detected

    # The override is the user's to make, but not every override is possible:
    # text labels have no numeric order, so nothing downstream can regress on
    # them. Refuse it here rather than letting it fail somewhere further in.
    if chosen == "regression" and not pd.api.types.is_numeric_dtype(y):
        sample = ", ".join(str(v) for v in y.drop_duplicates().head(3))
        raise DatasetError(
            f"'{y.name}' holds text labels ({sample}...). There is no meaningful value "
            f"halfway between two labels, so this cannot be a regression target.")

    if chosen == "classification":
        counts = y.value_counts()
        if counts.min() < 2:
            # A continuous column forced into classes fails this test in bulk, and
            # listing two hundred singleton "classes" explains nothing. Name the
            # actual problem instead.
            if detected == "regression":
                raise DatasetError(
                    f"'{y.name}' takes {len(counts)} distinct values across {len(y)} rows. "
                    f"That is a quantity to predict, not a set of labels to choose between.")
            rare = counts[counts < 2].index
            shown = ", ".join(str(c) for c in rare[:4])
            extra = f" and {len(rare) - 4} other(s)" if len(rare) > 4 else ""
            raise DatasetError(
                f"Class(es) {shown}{extra} appear only once, so they cannot be split "
                f"across a training and a test set.")

    return Dataset(X=X.reset_index(drop=True), y=y.reset_index(drop=True),
                   kind=chosen, detected_kind=detected, notes=notes)


def preview(dataset, n=10):
    """The first rows as they are after cleaning, so the user can check that the
    file was understood before drawing anything from it."""
    frame = dataset.X.head(n).copy()
    frame[dataset.target] = dataset.y.head(n)
    return {
        "columns": list(frame.columns),
        "rows": [[_cell(v) for v in row] for row in frame.itertuples(index=False)],
        "shown": len(frame),
    }


def _cell(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def feature_relevance(dataset):
    """Rank the features by how much each one says about the target on its own.

    For regression that is the magnitude of the correlation.  For classification
    it is the ratio of between-class to within-class variance -- the quantity a
    one-way ANOVA is built on: a feature scores highly when the class means are
    far apart relative to the scatter inside each class.

    Both measures look at one feature at a time, so neither sees two features
    that are useless alone and decisive together.  This is a reading aid for
    choosing what to plot, not a feature selector, and the page says so.
    """
    scores = []

    if dataset.kind == "regression":
        for name in dataset.X.columns:
            r = dataset.X[name].corr(dataset.y)
            scores.append({"name": name, "score": 0.0 if pd.isna(r) else abs(float(r))})
        measure = "|correlation with target|"
    else:
        groups = [dataset.X[(dataset.y == c).to_numpy()] for c in dataset.classes()]
        k, n = len(groups), len(dataset.X)
        for name in dataset.X.columns:
            column = dataset.X[name]
            grand = column.mean()
            between = sum(len(g) * (g[name].mean() - grand) ** 2 for g in groups)
            within = sum((len(g) - 1) * g[name].var() for g in groups)
            if within <= 0 or k < 2 or n <= k:
                scores.append({"name": name, "score": 0.0})
                continue
            scores.append({"name": name,
                           "score": float((between / (k - 1)) / (within / (n - k)))})
        measure = "between-class / within-class variance"

    scores.sort(key=lambda s: -s["score"])
    top = scores[0]["score"] or 1.0
    for s in scores:
        s["share"] = max(2, round(100 * s["score"] / top))
    return {"measure": measure, "scores": scores}
