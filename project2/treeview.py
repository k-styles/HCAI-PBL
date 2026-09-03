"""Reading a fitted tree back out as sentences.

matplotlib's tree drawing is the conventional way to show this and it is also
the first thing to become unreadable: past a dozen leaves it is a wall of tiny
boxes. Since the whole subject of this project is whether a person can actually
follow the model, the tree is also walked here and turned into nested
conditions in plain words.

One detail that matters for readability: the model is fitted on one-hot
columns, so a split on "island=Biscoe <= 0.5" really means "the island is not
Biscoe". Printing the threshold would be faithful to the arithmetic and useless
to the reader, so categorical splits are put back into words.
"""

from . import data

LEAF = -1


def _condition(penguins, feature_index, threshold, going_left):
    column = penguins.columns[feature_index]
    if "=" in column:
        feature, level = column.split("=", 1)
        # the one-hot column is 0 or 1, so the split is really "is / is not"
        verb = "is not" if going_left else "is"
        return f"{data.PRETTY[feature]} {verb} {level}"
    comparison = "≤" if going_left else ">"
    return f"{data.PRETTY[column]} {comparison} {threshold:.4g}"


def as_rules(estimator, penguins):
    """Nested dicts the template can render as an indented decision list."""
    tree = estimator.tree_
    classes = list(estimator.classes_)

    def node(index, condition):
        counts = tree.value[index][0] * tree.weighted_n_node_samples[index]
        total = counts.sum()
        winner = classes[int(counts.argmax())]
        share = counts.max() / total if total else 0.0
        entry = {
            "condition": condition,
            "samples": int(round(total)),
            "prediction": winner,
            "purity": share,
            "counts": [{"label": c, "n": int(round(n)),
                        "share": (n / total if total else 0.0)}
                       for c, n in zip(classes, counts)],
            "leaf": tree.children_left[index] == LEAF,
            "children": [],
        }
        if not entry["leaf"]:
            for child, left in ((tree.children_left[index], True),
                                (tree.children_right[index], False)):
                entry["children"].append(
                    node(child, _condition(penguins, tree.feature[index],
                                           tree.threshold[index], left)))
        return entry

    return node(0, None)


def flatten(root):
    """Every root-to-leaf path as one readable sentence, longest first.

    This is the form that answers "why did it say Gentoo": a single chain of
    conditions, rather than a diagram the reader has to trace themselves.
    """
    paths = []

    def walk(entry, conditions):
        here = conditions + ([entry["condition"]] if entry["condition"] else [])
        if entry["leaf"]:
            paths.append({"conditions": here, "prediction": entry["prediction"],
                          "samples": entry["samples"], "purity": entry["purity"]})
        else:
            for child in entry["children"]:
                walk(child, here)

    walk(root, [])
    paths.sort(key=lambda p: -p["samples"])
    return paths
