"""Offline preparation. Nothing here may run inside a web request.

Training the classifier takes ~30 s and the out-of-fold pass ~3 minutes, so every
expensive artefact is computed once by `python manage.py prepare_project3` and
cached to disk. The views load the cache and do arithmetic.
"""

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

from . import acquisition, data, deferral, expert

CACHE = Path(__file__).resolve().parent / "cache"

# 50,000 features rather than 300,000: measured at 0.9233 against 0.9241 test
# accuracy -- eight ten-thousandths -- for a pickle of 3.6 MB against 21.9 MB.
# The deployment has 512 MB total, so that trade is worth making.
MAX_FEATURES = 50000
FOLDS = 5
SEED = 0


def _classifier():
    """Logistic regression, not the marginally better LinearSVC.

    LinearSVC scored 0.9287 against 0.9241 on the same features, but has no
    calibrated class probabilities. Tasks 3 and 4 both need to know how confident
    the classifier is, not just what it guessed, so 0.46 points of accuracy is
    the price of admission.
    """
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                        max_features=MAX_FEATURES, strip_accents="unicode"),
        LogisticRegression(max_iter=1500, C=4, random_state=SEED))


def build_baseline(verbose=True):
    """Task 1, plus the out-of-fold predictions everything downstream needs.

    The out-of-fold pass is not optional bookkeeping. Scored on its own training
    data this classifier makes 3,403 mistakes; scored out-of-fold it makes
    11,350. A deferral model trained on the in-sample errors would be learning
    from a third of the evidence and would conclude the classifier almost never
    needs help.
    """
    agnews = data.load()
    train, test = agnews.train, agnews.test

    started = time.time()
    model = _classifier().fit(train.text, train.topic)
    classes = list(model.classes_)

    test_proba = model.predict_proba(test.text)
    test_pred = np.array(classes)[test_proba.argmax(1)]

    if verbose:
        print(f"  fitted in {time.time() - started:.0f}s; "
              f"test accuracy {(test_pred == test.topic).mean():.4f}")

    started = time.time()
    oof_proba = cross_val_predict(_classifier(), train.text, train.topic,
                                  cv=FOLDS, method="predict_proba")
    oof_pred = np.array(classes)[oof_proba.argmax(1)]
    if verbose:
        print(f"  {FOLDS}-fold out-of-fold pass in {time.time() - started:.0f}s; "
              f"accuracy {(oof_pred == train.topic).mean():.4f}")

    artefact = {
        "classes": classes,
        "model": model,
        "test_proba": test_proba.astype(np.float32),
        "test_pred": test_pred,
        "oof_proba": oof_proba.astype(np.float32),
        "oof_pred": oof_pred,
        "test_accuracy": float((test_pred == test.topic).mean()),
        "oof_accuracy": float((oof_pred == train.topic).mean()),
        "in_sample_accuracy": float((model.predict(train.text) == train.topic).mean()),
        "max_features": MAX_FEATURES,
        "folds": FOLDS,
    }
    CACHE.mkdir(exist_ok=True)
    with open(CACHE / "baseline.pkl", "wb") as handle:
        pickle.dump(artefact, handle, protocol=5)
    if verbose:
        size = (CACHE / "baseline.pkl").stat().st_size / 1e6
        print(f"  cached baseline.pkl ({size:.1f} MB)")
    return artefact


def load_baseline():
    with open(CACHE / "baseline.pkl", "rb") as handle:
        return pickle.load(handle)


CLUSTERS = expert.CLUSTERS
COMPONENTS = 100
BUDGETS = [120, 200, 300, 450, 650, 900, 1250, 1700, 2300, 3000]
SEEDS = [0, 1, 2]
TAU_GRID = np.round(np.arange(-0.40, 0.61, 0.02), 3)


def build_world(verbose=True):
    """Everything the two experts and both tasks are built on.

    The k-means regions, the low-dimensional text representation, the expert
    answers, and the competence features. All of it deterministic given the
    seeds, and all of it far too slow for a web request: the SVD alone is a
    minute.
    """
    base = load_baseline()
    agnews = data.load()
    train, test = agnews.train, agnews.test

    vectoriser = base["model"].named_steps["tfidfvectorizer"]
    Xtrain = vectoriser.transform(train.text)
    Xtest = vectoriser.transform(test.text)

    started = time.time()
    svd = TruncatedSVD(n_components=COMPONENTS, random_state=SEED).fit(Xtrain)
    reduced_train, reduced_test = svd.transform(Xtrain), svd.transform(Xtest)
    if verbose:
        print(f"  SVD to {COMPONENTS} components in {time.time() - started:.0f}s")

    started = time.time()
    kmeans = MiniBatchKMeans(n_clusters=CLUSTERS, random_state=SEED,
                             n_init=10, batch_size=4096).fit(Xtrain)
    clusters_train = kmeans.labels_
    clusters_test = kmeans.predict(Xtest)
    if verbose:
        print(f"  {CLUSTERS} regions in {time.time() - started:.0f}s")

    terms = np.array(vectoriser.get_feature_names_out())
    top_terms = {}
    for k in range(CLUSTERS):
        rows = Xtrain[clusters_train == k]
        if rows.shape[0]:
            weights = np.asarray(rows.mean(axis=0)).ravel()
            top_terms[k] = terms[np.argsort(weights)[-8:][::-1]].tolist()

    truth_train = train.topic.to_numpy()
    truth_test = test.topic.to_numpy()
    machine_ok_train = base["oof_pred"] == truth_train
    machine_ok_test = base["test_pred"] == truth_test

    experts = {}
    desks = expert.desk_expert(clusters_train, clusters_test, seed=SEED)
    surface_train = expert.surface_features(train.text)
    surface_test = expert.surface_features(test.text)
    reader = expert.headline_expert(surface_train, surface_test, surface_train)

    for key, competence, label, description in [
        ("desks", desks, "Specialist desks",
         "Reliable inside five of twelve regions of the input space, and little "
         "better than guessing outside them."),
        ("reader", reader, "Headline reader",
         "Competence varies smoothly with how long, how numeric and how "
         "name-heavy the text is, and has nothing to do with its subject."),
    ]:
        answers_train, right_train = expert.answer(
            competence["train"], truth_train, agnews.topics, seed=SEED + 11)
        answers_test, right_test = expert.answer(
            competence["eval"], truth_test, agnews.topics, seed=SEED + 12)
        experts[key] = {
            "label": label,
            "description": description,
            "train_ok": right_train,
            "test_ok": right_test,
            "test_pred": answers_test,
            "accuracy": float(right_test.mean()),
            "specialist": competence.get("specialist"),
            "regions": expert.profile_by_region(
                clusters_test, right_test, machine_ok_test, CLUSTERS),
        }
        if verbose:
            print(f"  expert '{key}': test accuracy {right_test.mean():.4f}")

    world = {
        "features_train": deferral.competence_features(
            reduced_train, base["oof_proba"]).astype(np.float32),
        "features_test": deferral.competence_features(
            reduced_test, base["test_proba"]).astype(np.float32),
        "clusters_train": clusters_train,
        "clusters_test": clusters_test,
        "top_terms": top_terms,
        "machine_ok_train": machine_ok_train,
        "machine_ok_test": machine_ok_test,
        "experts": experts,
        "components": COMPONENTS,
    }
    with open(CACHE / "world.pkl", "wb") as handle:
        pickle.dump(world, handle, protocol=5)
    if verbose:
        print(f"  cached world.pkl "
              f"({(CACHE / 'world.pkl').stat().st_size / 1e6:.1f} MB)")
    return world


def build_experiments(verbose=True):
    """Task 3's threshold sweep and Task 4's query-budget curves."""
    base = load_baseline()
    world = load_world()
    results = {}

    for key, info in world["experts"].items():
        heads = deferral.fit_heads(world["features_train"], world["machine_ok_train"],
                                   info["train_ok"], seed=SEED)
        advantage, rows = deferral.sweep(
            heads, world["features_test"], world["machine_ok_test"],
            info["test_ok"], TAU_GRID)

        references = deferral.reference_policies(
            world["machine_ok_test"], info["test_ok"], base["test_proba"])
        best = max(rows, key=lambda r: r["accuracy"])
        matched = references["by_confidence"](best["deferral_rate"])

        results[key] = {
            "sweep": rows,
            "best": best,
            "advantage_test": advantage.astype(np.float32),
            "never": deferral.evaluate(references["never"], world["machine_ok_test"], info["test_ok"]),
            "always": deferral.evaluate(references["always"], world["machine_ok_test"], info["test_ok"]),
            "oracle": deferral.evaluate(references["oracle"], world["machine_ok_test"], info["test_ok"]),
            "by_confidence": deferral.evaluate(matched, world["machine_ok_test"], info["test_ok"]),
        }
        if verbose:
            print(f"  '{key}' best team accuracy {best['accuracy']:.4f} at tau={best['tau']:+.2f} "
                  f"(rate {best['deferral_rate']:.3f}); oracle {results[key]['oracle']['accuracy']:.4f}")

        curves = {}
        for strategy in acquisition.STRATEGIES:
            runs = [acquisition.run(
                        strategy, world["features_train"], world["clusters_train"],
                        base["oof_proba"], world["machine_ok_train"], info["train_ok"],
                        world["features_test"], world["machine_ok_test"], info["test_ok"],
                        BUDGETS, heads.classifier_head, seed=s, tau=best["tau"])
                    for s in SEEDS]
            merged = []
            for i, budget in enumerate(BUDGETS):
                accs = [r[i]["accuracy"] for r in runs]
                f1s = [r[i]["f1"] for r in runs]
                merged.append({
                    "budget": budget,
                    "accuracy": float(np.mean(accs)),
                    "accuracy_sd": float(np.std(accs)),
                    "f1": float(np.mean(f1s)),
                    "f1_sd": float(np.std(f1s)),
                })
            curves[strategy] = merged
            if verbose:
                last = merged[-1]
                print(f"    {strategy:11} at {BUDGETS[-1]} queries: "
                      f"{last['accuracy']:.4f} (sd {last['accuracy_sd']:.4f})")
        results[key]["curves"] = curves

    with open(CACHE / "experiments.pkl", "wb") as handle:
        pickle.dump({"results": results, "budgets": BUDGETS, "seeds": SEEDS,
                     "tau_grid": TAU_GRID.tolist()}, handle, protocol=5)

    # What the web app actually needs, and nothing else. world.pkl carries the
    # 120,000-row training matrix, which is a hundred megabytes and of no use to
    # a page that only ever displays test-set results.
    app = {
        "features_test": world["features_test"],
        "clusters_test": world["clusters_test"],
        "top_terms": world["top_terms"],
        "machine_ok_test": world["machine_ok_test"],
        "machine_accuracy": float(world["machine_ok_test"].mean()),
        "experts": {k: {kk: vv for kk, vv in info.items() if kk != "train_ok"}
                    for k, info in world["experts"].items()},
        "results": {k: {kk: vv for kk, vv in r.items()} for k, r in results.items()},
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "tau_grid": TAU_GRID.tolist(),
        "components": COMPONENTS,
        "strategies": acquisition.STRATEGIES,
    }
    with open(CACHE / "app.pkl", "wb") as handle:
        pickle.dump(app, handle, protocol=5)
    if verbose:
        for name in ("experiments.pkl", "app.pkl"):
            print(f"  cached {name} ({(CACHE / name).stat().st_size / 1e6:.1f} MB)")
    return results


def load_app():
    """The only artefact the views touch."""
    with open(CACHE / "app.pkl", "rb") as handle:
        return pickle.load(handle)


def load_world():
    with open(CACHE / "world.pkl", "rb") as handle:
        return pickle.load(handle)


def load_experiments():
    with open(CACHE / "experiments.pkl", "rb") as handle:
        return pickle.load(handle)
