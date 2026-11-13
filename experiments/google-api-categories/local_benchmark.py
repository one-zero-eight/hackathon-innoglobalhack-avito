"""Оценить локальные модели и их комбинации на групповых разбиениях."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from benchmark import CATEGORIES, DATASET, HERE, ROOT, score

SPLITSHIFT = ROOT.parent / "IGH cases/splitshift/src"
LEGACY_MAP = {101: 101, 102: 104, 103: 105, 105: 103, 106: 106, 107: 106, 108: 102}
IDS = np.array(list(CATEGORIES))


def run() -> None:
    """Обучить на train, выбрать пороги на validation, измерить на test."""
    sys.path.insert(0, str(SPLITSHIFT))
    from splitshift.detector import MicroCategoryDetector
    from splitshift.model import SplitIntentModel

    with DATASET.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    saved = json.loads((HERE / "2026-11-13_decision-1_seed43.json").read_text())
    index = {(row["group_id"], int(row["source_mc_id"])): i for i, row in enumerate(rows)}
    diagnostics = np.array([index[row["group_id"], row["source_mc_id"]] for row in saved["sample"]])
    groups = np.array([row["group_id"] for row in rows])
    pool = np.flatnonzero(~np.isin(groups, groups[diagnostics]))
    fit, holdout = next(GroupShuffleSplit(test_size=0.2, random_state=44).split(pool, groups=groups[pool]))
    train = pool[fit]
    holdout = pool[holdout]
    dev, test = next(GroupShuffleSplit(test_size=0.5, random_state=45).split(holdout, groups=groups[holdout]))
    dev, test = holdout[dev], holdout[test]
    provenance = {}
    with DATASET.with_suffix(".jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            provenance[row["group_id"], row["source_mc_id"]] = {
                origin["dataset"] for origin in row["provenance"]
            }

    texts = np.array([row["description"] for row in rows])
    sources = np.array([int(row["source_mc_id"]) for row in rows])
    labels = [set(json.loads(row["target_mc_ids"])) for row in rows]
    allowed = sources[:, None] != IDS
    truth = np.array([[mcid in target for mcid in IDS] for target in labels])
    assert np.array_equal(np.flatnonzero((truth & ~allowed).any(axis=1)), [])
    assert all(labels[i] == set(row["expected"]) for i, row in zip(diagnostics, saved["sample"], strict=True))

    detector = MicroCategoryDetector()
    candidates = [{LEGACY_MAP.get(mcid) for mcid in detector.detect(text)} for text in texts]
    regex = np.array([[mcid in candidate for mcid in IDS] for candidate in candidates]) & allowed
    # Count legacy mentions consistently for merged sources, including ambiguous 106.
    gate = SplitIntentModel().fit(texts[train], truth[train].any(axis=1), np.zeros(len(train)))
    gated = regex & gate.predict(texts, np.zeros(len(texts)))[:, None]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True)
    vectorizer.fit(texts[train])
    features = vectorizer.transform(texts)
    probabilities = np.zeros(truth.shape)
    for column in range(len(IDS)):
        eligible = train[allowed[train, column]]
        classifier = LogisticRegression(C=4, max_iter=500)
        classifier.fit(features[eligible], truth[eligible, column])
        probabilities[:, column] = classifier.predict_proba(features)[:, 1]

    def measure(selection, predictions):
        return score([labels[i] for i in selection],
                     [set(IDS[row].tolist()) for row in predictions])

    predictions = {"empty": np.zeros_like(truth), "all": allowed, "regex": regex, "regex_gate": gated}
    settings = {}
    for method, prior in (("tfidf", np.zeros_like(truth)), ("tfidf_regex", regex), ("tfidf_gated_regex", gated)):
        choices = [
            (threshold, bonus, (probabilities + bonus * prior >= threshold) & allowed)
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
            for bonus in ((0,) if method == "tfidf" else (0.05, 0.1, 0.2))
        ]
        threshold, bonus, predicted = max(choices, key=lambda choice: measure(dev, choice[2][dev])["micro_f1"])
        predictions[method] = predicted
        settings[method] = {"threshold": threshold, "bonus": bonus}

    test_origins = [provenance[rows[i]["group_id"], int(sources[i])] for i in test]
    slices = {
        "pooled": test,
        "innoglobalhack": test[[origins == {"new_train"} for origins in test_origins]],
        "purple": test[["new_train" not in origins for origins in test_origins]],
    }
    results = {
        method: {name: measure(selection, predicted[selection]) for name, selection in slices.items()}
        for method, predicted in predictions.items()
    }
    google = np.array([[int(mcid) in categories for mcid in IDS]
                       for categories in saved["results"]["guided"]["predicted"]])
    paired = {"decision_1": measure(diagnostics, google)}
    for method in ("regex", "regex_gate", "tfidf", "tfidf_regex", "tfidf_gated_regex"):
        local = predictions[method][diagnostics]
        paired[method] = measure(diagnostics, local)
        paired[f"decision_1_union_{method}"] = measure(diagnostics, google | local)
        paired[f"decision_1_intersection_{method}"] = measure(diagnostics, google & local)

    # A preselected fresh API slice from the same unseen groups as the local test.
    fresh = []
    seen = set()
    for i in np.random.default_rng(46).permutation(test):
        if groups[i] not in seen:
            seen.add(groups[i])
            fresh.append(i)
        if len(fresh) == 32:
            break
    routing = {}
    for name, size, seed in (("validation", 64, 47), ("evaluation", 128, 48)):
        selection = []
        seen = set(groups[fresh])
        for i in np.random.default_rng(seed).permutation(test):
            if groups[i] in seen:
                continue
            seen.add(groups[i])
            selection.append({
                "group_id": str(groups[i]), "source_mc_id": int(sources[i]),
                "expected": sorted(labels[i]),
                "origins": sorted(provenance[rows[i]["group_id"], int(sources[i])]),
                "tfidf_gated_regex": IDS[predictions["tfidf_gated_regex"][i]].tolist(),
                "regex_gate": IDS[gated[i]].tolist(),
            })
            if len(selection) == size:
                break
        assert len(selection) == size
        routing[name] = selection
        test = test[~np.isin(groups[test], [row["group_id"] for row in selection])]
    (HERE / "2026-11-13_routing-sample.json").write_text(json.dumps(routing, indent=2) + "\n")
    report = {
        "split_sizes": {"train": len(train), "validation": len(dev), **{k: len(v) for k, v in slices.items()}},
        "settings": settings,
        "holdout": results,
        "saved_32_diagnostics": paired,
        "fresh_32": [
            {"group_id": str(groups[i]), "source_mc_id": int(sources[i]), "expected": sorted(labels[i]),
             "local": {name: IDS[predicted[i]].tolist() for name, predicted in predictions.items()}}
            for i in fresh
        ],
    }
    (HERE / "2026-11-13_local-comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
