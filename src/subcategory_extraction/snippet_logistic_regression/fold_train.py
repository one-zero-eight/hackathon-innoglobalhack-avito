"""Групповая кросс-валидация регрессии на указанном наборе."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .pipeline import features, identity, stopped_windows
from .train import MICROCATEGORIES, SOURCE_MC_ID, duplicate_groups, learn_anchors, load_training, metrics

logger = logging.getLogger(__name__)


def main() -> None:
    """Обучить пять моделей и сохранить предсказания на отложенных строках."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--max-window", action="store_true")
    args = parser.parse_args()
    fixed = ({int(cid): terms for cid, terms in json.loads(args.dictionary.read_text()).items()}
             if args.dictionary else None)
    texts, targets = load_training(args.dataset)
    categories = [cid for cid in MICROCATEGORIES if cid != SOURCE_MC_ID]
    truth = np.asarray([[cid in ids for cid in categories] for ids in targets])
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.assignments:
        baseline = json.loads(args.assignments.read_text())
        if baseline["dataset_sha256"] != dataset_hash or len(baseline["rows"]) != len(texts):
            raise ValueError("Назначения фолдов не соответствуют набору")
        groups = np.array([row["group"] for row in baseline["rows"]])
        saved = np.array([row["fold"] for row in baseline["rows"]])
        grouping = baseline["grouping"]
        splits = [(np.flatnonzero(saved != fold), np.flatnonzero(saved == fold)) for fold in range(5)]
    else:
        groups, grouping = duplicate_groups(texts)
        strata = truth @ (2 ** np.arange(len(categories)))
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = outer.split(texts, strata, groups)
    methods = ("union", "max") if args.max_window else ("union",)
    predicted = {method: np.zeros_like(truth) for method in methods}
    anchored = np.zeros_like(truth)
    assignments = np.full(len(texts), -1)
    folds = {method: [] for method in methods}
    width, mode = 11, "before"
    for fold, (train, test) in enumerate(splits):
        fit, tune = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=43)
                         .split(train, groups=groups[train]))
        fit, tune = train[fit], train[tune]
        assert not set(groups[train]).intersection(groups[test])
        assert not set(groups[fit]).intersection(groups[tune])
        assert (assignments[test] == -1).all()
        for split in (fit, tune, test):
            if not (truth[split].any(axis=0) & (~truth[split]).any(axis=0)).all():
                raise ValueError(f"Недостаточно примеров в фолде {fold}")
        ranked = fixed if fixed is not None else learn_anchors([texts[i] for i in fit], truth[fit], categories)
        indices = np.r_[fit, tune, test]
        snippets = [stopped_windows(texts[i], ranked, width, mode, fixed_dictionary=fixed is not None)
                    for i in indices]
        vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
        x_fit = vectorizer.fit_transform([features(windows, "pairs") for windows in snippets[:len(fit)]])
        evaluation = snippets[len(fit):]
        x_eval = vectorizer.transform([features(windows, "pairs") for windows in evaluation])
        eval_hits = np.array([bool(windows) for windows in evaluation])
        models = [LogisticRegression(solver="liblinear", max_iter=500, random_state=44)
                  .fit(x_fit, truth[fit, column]) for column in range(len(categories))]
        pooled = np.column_stack([model.predict_proba(x_eval)[:, 1] for model in models])
        scores_by_method = {"union": pooled}
        if args.max_window:
            maxima = np.zeros_like(pooled)
            for start in range(0, len(evaluation), 64):
                owners = []
                window_features = []
                for row, windows in enumerate(evaluation[start:start + 64], start):
                    owners.extend([row] * len(windows))
                    window_features.extend(features([window], "pairs") for window in windows)
                if owners:
                    matrix = vectorizer.transform(window_features)
                    probabilities = np.column_stack([model.predict_proba(matrix)[:, 1] for model in models])
                    np.maximum.at(maxima, owners, probabilities)
            scores_by_method["max"] = maxima
        for method, matrix_scores in scores_by_method.items():
            thresholds = []
            for column, scores in enumerate(matrix_scores.T):
                candidate = np.arange(0.1, 0.91, 0.025)
                tuning = (scores[:len(tune), None] >= candidate) & eval_hits[:len(tune), None]
                labels = truth[tune, column]
                f1 = 2 * (labels[:, None] & tuning).sum(axis=0) / np.maximum(1, labels.sum() + tuning.sum(axis=0))
                threshold = float(candidate[np.argmax(f1)])
                thresholds.append(threshold)
                predicted[method][test, column] = (scores[len(tune):] >= threshold) & eval_hits[len(tune):]
            report = metrics(truth[test], predicted[method][test], categories)
            folds[method].append({"fold": fold, "fit": len(fit), "tune": len(tune), "test": len(test),
                                  "thresholds": thresholds, **report})
            logger.info("Fold %d %s: %d rows, micro-F1 %.4f", fold, method, len(test), report["micro_f1"])
        for i, windows in zip(test, evaluation[len(tune):], strict=True):
            found = {term for _, term in windows}
            anchored[i] = [bool(found.intersection(ranked.get(cid, ()))) for cid in categories]
        assignments[test] = fold
    assert (assignments >= 0).all()
    for method in methods:
        write_report(args, method, dataset_hash, grouping, width, mode, categories, folds[method],
                     truth, predicted[method], anchored, groups, assignments)


def write_report(args, method, dataset_hash, grouping, width, mode, categories, folds,
                 truth, predicted, anchored, groups, assignments) -> None:
    """Сохранить метрики и предсказания для одного способа объединения."""
    missed = truth & ~predicted
    report = {
        "dataset_sha256": dataset_hash,
        "dictionary_sha256": hashlib.sha256(args.dictionary.read_bytes()).hexdigest() if args.dictionary else None,
        "aggregation": method,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")},
        "grouping": grouping, "seeds": [42, 43, 44], "policy": [mode, width, 160 if args.dictionary is None else None],
        "categories": categories, "folds": folds, **metrics(truth, predicted, categories),
        "tp": (truth & predicted).sum(axis=0).tolist(),
        "fp": (~truth & predicted).sum(axis=0).tolist(),
        "fn": missed.sum(axis=0).tolist(),
        "fn_with_same_category_anchor": (missed & anchored).sum(axis=0).tolist(),
        "rows": [{"group": int(group), "fold": int(fold),
                  "predicted": [cid for cid, hit in zip(categories, row, strict=True) if hit]}
                 for group, fold, row in zip(groups, assignments, predicted, strict=True)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output if method == "union" else args.output.with_name(args.output.stem + "-max.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
