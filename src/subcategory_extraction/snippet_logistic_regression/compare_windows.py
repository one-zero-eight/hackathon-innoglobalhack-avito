"""Сравнить словарь и агрегацию окон на зафиксированных fit и tune."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from run import prepare_rows
from .pipeline import features, identity, stopped_windows
from .train import SOURCE_MC_ID, MICROCATEGORIES, learn_anchors


def measure(truth, predicted, anchors, weights) -> dict:
    """Посчитать ошибки и micro-F1 с исходными и групповыми весами."""
    tp = truth & predicted
    fp = ~truth & predicted
    fn = truth & ~predicted

    def f1(mask, row_weights):
        correct = (row_weights * tp[mask].sum(axis=1)).sum()
        total = (row_weights * (truth[mask].sum(axis=1) + predicted[mask].sum(axis=1))).sum()
        return float(2 * correct / max(total, 1e-12))

    all_rows = np.ones(len(truth), dtype=bool)
    sparse = np.isin(truth.sum(axis=1), [1, 2])
    empty = ~truth.any(axis=1)
    return {
        "micro_f1": f1(all_rows, 1),
        "group_weighted_micro_f1": f1(all_rows, weights),
        "one_or_two_labels_micro_f1": f1(sparse, 1),
        "empty_rows_with_fp": int(predicted[empty].any(axis=1).sum()),
        "empty_rows": int(empty.sum()),
        "tp": tp.sum(axis=0).tolist(),
        "fp": fp.sum(axis=0).tolist(),
        "fn": fn.sum(axis=0).tolist(),
        "fp_without_own_anchor": (fp & ~anchors).sum(axis=0).tolist(),
        "fn_without_own_anchor": (fn & ~anchors).sum(axis=0).tolist(),
    }


def main() -> None:
    """Обучить одну регрессию и сравнить два способа агрегации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.split / "manifest.json").read_text())
    datasets = {}
    for name in ("fit", "tune", "assignments"):
        path = args.split / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["output_sha256"][path.name]:
            raise ValueError(f"Изменился файл {path}")
        with path.open(encoding="utf-8") as file:
            datasets[name] = [json.loads(line) for line in file]

    categories = sorted(set(MICROCATEGORIES) - {SOURCE_MC_ID})
    fit, tune = (prepare_rows(datasets[name], SOURCE_MC_ID) for name in ("fit", "tune"))
    fit_texts = [row.inp.description for row in fit]
    fit_truth, truth = (np.array([[cid in row.expected for cid in categories] for row in rows])
                        for rows in (fit, tune))
    ranked = learn_anchors(fit_texts, fit_truth, categories)
    terms = [set(ranked[cid]) for cid in categories]

    def windows(text: str) -> list:
        return stopped_windows(text, ranked, 11, "before")

    vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
    matrix = vectorizer.fit_transform(features(windows(text), "pairs") for text in fit_texts)
    models = [LogisticRegression(solver="liblinear", max_iter=500, random_state=44).fit(matrix, labels)
              for labels in fit_truth.T]
    coefficients = np.stack([model.coef_[0] for model in models])
    intercepts = np.array([model.intercept_[0] for model in models])
    scores = {name: np.zeros_like(truth, dtype=float) for name in ("all_windows", "own_windows")}
    anchors = np.zeros_like(truth)
    source_rows = [row for row in datasets["assignments"] if row["split"] == "tune"]
    counts = Counter(row["group"] for row in source_rows)
    weights = np.array([1 / counts[row["group"]] for row in source_rows])
    assert len(source_rows) == len(tune)
    witnesses = []
    for i, row in enumerate(tune):
        snippets = windows(row.inp.description)
        witness = {}
        if snippets:
            ownership = np.array([[anchor in keys for keys in terms] for _, anchor in snippets])
            anchors[i] = ownership.any(axis=0)
            x = vectorizer.transform([features([snippet], "pairs") for snippet in snippets])
            probabilities = expit(x @ coefficients.T + intercepts)
            for name, values in (("all_windows", probabilities),
                                 ("own_windows", np.where(ownership, probabilities, 0))):
                scores[name][i] = values.max(axis=0)
                witness[name] = [snippets[j] if values[j, c] > 0 else None
                                 for c, j in enumerate(values.argmax(axis=0))]
        witnesses.append(witness)

    result = {"dictionary": measure(truth, anchors, anchors, weights)}
    predictions = {"dictionary": anchors}
    grid = np.arange(0.10, 0.91, 0.025)
    for name, values in scores.items():
        candidates = values[:, :, None] >= grid
        f1 = 2 * (truth[:, :, None] & candidates).sum(axis=0) / np.maximum(
            1, candidates.sum(axis=0) + truth.sum(axis=0)[:, None]
        )
        thresholds = grid[f1.argmax(axis=1)]
        predicted = values >= thresholds
        predictions[name] = predicted
        result[name] = {"thresholds": thresholds.tolist(), **measure(truth, predicted, anchors, weights)}

    data = {
        "fit": len(fit), "tune": len(tune), "categories": categories, "ranked": ranked,
        "input_sha256": {f"{name}.jsonl": manifest["output_sha256"][f"{name}.jsonl"]
                         for name in datasets},
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(__file__).with_name("train.py"),
                                     Path(__file__).with_name("pipeline.py"))},
        "results": result,
        "rows": [{**source, "truth": truth[i].tolist(), "own_anchor": anchors[i].tolist(),
                  "predictions": {name: values[i].tolist() for name, values in predictions.items()},
                  "top_windows": witnesses[i]} for i, source in enumerate(source_rows)],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    logging.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
