"""Обучить бинарный логрег формата заказа на размеченной выборке."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from ..snippet_logistic_regression.train import duplicate_groups
from .pipeline import ARTIFACT, THRESHOLD, identity, order_features

CASE_TRAIN = "docs/innoglobalhack_avito/public-train-annotated.jsonl"


def fit(features: list[list[str]], labels: np.ndarray) -> tuple[CountVectorizer, LogisticRegression]:
    """Обучить словарь и логрег на текущей обучающей части."""
    vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
    matrix = vectorizer.fit_transform(features)
    model = LogisticRegression(solver="liblinear", max_iter=500, random_state=44)
    return vectorizer, model.fit(matrix, labels)


def main() -> None:
    """Оценить вне обучающих фолдов и сохранить итоговый артефакт."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    texts = [row["description"] for row in rows]
    labels = np.array([row["group"] == "refusal" for row in rows])
    features = [order_features(text) for text in texts]
    groups, grouping = duplicate_groups(texts)
    internal = np.array([i for i, row in enumerate(rows) if row["source"] == CASE_TRAIN])
    external = np.setdiff1d(np.arange(len(rows)), internal)
    probabilities = np.full(len(rows), np.nan)
    folds = np.full(len(rows), -1)

    def predict(train: np.ndarray, test: np.ndarray, fold: int) -> None:
        assert not set(groups[train]).intersection(groups[test])
        vectorizer, model = fit([features[i] for i in train], labels[train])
        matrix = vectorizer.transform([features[i] for i in test])
        probabilities[test] = model.predict_proba(matrix)[:, 1]
        probabilities[[i for i in test if not features[i]]] = 0.0
        folds[test] = fold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=44)
    for fold, (train, test) in enumerate(splitter.split(internal, labels[internal], groups[internal])):
        predict(internal[train], internal[test], fold)
    if len(external):
        train = internal[~np.isin(groups[internal], groups[external])]
        predict(train, external, -1)
    assert np.isfinite(probabilities).all()

    predicted = probabilities >= THRESHOLD
    digest = hashlib.sha256(raw).hexdigest()
    report = {
        "dataset_sha256": digest,
        "threshold": THRESHOLD,
        "grouping": grouping,
        "false_positives": int((predicted & ~labels).sum()),
        "false_negatives": int((~predicted & labels).sum()),
        "rows": [
            {
                "source_id": row["source_id"], "expected_refusal": bool(labels[i]),
                "probability": float(probabilities[i]), "predicted_refusal": bool(predicted[i]),
                "fold": int(folds[i]), "duplicate_group": int(groups[i]),
                "control_type": row["control_type"], "quote": row["order_format_quote"],
            }
            for i, row in enumerate(rows)
        ],
    }

    vectorizer, model = fit([features[i] for i in internal], labels[internal])
    artifact = {
        "vocabulary": vectorizer.vocabulary_,
        "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "dataset_sha256": digest,
        "training_rows": len(internal),
    }
    with gzip.open(ARTIFACT, "wt", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
