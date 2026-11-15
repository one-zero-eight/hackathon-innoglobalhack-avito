"""Обучить фразовый логрег и сравнить его на исходных фолдах и новом контроле."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from ..order_format_logistic_regression.pipeline import (
    THRESHOLD as DOCUMENT_THRESHOLD,
    OrderFormatClassifier as DocumentClassifier,
)
from ..order_format_logistic_regression.train import CASE_TRAIN
from ..snippet_logistic_regression.train import duplicate_groups
from .pipeline import ARTIFACT, THRESHOLD, OrderFormatClassifier, identity, phrase_features, phrases


def read_jsonl(path: Path) -> list[dict]:
    """Прочитать снимок размеченных объявлений."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fit(annotations: list[dict]) -> dict:
    """Построить тот же бинарный словарь n-грамм и обучить логрег классов фраз."""
    examples = [phrase for row in annotations for phrase in row["phrases"]]
    vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
    matrix = vectorizer.fit_transform([phrase_features(example["text"]) for example in examples])
    model = LogisticRegression(max_iter=500, random_state=44)
    model.fit(
        matrix, [example["label"] for example in examples],
        sample_weight=[example.get("weight", 1.0) for example in examples],
    )
    return {
        "vocabulary": vectorizer.vocabulary_,
        "classes": model.classes_.tolist(),
        "weights": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }


def check(model: OrderFormatClassifier, row: dict, baseline: list[int]) -> dict:
    """Сравнить результат снятия категорий с разметкой и старым фильтром."""
    candidates = row["probe_mc_ids"]
    predicted, evidence = model.assess(row["description"], candidates)
    return {
        "source_id": row.get("source_id"),
        "family": row.get("family"),
        "description": row["description"] if row.get("family") else None,
        "candidates": candidates,
        "expected": row["expected_after_filter"],
        "baseline": baseline,
        "predicted": predicted,
        "evidence": evidence,
    }


def with_background(row: dict, annotation: dict) -> dict:
    """Дать модели фоновые фрагменты из объявлений без общего запрета."""
    if row["group"] != "control":
        return annotation
    marked = annotation["phrases"]
    rest = [
        part for part in dict.fromkeys(phrases(row["description"]))
        if not any(example["text"] in part or part in example["text"] for example in marked)
    ]
    if not rest:
        return annotation
    weight = min(0.5, 4 / len(rest))
    return {**annotation, "phrases": marked + [
        {"text": part, "label": "neutral", "weight": weight} for part in rest
    ]}


def summarize(rows: list[dict]) -> dict:
    """Посчитать ошибки на уровне объявления и отдельной категории."""
    summary = {}
    for method in ("baseline", "predicted"):
        over = [set(row["expected"]) - set(row[method]) for row in rows]
        under = [set(row[method]) - set(row["expected"]) for row in rows]
        summary[method] = {
            "exact": sum(row["expected"] == row[method] for row in rows),
            "false_removal_ads": sum(map(bool, over)),
            "missed_removal_ads": sum(map(bool, under)),
            "false_removal_categories": sum(map(len, over)),
            "missed_removal_categories": sum(map(len, under)),
        }
    return summary


def main() -> None:
    """Обучить на разметке Avito, замерить ошибки и сохранить веса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    source = directory / "ads-100.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = read_jsonl(source)
    old = json.loads((directory / "logreg-results.json").read_text(encoding="utf-8"))
    manifest_path = directory / "phrase-labels.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest != manifest["dataset_sha256"] or digest != old["dataset_sha256"]:
        raise ValueError("Разметка фраз и старые прогнозы должны относиться к той же выборке")

    annotations = manifest["rows"]
    additional = manifest["additional"]
    source_ids = {row["source_id"] for row in additional}
    descriptions = {
        row["item_id"]: row["description"]
        for row in read_jsonl(Path(CASE_TRAIN)) if row["item_id"] in source_ids
    }
    texts = [row["description"] for row in rows] + [descriptions[row["source_id"]] for row in additional]
    groups, grouping = duplicate_groups(texts)
    for text, annotation in zip(texts, annotations + additional, strict=True):
        if any(phrase["text"] not in text for phrase in annotation["phrases"]):
            raise ValueError(f"Размеченная фраза отсутствует в объявлении {annotation['source_id']}")
    examples = [with_background(row, annotation) for row, annotation in zip(rows, annotations, strict=True)]

    baseline = old["rows"]
    folds = np.array([row["fold"] for row in baseline])
    internal = [i for i, row in enumerate(rows) if row["source"] == CASE_TRAIN]
    predictions = {}
    for fold in sorted(set(folds)):
        test = np.flatnonzero(folds == fold)
        test_groups = set(groups[test])
        original_test_groups = {baseline[i]["duplicate_group"] for i in test}
        training = [
            examples[i] for i in internal
            if groups[i] not in test_groups and baseline[i]["duplicate_group"] not in original_test_groups
        ]
        training.extend(row for i, row in enumerate(additional, len(rows)) if groups[i] not in test_groups)
        model = OrderFormatClassifier(fit(training))
        for i in test:
            row, prior = rows[i], baseline[i]
            if row["source_id"] != annotations[i]["source_id"] or row["source_id"] != prior["source_id"]:
                raise ValueError("Нарушен порядок строк исходной выборки")
            before = [] if prior["predicted_refusal"] else row["probe_mc_ids"]
            predictions[int(i)] = {**check(model, row, before), "fold": int(fold)}

    training = [examples[i] for i in internal] + additional
    artifact = fit(training)
    model = OrderFormatClassifier(artifact)
    doc_model = DocumentClassifier()
    controls = {}
    for name, filename in (("control", "phrase-control.jsonl"), ("fresh_control", "phrase-blind.jsonl")):
        path = directory / filename
        results = []
        for row in read_jsonl(path):
            before = (
                [] if doc_model.refusal_probability(row["description"]) >= DOCUMENT_THRESHOLD
                else row["probe_mc_ids"]
            )
            results.append(check(model, row, before))
        controls[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "summary": summarize(results),
            "rows": results,
        }

    ordered = [predictions[i] for i in range(len(rows))]
    report = {
        "threshold": THRESHOLD,
        "dataset_sha256": digest,
        "annotations_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "grouping": grouping,
        "training_labels": dict(Counter(phrase["label"] for row in training for phrase in row["phrases"])),
        "original": {"summary": summarize(ordered), "rows": ordered},
        **controls,
    }
    with gzip.open(ARTIFACT, "wt", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False)
    (directory / "phrase-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
