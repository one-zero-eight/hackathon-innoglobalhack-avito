"""Обучение локальной части гибрида на train без validation и подбора порога."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from main import MICROCATEGORIES
from run import normalize_description, read_records
from benchmark_support import ROOT, TRAIN as DATASET
from subcategory_extraction.benchmarks.tfidf_quality.run import load_rows
from subcategory_extraction.tfidf.pipeline import DEFAULT_ARTIFACT

logger = logging.getLogger(__name__)
PUBLIC_DATASET = ROOT / "docs/innoglobalhack_avito/public-test-dataset.parquet"
EXCLUSION_SIMILARITY = 0.85
LOCAL_THRESHOLD = 0.5


def exclude_test_components(rows: list[dict], test_path: Path) -> tuple[list[dict], dict]:
    """Исключить близкие к test тексты из обучения, не читая его целевые метки."""
    test_texts = {normalize_description(row["description"]) for row in read_records(test_path)}
    train_texts = {normalize_description(row["inp"].description) for row in rows}
    texts = sorted(train_texts | test_texts)
    indices = {text: index for index, text in enumerate(texts)}
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        """Найти компоненту похожих описаний."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    grouping = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=40_000, dtype=np.float32,
    ).fit_transform(texts)
    neighbors = NearestNeighbors(
        metric="cosine", algorithm="brute", radius=1 - EXCLUSION_SIMILARITY, n_jobs=1,
    ).fit(grouping).radius_neighbors(grouping, return_distance=False)
    for index, related in enumerate(neighbors):
        for other in related:
            if other > index:
                first, second = find(index), find(int(other))
                if first != second:
                    parents[max(first, second)] = min(first, second)
    excluded_components = {find(indices[text]) for text in test_texts}
    kept = [
        row for row in rows
        if find(indices[normalize_description(row["inp"].description)]) not in excluded_components
    ]
    if not kept:
        raise ValueError("После исключения пересечений не осталось обучающих примеров")
    return kept, {
        "method": "connected_components_char_wb_tfidf_cosine",
        "similarity_threshold": EXCLUSION_SIMILARITY,
        "test_labels_used": False,
        "unique_train_before": len(rows),
        "unique_train_after": len(kept),
        "excluded_unique_descriptions": len(rows) - len(kept),
    }


def train(dataset: Path, exclude_dataset: Path, artifact_path: Path, report_path: Path) -> dict:
    """Обучить фиксированную модель и сохранить совместимый JSON-артефакт."""
    rows, exclusion = exclude_test_components(load_rows(dataset), exclude_dataset)
    texts = [row["inp"].description for row in rows]
    categories = list(MICROCATEGORIES)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts)
    weights, intercepts, trained_categories = [], [], []
    for category in categories:
        eligible = [index for index, row in enumerate(rows) if row["inp"].mc_id != category]
        labels = np.array([category in rows[index]["targets"] for index in eligible], dtype=int)
        if not eligible or len(np.unique(labels)) < 2:
            weights.append(np.zeros(features.shape[1]))
            intercepts.append(40.0 if len(labels) and labels[0] else -40.0)
            continue
        classifier = LogisticRegression(C=4, max_iter=500, random_state=42)
        classifier.fit(features[eligible], labels)
        weights.append(classifier.coef_[0])
        intercepts.append(float(classifier.intercept_[0]))
        trained_categories.append(category)
    training_path = artifact_path.parent / "hybrid-training.jsonl"
    if len({dataset.resolve(), exclude_dataset.resolve(), artifact_path.resolve(), report_path.resolve(), training_path.resolve()}) != 5:
        raise ValueError("Пути обучения, оценки и выходных файлов должны различаться")
    artifact = {
        "version": 1,
        "categories": categories,
        "trained_categories": trained_categories,
        "threshold": LOCAL_THRESHOLD,
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "weights": np.asarray(weights).tolist(),
        "intercepts": intercepts,
        "training": {
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "validation_used": False,
            "hyperparameters_selected_on_test": False,
            "exclusion": exclusion,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    with training_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            record = {
                "description": row["inp"].description,
                "found_microcategories": [{"mcId": category} for category in sorted(row["targets"])],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "dataset": str(dataset.resolve()),
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "actual_training_dataset": str(training_path.resolve()),
        "training_rows": len(rows),
        "trained_categories": trained_categories,
        "fixed_threshold": LOCAL_THRESHOLD,
        "validation_used": False,
        "threshold_search_used": False,
        "test_labels_used_for_training": False,
        "exclusion": exclusion,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Обучено %d голов на %d описаниях без validation; артефакт: %s", len(trained_categories), len(rows), artifact_path)
    return report


def main() -> None:
    """Запустить обучение без разбиения на validation/holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--exclude-dataset", type=Path, default=PUBLIC_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=ROOT / "out/jev-hybrid-training.json")
    args = parser.parse_args()
    train(args.dataset, args.exclude_dataset, args.artifact, args.report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
