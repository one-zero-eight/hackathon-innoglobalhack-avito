"""Обучить TF-IDF на официальном train и оценить групповой holdout."""

import argparse
import hashlib
import json
import logging
import os
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from main import MICROCATEGORIES, PipelineConfig, PipelineInput, load_pipeline
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.tfidf.pipeline import DEFAULT_ARTIFACT

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "docs/innoglobalhack_avito/public-train-annotated.jsonl"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def normalize(text: str) -> str:
    """Объединить точные дубликаты по нормализованному описанию."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def load_rows(path: Path) -> list[dict]:
    """Прочитать официальную разметку, проверив конфликтующие дубликаты."""
    groups: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        inp = PipelineInput(description=row["description"], mcId=101)
        targets = {target["mcId"] for target in row["found_microcategories"]}
        if targets - MICROCATEGORIES.keys() or inp.mc_id in targets:
            raise ValueError("Некорректные целевые категории")
        key = normalize(inp.description)
        if key in groups and groups[key]["targets"] != targets:
            raise ValueError("Конфликт разметки одинаковых описаний")
        groups.setdefault(key, {
            "inp": inp, "targets": targets,
            "group_id": hashlib.sha256(key.encode()).hexdigest(),
        })
    return list(groups.values())


def main() -> None:
    """Обучить, выбрать порог на validation и проверить отдельный holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=ROOT / "out/tfidf-benchmark.json")
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    indices = np.arange(len(rows))
    train, remaining = train_test_split(indices, test_size=0.3, random_state=42)
    dev, holdout = train_test_split(remaining, test_size=0.5, random_state=43)
    texts = [row["inp"].description for row in rows]
    categories = list(MICROCATEGORIES)
    truth = np.array([[cid in row["targets"] for cid in categories] for row in rows])
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True,
    )
    vectorizer.fit([texts[i] for i in train])
    features = vectorizer.transform(texts)
    weights, intercepts = [], []
    for column, cid in enumerate(categories):
        eligible = [i for i in train if rows[i]["inp"].mc_id != cid]
        labels = truth[eligible, column]
        if not eligible or len(np.unique(labels)) < 2:
            weights.append(np.zeros(features.shape[1]))
            intercepts.append(40.0 if len(labels) and labels[0] else -40.0)
            continue
        classifier = LogisticRegression(C=4, max_iter=500)
        classifier.fit(features[eligible], labels)
        weights.append(classifier.coef_[0])
        intercepts.append(float(classifier.intercept_[0]))
    weights_array = np.array(weights)
    logits = np.asarray(features @ weights_array.T) + np.array(intercepts)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))

    def predictions(selection: np.ndarray, threshold: float) -> list[set[int]]:
        """Применить единый порог и исключить исходную категорию."""
        return [
            {cid for cid, p in zip(categories, probabilities[i], strict=True)
             if p >= threshold and cid != rows[i]["inp"].mc_id}
            for i in selection
        ]

    expected_dev = [rows[i]["targets"] for i in dev]
    threshold = max(THRESHOLDS, key=lambda value: score(
        expected_dev, predictions(dev, value),
    )["micro_f1"])
    artifact = {
        "version": 1, "categories": categories, "threshold": threshold,
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(), "weights": weights_array.tolist(),
        "intercepts": intercepts,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    os.environ["TFIDF_ARTIFACT_PATH"] = str(args.artifact.resolve())
    pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
        subcategory_extraction="tfidf", category_validation=None, draft_generation=None,
    ))
    predicted = [set(pipeline.run(rows[i]["inp"])) for i in holdout]
    if predicted != predictions(holdout, threshold):
        raise ValueError("Предсказания JSON-артефакта не совпадают с обученной моделью")
    report = {
        "module": "tfidf", "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "unique_descriptions": len(rows), "threshold": threshold,
        "splits": {name: [rows[i]["group_id"] for i in selection]
                   for name, selection in (("train", train), ("validation", dev), ("holdout", holdout))},
        "validation": score(expected_dev, predictions(dev, threshold)),
        "holdout": score([rows[i]["targets"] for i in holdout], predicted),
        "artifact_predictions_match": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("TF-IDF: threshold %.1f, holdout micro-F1 %.4f",
                 threshold, report["holdout"]["micro_f1"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
