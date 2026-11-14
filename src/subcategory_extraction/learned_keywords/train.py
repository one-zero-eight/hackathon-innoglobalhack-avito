"""Обучение TF-IDF и бинарных фраз на train InnoGlobalHack из PostgreSQL."""

import hashlib
import json
import logging
import platform
import unicodedata
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

from database import read_records as database_read_records, table_sha256
from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.learned_keywords.pipeline import DEFAULT_ARTIFACT

ROOT = Path(__file__).resolve().parents[3]
# Обучаемся на ручной смысловой разметке train из PostgreSQL
# (преемник прежнего docs/improved_dataset/train.jsonl).
TRAIN_TABLE = "manual_train"
KEYWORDS_PER_CATEGORY = 40
REGULARIZATION = 4
MAX_ITERATIONS = 500
SOURCE_MC_ID = 101


def load_training(records: list[dict] | None = None) -> tuple[list[str], list[set[int]]]:
    """Проверить контракт и убрать точные нормализованные дубликаты train."""
    groups: dict[str, tuple[str, set[int]]] = {}
    if records is None:
        records = database_read_records(TRAIN_TABLE)
    for row in records:
        inp = PipelineInput(description=row["description"], mcId=SOURCE_MC_ID)
        labels = [item["mcId"] for item in row["found_microcategories"]]
        if any(type(cid) is not int or cid not in MICROCATEGORIES or cid == inp.mc_id for cid in labels):
            raise ValueError("Некорректные категории train")
        key = " ".join(unicodedata.normalize("NFKC", inp.description).casefold().replace("ё", "е").split())
        if key in groups and groups[key][1] != set(labels):
            raise ValueError("Конфликт меток дубликатов train")
        groups.setdefault(key, (inp.description, set(labels)))
    return [text for text, _ in groups.values()], [ids for _, ids in groups.values()]


def main() -> None:
    """Сохранить единственную combined-модель без подбора по test и без API."""
    texts, targets = load_training()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    full = vectorizer.fit_transform(texts)
    binary = full.sign()
    categories = list(MICROCATEGORIES)
    shape = (len(categories), full.shape[1])
    weights, keyword_weights = np.zeros(shape), np.zeros(shape)
    intercepts = np.full(len(categories), -40.0)
    keywords: dict[int, list[str]] = {}
    terms = vectorizer.get_feature_names_out()
    for index, cid in enumerate(categories):
        if cid == SOURCE_MC_ID:
            continue
        labels = np.array([cid in ids for ids in targets])
        if labels.all() or not labels.any():
            raise ValueError(f"Для {cid} нужны оба класса в train")
        statistics, _ = chi2(binary, labels)
        positive = np.asarray(binary[labels].mean(axis=0)).ravel()
        negative = np.asarray(binary[~labels].mean(axis=0)).ravel()
        eligible = np.flatnonzero(positive > negative)
        columns = eligible[np.argsort(-statistics[eligible], kind="stable")[:KEYWORDS_PER_CATEGORY]]
        features = hstack([full, binary[:, columns]], format="csr")
        classifier = LogisticRegression(C=REGULARIZATION, max_iter=MAX_ITERATIONS, random_state=44)
        classifier.fit(features, labels)
        weights[index] = classifier.coef_[0, :full.shape[1]]
        keyword_weights[index, columns] = classifier.coef_[0, full.shape[1]:]
        intercepts[index] = classifier.intercept_[0]
        keywords[cid] = terms[columns].tolist()
    artifact = {
        "version": 1, "categories": categories, "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(), "weights": weights.tolist(),
        "keyword_weights": keyword_weights.tolist(), "intercepts": intercepts.tolist(),
    }
    DEFAULT_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        "train": TRAIN_TABLE, "train_sha256": table_sha256(TRAIN_TABLE),
        "rows": len(texts), "source_mc_id": SOURCE_MC_ID, "trained_categories": categories[1:],
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("pipeline.py", "train.py")},
        "model": "combined", "keywords_per_category": KEYWORDS_PER_CATEGORY,
        "C": REGULARIZATION, "max_iter": MAX_ITERATIONS, "threshold": 0.5,
        "selection": "fixed before test; no validation or test tuning; all groups remain in train",
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "artifact_sha256": hashlib.sha256(DEFAULT_ARTIFACT.read_bytes()).hexdigest(), "keywords": keywords,
    }
    DEFAULT_ARTIFACT.with_name("metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Обучено на %d строках; веса: %s", len(texts), DEFAULT_ARTIFACT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
