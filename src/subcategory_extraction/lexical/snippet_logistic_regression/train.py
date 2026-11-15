"""Обучение и сравнение логистических регрессий на InnoGlobalHack."""

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from main import MICROCATEGORIES
from run import normalize_description
from subcategory_extraction.lexical.learned_keywords.train import SOURCE_MC_ID, load_training as load_labeled_records
from subcategory_extraction.lexical.snippet_logistic_regression.pipeline import ARTIFACT, STOP_MODES, ad_features, identity

logger = logging.getLogger(__name__)


# Снимок, на котором обучен поставляемый артефакт.
TRAIN = Path(__file__).resolve().parents[4] / "docs/improved_dataset/train.jsonl"


def load_training(path: Path = TRAIN) -> tuple[list[str], list[set[int]]]:
    """Прочитать зафиксированный снимок для воспроизведения эксперимента."""
    with path.open(encoding="utf-8") as file:
        return load_labeled_records([json.loads(line) for line in file])


def duplicate_groups(texts: list[str]) -> tuple[np.ndarray, dict]:
    """Сгруппировать близкие описания по косинусной близости."""
    similarity = 0.85
    normalized = [normalize_description(text) for text in texts]
    unique, inverse = np.unique(normalized, return_inverse=True)
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=40_000, dtype=np.float32,
    ).fit_transform(unique)
    graph = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix).radius_neighbors_graph(
        matrix, radius=1 - similarity,
    )
    count, indices = connected_components(graph, directed=False)
    return indices[inverse], {"similarity_threshold": similarity, "groups": count}


def split_rows(texts: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    """Разделить группы между обучением и двумя проверками."""
    groups, grouping = duplicate_groups(texts)
    remaining, control = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                              .split(texts, groups=groups))
    fit, tune = next(GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=43)
                     .split(remaining, groups=groups[remaining]))
    return {"train": remaining[fit], "validation": remaining[tune], "control": control}, {
        **grouping, "seeds": [42, 43],
    }


def learn_anchors(texts: list[str], labels: np.ndarray, categories: list[int]) -> dict[int, list[str]]:
    """Выбрать положительно связанные униграммы и биграммы на train."""
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    ranked = {}
    for cid, target in zip(categories, labels.T, strict=True):
        statistics, _ = chi2(matrix, target)
        positive = np.asarray(matrix[target].mean(axis=0)).ravel()
        negative = np.asarray(matrix[~target].mean(axis=0)).ravel()
        eligible = np.flatnonzero(positive > negative)
        top = eligible[np.argsort(-statistics[eligible], kind="stable")]
        ranked[cid] = terms[top[:160]].tolist()
    return ranked


def metrics(truth: np.ndarray, predicted: np.ndarray, categories: list[int]) -> dict:
    """Посчитать micro-F1 и precision/recall для каждой категории."""
    _, _, micro, _ = precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(truth, predicted, zero_division=0)
    return {"micro_f1": micro, "per_category": {
        cid: {"precision": p, "recall": r, "support": int(n)}
        for cid, p, r, n in zip(categories, precision, recall, support, strict=True)
    }}


def main() -> None:
    """Выбрать вариант на validation и сохранить веса для CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--pair", type=int, nargs=2, metavar=("WIDTH", "QUOTA"))
    parser.add_argument("--stops", action="store_true")
    args = parser.parse_args()
    if args.stops and not args.pair:
        parser.error("--stops требует --pair")
    matrix_dir = args.matrix_dir
    if args.pair and (matrix_dir is None or args.pair[0] < 3 or args.pair[1] < 1):
        parser.error("--pair требует --matrix-dir, ширину от 3 и число ключей от 1")
    if matrix_dir is not None:
        matrix_dir.mkdir(parents=True, exist_ok=True)
    texts, targets = load_training()
    splits, grouping = split_rows(texts)
    train, validation, control = (splits[name] for name in ("train", "validation", "control"))
    categories = [cid for cid in MICROCATEGORIES if cid != SOURCE_MC_ID]
    labels = np.array([[cid in ids for cid in categories] for ids in targets])
    if np.any(labels[train].all(axis=0) | ~labels[train].any(axis=0)):
        raise ValueError("В train нужны оба класса для каждой категории")
    ranking = learn_anchors([texts[i] for i in train], labels[train], categories)
    policies = [
        ("pairs", 7, 40),
        *(("pairs", width, 40) for width in (3, 5, 9, 11)),
        *(("pairs", 7, quota) for quota in (20, 80, 160)),
        ("priority", 7, 40), ("split", 7, 40), ("category", 7, 40),
    ]
    if matrix_dir is not None:
        policies = [("pairs", width, quota) for width in (7, 11) for quota in (40, 160)]
    if args.pair:
        width, quota = args.pair
        policies = [(mode, width, quota) for mode in (STOP_MODES if args.stops else ("pairs",))]
    results = []
    best = None
    best_f1 = -1.0
    for mode, width, quota in policies:
        ranked = {cid: terms[:quota] for cid, terms in ranking.items()}
        anchors = {term for terms in ranked.values() for term in terms}
        rows, hits = zip(*(ad_features(text, anchors, ranked, width, mode) for text in texts), strict=True)
        hits = np.asarray(hits)
        vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
        x_train = vectorizer.fit_transform([rows[i] for i in train])
        x_eval = vectorizer.transform([rows[i] for i in np.r_[validation, control]])
        prefixes = np.array([term.split(":", 1)[0] for term in vectorizer.get_feature_names_out()])
        models = []
        coefficients = np.zeros((len(categories), x_train.shape[1]))
        scores = []
        for i, cid in enumerate(categories):
            columns = np.isin(prefixes, ["w", "c", "a", str(cid)])
            model = LogisticRegression(solver="liblinear", max_iter=500, random_state=44)
            model.fit(x_train[:, columns], labels[train, i])
            models.append(model)
            coefficients[i, columns] = model.coef_[0]
            scores.append(model.predict_proba(x_eval[:, columns])[:, 1])
        probabilities = np.column_stack(scores)
        valid_scores = probabilities[:len(validation)]
        thresholds = []
        for column in range(len(categories)):
            truth = labels[validation, column]
            candidate = np.arange(0.1, 0.91, 0.025)
            prediction = (valid_scores[:, column, None] >= candidate) & hits[validation, None]
            f1 = 2 * (truth[:, None] & prediction).sum(axis=0) / np.maximum(1, truth.sum() + prediction.sum(axis=0))
            thresholds.append(float(candidate[np.argmax(f1)]))
        prediction = (probabilities >= thresholds) & hits[np.r_[validation, control], None]
        valid = metrics(labels[validation], prediction[:len(validation)], categories)
        measured = {
            "policy": [mode, width, quota],
            "anchors_total": len(anchors),
            "anchors_per_category": {cid: len(terms) for cid, terms in ranked.items()}, "validation": valid,
            "control": metrics(labels[control], prediction[len(validation):], categories),
            "validation_no_hits": int((~hits[validation]).sum()),
            "validation_labels_on_no_hits": int(labels[validation][~hits[validation]].sum()),
        }
        results.append(measured)
        logger.info("%s: validation %.4f, control %.4f", measured["policy"],
                    valid["micro_f1"], measured["control"]["micro_f1"])
        cutoff = np.log(np.array(thresholds) / (1 - np.array(thresholds))) - [m.intercept_[0] for m in models]
        model_data = {
            "policy": [mode, width, 0], "categories": categories, "anchors": sorted(anchors),
            "ranked": ranked, "quota": quota,
            "vocabulary": vectorizer.vocabulary_, "weights": coefficients.tolist(),
            "cutoff": cutoff.tolist(),
        }
        if matrix_dir is not None:
            suffix = f"-{mode}" if args.stops else ""
            path = matrix_dir / f"{width}x{quota}{suffix}.json.gz"
            path.write_bytes(gzip.compress(json.dumps(model_data, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
            continue
        if valid["micro_f1"] > best_f1:
            best_f1 = valid["micro_f1"]
            best = model_data
    if matrix_dir is not None:
        return
    assert best is not None
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_bytes(gzip.compress(json.dumps(best, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    report = {
        "train_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256(ARTIFACT.parent.parent.joinpath(name).read_bytes()).hexdigest()
                        for name in ("pipeline.py", "train.py")},
        "model_sha256": hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        "grouping": grouping, "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "selected_policy": [*best["policy"][:2], best["quota"]], "comparisons": results,
    }
    ARTIFACT.with_name("metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
