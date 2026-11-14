"""Threshold-only group cross-fitting и shrinkage порогов к безопасному 0.5."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from run import normalize_description
from subcategory_extraction.semantic_encoder_f1.threshold_search import pick_thresholds


def validation_groups(rows: list[dict]) -> np.ndarray:
    """Удержать известные близкие validation-шаблоны вместе при подборе порогов."""
    texts = [normalize_description(row["description"]) for row in rows]
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        """Найти компоненту близких описаний."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    features = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=40_000, dtype=np.float32,
    ).fit_transform(texts)
    neighbors = NearestNeighbors(
        metric="cosine", algorithm="brute", radius=0.15, n_jobs=1,
    ).fit(features).radius_neighbors(features, return_distance=False)
    exact = {}
    for index, text in enumerate(texts):
        previous = exact.setdefault(text, index)
        for other in (previous, *neighbors[index]):
            first, second = find(index), find(int(other))
            if first != second:
                parents[max(first, second)] = min(first, second)
    return np.array([find(index) for index in range(len(rows))])


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict:
    """Посчитать micro-F1 и TP/FP/FN по всем назначениям категорий."""
    tp = int(np.count_nonzero(labels & predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    fn = int(np.count_nonzero(labels & ~predicted))
    denominator = 2 * tp + fp + fn
    return {
        "micro_f1": 2 * tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def select_robust_thresholds(
    probabilities: np.ndarray, labels: np.ndarray, categories: list[int], groups: np.ndarray,
) -> dict:
    """Выбрать сложность порогов по group-crossfit validation, затем fit всей validation.

    Это cross-fitting только ПОРОГОВ, не OOF оценка encoder: checkpoint уже
    выбран на validation. Control/test функция не принимает. Усложнение
    разрешено лишь при приросте pooled crossfit micro-F1 хотя бы на 0.002.
    """
    # Общая проверка формы/диапазонов и точный полный validation fit.
    full = pick_thresholds(probabilities, labels, categories)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if groups.shape != (len(labels),) or len(np.unique(groups)) < 3:
        raise ValueError("Нужны минимум три validation-группы и group id каждой строки")
    strategies = [("fixed", "fixed", 0.0)] + [
        (f"{method}_shrink_{round(strength * 100):03d}", method, strength)
        for method in ("global", "per_category") for strength in (0.25, 0.5, 0.75, 1.0)
    ]
    predictions = {name: np.zeros_like(labels, dtype=bool) for name, _, _ in strategies}
    folds = []
    for number, (training, heldout) in enumerate(GroupKFold(n_splits=3).split(probabilities, groups=groups), start=1):
        fitted = pick_thresholds(probabilities[training], labels[training], categories)
        fold = {"fold": number, "fit_rows": len(training), "heldout_rows": len(heldout), "metrics": {}}
        for name, method, strength in strategies:
            raw = np.array([fitted["candidates"][method]["thresholds"][str(category)] for category in categories])
            boundary = 0.5 + strength * (raw - 0.5)
            predicted = probabilities[heldout] >= boundary
            predictions[name][heldout] = predicted
            fold["metrics"][name] = binary_metrics(labels[heldout], predicted)
        folds.append(fold)
    crossfit = {name: binary_metrics(labels, predicted) for name, predicted in predictions.items()}
    selected = "fixed"
    best = crossfit["fixed"]["micro_f1"]
    minimum_gain = 0.002
    for name, _, _ in strategies[1:]:
        value = crossfit[name]["micro_f1"]
        if value >= crossfit["fixed"]["micro_f1"] + minimum_gain and value > best:
            selected, best = name, value
    _, method, strength = next(strategy for strategy in strategies if strategy[0] == selected)
    raw = full["candidates"][method]["thresholds"]
    thresholds = {str(category): float(0.5 + strength * (raw[str(category)] - 0.5)) for category in categories}
    full_prediction = probabilities >= np.array([thresholds[str(category)] for category in categories])
    return {
        "selected_method": selected, "thresholds": {"101": 0.5, **thresholds},
        "selection_metric": crossfit[selected],
        "validation_fit_metric": binary_metrics(labels, full_prediction),
        "crossfit_candidates": crossfit, "folds": folds,
        "method": "threshold_only_GroupKFold3_shrinkage_towards_0.5",
        "minimum_crossfit_gain": minimum_gain,
        "group_count": len(np.unique(groups)),
        "encoder_outputs_are_oof": False,
        "full_validation_raw_search": full,
        "selection_data": "validation only; no control/public-test scores or labels",
    }
