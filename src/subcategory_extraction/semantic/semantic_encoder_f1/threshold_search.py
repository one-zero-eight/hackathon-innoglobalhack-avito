"""Подбор порогов только по validation-вероятностям с общей целью micro-F1."""

import numpy as np


_MAX_PASSES = 20


def _metrics(tp: int, predicted: int, positives: int) -> dict:
    """Посчитать общие метрики из целочисленных счётчиков."""
    fp = predicted - tp
    fn = positives - tp
    return {
        "micro_f1": 2 * tp / (predicted + positives) if predicted + positives else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / predicted if predicted else 0.0,
        "recall": tp / positives if positives else 0.0,
    }


def _threshold_counts(
    scores: np.ndarray, truth: np.ndarray, original: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Посчитать TP и число ответов для всех различных состояний правила score >= t."""
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    positive_prefix = np.empty(len(scores) + 1, dtype=np.int64)
    positive_prefix[0] = 0
    np.cumsum(truth[order], dtype=np.int64, out=positive_prefix[1:])
    # 1 разрешает пустой ответ, только если максимальный score строго меньше 1.
    # Дополнительные пороги сохраняют исходный вариант и предпочтение к 0.5.
    thresholds = np.unique(np.concatenate((ordered, [0.0, 0.5, 1.0, original])))
    starts = np.searchsorted(ordered, thresholds, side="left")
    return thresholds, positive_prefix[-1] - positive_prefix[starts], len(scores) - starts


def _better(tp: int, predicted: int, old_tp: int, old_predicted: int, positives: int) -> bool:
    """Сравнить micro-F1 точно, без допуска и ошибок округления при равенстве."""
    return tp * max(old_predicted + positives, 1) > old_tp * max(predicted + positives, 1)


def _best_index(
    thresholds: np.ndarray,
    true_positives: np.ndarray,
    predicted: np.ndarray,
    positives: int,
    original: float,
    other_tp: int = 0,
    other_predicted: int = 0,
) -> int:
    """Найти лучший общий micro-F1; при равенстве сохранить исходный порог, затем 0.5."""
    best = 0
    best_tp = other_tp + int(true_positives[0])
    best_predicted = other_predicted + int(predicted[0])
    best_threshold = float(thresholds[0])
    best_rank = (best_threshold != original, best_threshold != 0.5, abs(best_threshold - 0.5), -best_threshold)
    for index in range(1, len(thresholds)):
        tp = other_tp + int(true_positives[index])
        count = other_predicted + int(predicted[index])
        threshold = float(thresholds[index])
        rank = (threshold != original, threshold != 0.5, abs(threshold - 0.5), -threshold)
        if _better(tp, count, best_tp, best_predicted, positives) or (
            not _better(best_tp, best_predicted, tp, count, positives) and rank < best_rank
        ):
            best = index
            best_tp, best_predicted, best_rank = tp, count, rank
    return best


def pick_thresholds(probabilities: np.ndarray, labels: np.ndarray, categories: list[int]) -> dict:
    """Сравнить fixed, точный global и покоординатный поиск по общей validation micro-F1.

    Равные вероятности переключаются вместе, предсказание задаётся через >=.
    При равенстве итогов предпочитается fixed, затем global, затем per_category.
    Классы без положительных примеров сохраняют глобальный порог и не получают
    отдельной настройки. Массивы не изменяются; контрольная выборка не требуется.
    """
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories должен быть непустым списком категорий 102..106")
    if any(type(category) is not int or category not in range(102, 107) for category in categories):
        raise ValueError("Допустимы только целочисленные категории 102..106")
    if len(set(categories)) != len(categories):
        raise ValueError("Категории не должны повторяться")
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(categories):
        raise ValueError("probabilities должен иметь форму N x len(categories)")
    if labels.shape != probabilities.shape or labels.dtype != np.dtype(bool):
        raise ValueError("labels должен быть bool-массивом той же формы, что probabilities")
    if probabilities.shape[0] == 0:
        raise ValueError("Validation-выборка не должна быть пустой")
    if not np.issubdtype(probabilities.dtype, np.number) or np.iscomplexobj(probabilities):
        raise ValueError("Вероятности должны быть вещественными числами")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Вероятности должны быть конечными числами в [0, 1]")

    positives = int(np.count_nonzero(labels))
    fixed_predictions = probabilities >= 0.5
    fixed_tp = int(np.count_nonzero(fixed_predictions & labels))
    fixed_count = int(np.count_nonzero(fixed_predictions))
    fixed_thresholds = {str(category): 0.5 for category in categories}
    candidates = {
        "fixed": {
            "thresholds": fixed_thresholds,
            "metrics": _metrics(fixed_tp, fixed_count, positives),
        }
    }

    thresholds, tp_counts, prediction_counts = _threshold_counts(
        probabilities.ravel(), labels.ravel(), 0.5
    )
    global_index = _best_index(thresholds, tp_counts, prediction_counts, positives, 0.5)
    global_threshold = float(thresholds[global_index])
    global_tp = int(tp_counts[global_index])
    global_count = int(prediction_counts[global_index])
    candidates["global"] = {
        "thresholds": {str(category): global_threshold for category in categories},
        "metrics": _metrics(global_tp, global_count, positives),
    }

    column_positives = np.count_nonzero(labels, axis=0)
    zero_positive_categories = [
        category for column, category in enumerate(categories) if column_positives[column] == 0
    ]
    # Порядок обхода не зависит от перестановки столбцов входного массива.
    active_columns = sorted(
        (column for column in range(len(categories)) if column_positives[column] > 0),
        key=lambda column: categories[column],
    )
    column_options = {
        column: _threshold_counts(probabilities[:, column], labels[:, column], global_threshold)
        for column in active_columns
    }
    current_thresholds = [global_threshold] * len(categories)
    current_predictions = probabilities >= global_threshold
    current_tp = np.count_nonzero(current_predictions & labels, axis=0).tolist()
    current_counts = np.count_nonzero(current_predictions, axis=0).tolist()
    total_tp, total_count = global_tp, global_count
    accepted_updates = 0
    passes = 0
    converged = not active_columns
    for _ in range(_MAX_PASSES if active_columns else 0):
        passes += 1
        changed = False
        for column in active_columns:
            options, option_tp, option_counts = column_options[column]
            other_tp = total_tp - current_tp[column]
            other_count = total_count - current_counts[column]
            index = _best_index(
                options, option_tp, option_counts, positives, current_thresholds[column],
                other_tp, other_count,
            )
            proposed_tp = other_tp + int(option_tp[index])
            proposed_count = other_count + int(option_counts[index])
            if not _better(proposed_tp, proposed_count, total_tp, total_count, positives):
                continue
            current_thresholds[column] = float(options[index])
            current_tp[column] = int(option_tp[index])
            current_counts[column] = int(option_counts[index])
            total_tp, total_count = proposed_tp, proposed_count
            accepted_updates += 1
            changed = True
        if not changed:
            converged = True
            break
    candidates["per_category"] = {
        "thresholds": {
            str(category): current_thresholds[column] for column, category in enumerate(categories)
        },
        "metrics": _metrics(total_tp, total_count, positives),
    }

    selected_method = "fixed"
    selected_tp, selected_count = fixed_tp, fixed_count
    for method, tp, count in (("global", global_tp, global_count), ("per_category", total_tp, total_count)):
        if _better(tp, count, selected_tp, selected_count, positives):
            selected_method = method
            selected_tp, selected_count = tp, count
    return {
        "selected_method": selected_method,
        "thresholds": dict(candidates[selected_method]["thresholds"]),
        "candidates": candidates,
        "search": {
            "objective": "micro_f1",
            "data": "validation_only",
            "comparison": ">=",
            "global": "Точный перебор уникальных score; равные score переключаются атомарно",
            "boundary_thresholds": [0.0, 0.5, 1.0],
            "tie_preference": "Исходный порог, затем 0.5, близость к 0.5, больший порог",
            "method_tie_preference": ["fixed", "global", "per_category"],
            "per_category": {
                "initialization": "global",
                "objective": "Общий micro-F1, не среднее по категориям",
                "acceptance": "Только строгое улучшение, точное сравнение дробей",
                "category_order": [categories[column] for column in active_columns],
                "max_passes": _MAX_PASSES,
                "passes": passes,
                "accepted_updates": accepted_updates,
                "converged": converged,
            },
            "zero_positive_categories": zero_positive_categories,
            "zero_positive_policy": "Без отдельной настройки; сохранён global (в fixed — 0.5)",
            "undefined_metric_value": 0.0,
        },
    }
