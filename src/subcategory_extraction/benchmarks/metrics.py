"""Общие метрики для сравнения вариантов выделения категорий."""

from main import VALID_MCIDS
from run import CLASSIFICATION_POINTS
from sklearn.metrics import f1_score


def score(expected: list[set[int]], predicted: list[set[int]]) -> dict:
    """Посчитать micro-F1, баллы и ошибки по категориям без зависимости от модели."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Нужны непустые эталоны и предсказания одинакового размера")
    if any(ids - VALID_MCIDS for ids in expected + predicted):
        raise ValueError("Неизвестная категория при расчёте метрик")
    counts = {
        cid: {
            "tp": sum(cid in truth and cid in output for truth, output in zip(expected, predicted, strict=True)),
            "fp": sum(cid not in truth and cid in output for truth, output in zip(expected, predicted, strict=True)),
            "fn": sum(cid in truth and cid not in output for truth, output in zip(expected, predicted, strict=True)),
        }
        for cid in sorted(VALID_MCIDS)
    }
    totals = {name: sum(row[name] for row in counts.values()) for name in ("tp", "fp", "fn")}
    categories = sorted(VALID_MCIDS)
    truth = [[int(cid in ids) for cid in categories] for ids in expected]
    output = [[int(cid in ids) for cid in categories] for ids in predicted]
    micro_f1 = float(f1_score(truth, output, average="micro", zero_division=0))
    for row in counts.values():
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        size = 2 * row["tp"] + row["fp"] + row["fn"]
        row["f1"] = 2 * row["tp"] / size if size else 0.0
    return {
        "micro_f1": micro_f1, "classification_points": CLASSIFICATION_POINTS * micro_f1,
        "macro_f1": float(f1_score(truth, output, average="macro", zero_division=0)),
        **totals, "by_category": counts,
    }
