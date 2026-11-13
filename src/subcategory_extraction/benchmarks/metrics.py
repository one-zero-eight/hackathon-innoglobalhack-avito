"""Общие метрики для сравнения вариантов выделения категорий."""

from main import VALID_MCIDS


def score(expected: list[set[int]], predicted: list[set[int]]) -> dict:
    """Посчитать micro-F1, баллы и ошибки по категориям без зависимости от модели."""
    if len(expected) != len(predicted):
        raise ValueError("Размеры эталонов и предсказаний не совпадают")
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
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    micro_f1 = 2 * totals["tp"] / denominator if denominator else 0.0
    return {"micro_f1": micro_f1, "points": 5 * micro_f1, **totals, "by_category": counts}
