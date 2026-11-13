"""Решение разногласий локальных методов вероятностями Google."""


def resolve_disagreements(
    baseline: set[int], alternative: set[int],
    probabilities: dict[int, float], margin: float | None,
) -> set[int]:
    """Сохранить базовое решение в зоне неопределённости."""
    result = baseline.copy()
    if margin is None:
        return result
    if not 0 <= margin <= 0.5:
        raise ValueError("Порог уверенности должен быть от 0 до 0.5")
    for category in baseline ^ alternative:
        probability = probabilities[category]
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError("Некорректная вероятность арбитра")
        if probability >= 0.5 + margin:
            result.add(category)
        elif probability < 0.5 - margin:
            result.discard(category)
    return result
