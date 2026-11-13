"""Проверка контрактных ограничений без повторной классификации."""

from main import BasePipeline, VALID_MCIDS


class CategoryValidationPipeline(BasePipeline):
    """Отбросить мусор, исходную категорию и дубликаты с сохранением порядка."""

    name = "strict"

    def run(self, candidates: list[int], current_mc_id: int) -> list[int]:
        """Оставить только допустимые дополнительные идентификаторы."""
        if type(current_mc_id) is not int or current_mc_id not in VALID_MCIDS:
            raise ValueError("Неизвестная исходная микрокатегория")
        if not isinstance(candidates, list):
            raise ValueError("Кандидаты должны быть списком")
        return list(dict.fromkeys(
            cid for cid in candidates
            if type(cid) is int and cid in VALID_MCIDS and cid != current_mc_id
        ))
