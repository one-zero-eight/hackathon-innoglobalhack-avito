"""Сервис 2: проверка выбранных микрокатегорий."""

import logging

from main import BasePipeline

logger = logging.getLogger(__name__)


class CategoryValidationPipeline(BasePipeline):
    """Шаг 2 пайплайна: фильтрация кандидатов из шага 1.

    TODO: реализовать правила проверки — оставляем только микрокатегории из
    таблицы 101–106, исключаем исходную mcId, дедуплицируем, отбрасываем
    мусорные ответы модели.
    """

    name = "category_validation"

    def run(self, candidates: list[int], current_mc_id: int) -> list[int]:
        """Отфильтровать кандидатов и вернуть валидные целевые mcId.

        TODO: заменить заглушку на реальные правила проверки.
        """
        logger.warning("%s.run: заглушка, возвращаем пустой список", self.name)
        return []
