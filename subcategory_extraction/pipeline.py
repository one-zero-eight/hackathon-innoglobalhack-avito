"""Сервис 1: выделение микрокатегорий услуг из текста объявления (jev-модель)."""

import logging

from main import BasePipeline, PipelineInput

logger = logging.getLogger(__name__)


class SubcategoryExtractionPipeline(BasePipeline):
    """Шаг 1 пайплайна: jev-модель выделяет услуги и их микрокатегории.

    TODO: реализовать выделение услуг из description и маппинг на
    микрокатегории 101–106. Промпт — в prompts/extract_subcategories.md,
    вызов модели — через main.LLMClient.
    """

    name = "subcategory_extraction"

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть список mcId микрокатегорий описанных услуг.

        TODO: заменить заглушку на вызов jev-модели и парсинг ответа.
        """
        logger.warning("%s.run: заглушка, возвращаем пустой список", self.name)
        return []
