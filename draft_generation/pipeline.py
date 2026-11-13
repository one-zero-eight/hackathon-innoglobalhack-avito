"""Сервис 3: генерация черновиков объявлений."""

import logging

from main import BasePipeline, PipelineInput

logger = logging.getLogger(__name__)


class DraftGenerationPipeline(BasePipeline):
    """Шаг 3 пайплайна: черновик объявления под каждую целевую микрокатегорию.

    TODO: реализовать генерацию title/text по фактам исходного объявления.
    Ограничения кейса (см. AGENTS.md, раздел 2): ничего не додумываем, не
    ссылаемся на исходное объявление, не утверждаем, что услугу можно
    заказать отдельно, если это не следует из текста.
    """

    name = "draft_generation"

    def run(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[dict]:
        """Сгенерировать по одному черновику на каждый целевой mcId.

        Черновик — dict контракта кейса: {"mcId": int, "title": str, "text": str}.
        Валидация и сериализация — в main.PipelineOutput.

        TODO: заменить заглушку на генерацию через jev-модель.
        """
        logger.warning("%s.run: заглушка, возвращаем пустой список", self.name)
        return []
