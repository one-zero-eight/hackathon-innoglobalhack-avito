"""Применить существующие regex-правила через интерфейс классификатора."""

from dataset_annotation.annotate_dataset import propose
from main import BasePipeline, PipelineInput


class SubcategoryExtractionPipeline(BasePipeline):
    """Выделить категории без сохранённых ответов и исключений отдельных строк."""

    name = "regex"

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть дополнительные категории по неизменённым общим правилам."""
        return [cid for cid in propose(inp.description)["labels"] if cid != inp.mc_id]
