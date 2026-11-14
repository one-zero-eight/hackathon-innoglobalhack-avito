"""Локальная генерация с тем же промптом и контрактом, что sample_llm."""

from draft_generation.local_llm.client import LocalLLMClient
from draft_generation.sample_llm.pipeline import (
    PROMPT_PATH,
    DraftGenerationPipeline as SamplePipeline,
)


class DraftGenerationPipeline(SamplePipeline):
    """Использовать общий генератор с локальным транспортом."""

    name = "local_llm"

    def __init__(self) -> None:
        self.client = LocalLLMClient()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
