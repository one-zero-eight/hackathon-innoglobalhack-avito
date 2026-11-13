"""Простой генератор: категория и описание → один ответ LLM."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from main import BasePipeline, Draft, MICROCATEGORIES, PipelineInput
from draft_generation.sample_llm.client import SampleLLMClient

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/draft.md"


class GeneratedText(BaseModel):
    """Строгая схема текста ответа LLM, без доверия к выданному mcId."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DraftGenerationPipeline(BasePipeline):
    """Создать по одному черновику на каждую целевую категорию."""

    name = "sample_llm"

    def __init__(self) -> None:
        self.client = SampleLLMClient()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def run(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[dict]:
        """Отправить категорию и описание в LLM, проверить JSON и собрать drafts."""
        if any(
            type(mc_id) is not int or mc_id not in MICROCATEGORIES or mc_id == inp.mc_id
            for mc_id in target_mc_ids
        ):
            raise ValueError("Целевые категории должны быть допустимыми и отличаться от исходной")
        drafts = []
        for mc_id in dict.fromkeys(target_mc_ids):
            payload = json.dumps({
                "category": MICROCATEGORIES[mc_id],
                "description": inp.description,
            }, ensure_ascii=False)
            answer = self.client.complete(
                self.prompt.format(payload=payload),
                response_schema=GeneratedText.model_json_schema(),
            )
            text = GeneratedText.model_validate_json(answer)
            draft = Draft(mcId=mc_id, title=text.title, text=text.text)
            drafts.append(draft.model_dump(by_alias=True))
        return drafts
