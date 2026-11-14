"""Самостоятельные черновики по фактам исходного объявления."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from main import BasePipeline, Draft, MICROCATEGORIES, PipelineInput, generation_client

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/draft.md"


class GeneratedText(BaseModel):
    """Строгий ответ модели; категорию назначает код, а не модель."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DraftGenerationPipeline(BasePipeline):
    """Сгенерировать один проверенный черновик для каждой целевой категории."""

    name = "llm"

    def __init__(self) -> None:
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.client = generation_client()

    def run(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[dict]:
        """Проверить все цели до API и не возвращать частичный результат."""
        if not isinstance(target_mc_ids, list) or any(
            type(cid) is not int or cid not in MICROCATEGORIES or cid == inp.mc_id
            for cid in target_mc_ids
        ):
            raise ValueError("Нужен список допустимых целевых mcId, отличных от исходной категории")
        if not target_mc_ids:
            return []
        if not inp.description.strip():
            raise ValueError("Нельзя создавать черновик без исходного описания")
        drafts: list[dict[str, Any]] = []
        schema = GeneratedText.model_json_schema()
        for cid in dict.fromkeys(target_mc_ids):
            payload = json.dumps({
                "target_mcId": cid, "category": MICROCATEGORIES[cid],
                "description": inp.description,
            }, ensure_ascii=False)
            raw = self.client.complete(self.prompt.format(payload=payload), response_schema=schema)
            try:
                generated = GeneratedText.model_validate_json(raw)
            except ValidationError:
                raise ValueError(f"Генератор вернул неверный JSON черновика для mcId {cid}") from None
            drafts.append(Draft(mcId=cid, title=generated.title, text=generated.text).model_dump(by_alias=True))
        return drafts
