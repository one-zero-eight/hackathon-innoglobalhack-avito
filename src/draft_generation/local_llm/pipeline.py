"""Генерация текста, сохранение исходных эмодзи и один черновик с буллетами."""

import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from main import BasePipeline, Draft, MICROCATEGORIES, PipelineInput, strip_seo_sections
from draft_generation.local_llm.client import LocalLLMClient

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/draft.md"
STYLE_PROMPT_PATH = PROMPT_PATH.with_name("style.md")
GUIDE_PATH = Path(__file__).resolve().parents[3] / "docs/innoglobalhack_avito/microcategories-guide.md"
EMOJI_ATOM = r"[\u2600-\u27BF\U0001F000-\U0001FAFF]\uFE0F?[\U0001F3FB-\U0001F3FF]?"
EMOJI = re.compile(
    rf"[\U0001F1E6-\U0001F1FF]{{2}}|[#*0-9]\uFE0F?\u20E3|{EMOJI_ATOM}(?:\u200D{EMOJI_ATOM})*"
)


class StructuredText(BaseModel):
    """Общее описание, работы одной категории и дополнительные условия."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    intro: str = Field(min_length=1)
    items: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    additional_info: list[Annotated[str, Field(min_length=1)]]


GENERATED_RESPONSE = TypeAdapter(StructuredText | None)


class EmojiStyle(BaseModel):
    """Только эмодзи для существующих фрагментов, без права менять факты."""

    model_config = ConfigDict(extra="forbid", strict=True)

    intro: int
    items: list[int]
    additional_info: list[int]


class DraftGenerationPipeline(BasePipeline):
    """Создать один черновик с буллетами для каждой целевой категории."""

    name = "local_llm"

    def __init__(self) -> None:
        self.client = LocalLLMClient()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.style_prompt = STYLE_PROMPT_PATH.read_text(encoding="utf-8")
        guide = GUIDE_PATH.read_text(encoding="utf-8")
        sections = re.split(r"^## (?=\d{3} )", guide, flags=re.MULTILINE)[1:]
        self.category_rules = {
            int(section[:3]): section.split("Примеры", 1)[0].strip()
            for section in sections
        }
        self.response_schema = json.dumps(StructuredText.model_json_schema(), ensure_ascii=False)

    def _apply_style(self, text: StructuredText, description: str, emojis: list[str]) -> StructuredText:
        """Добавить только исходные эмодзи, не меняя ни одного слова."""
        allowed = ["", *emojis]
        schema = EmojiStyle.model_json_schema()
        schema["properties"]["intro"]["enum"] = list(range(len(allowed)))
        for field in ("items", "additional_info"):
            count = len(getattr(text, field))
            schema["properties"][field].update(minItems=count, maxItems=count)
            schema["properties"][field]["items"]["enum"] = list(range(len(allowed)))
        answer = self.client.complete(self.style_prompt.format(
            payload=json.dumps({
                "description": description, "draft": text.model_dump(),
                "emojis": dict(enumerate(allowed)),
            }, ensure_ascii=False),
            response_schema=json.dumps(schema, ensure_ascii=False),
        ))
        style = EmojiStyle.model_validate_json(answer)
        if (
            len(style.items) != len(text.items)
            or len(style.additional_info) != len(text.additional_info)
            or any(not 0 <= value < len(allowed) for value in [style.intro, *style.items, *style.additional_info])
        ):
            raise ValueError("Оформление должно содержать только исходные эмодзи для существующих пунктов")

        def decorate(value: str, index: int) -> str:
            emoji = allowed[index]
            return f"{emoji} {value}" if emoji and not value.startswith(emoji) else value

        return text.model_copy(update={
            "intro": decorate(text.intro, style.intro),
            "items": [decorate(value, emoji) for value, emoji in zip(text.items, style.items)],
            "additional_info": [
                decorate(value, emoji) for value, emoji in zip(text.additional_info, style.additional_info)
            ],
        })

    def run(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[dict]:
        """Сгенерировать черновики по порядку уникальных категорий."""
        if any(
            type(mc_id) is not int or mc_id not in MICROCATEGORIES or mc_id == inp.mc_id
            for mc_id in target_mc_ids
        ):
            raise ValueError("Целевые категории должны быть допустимыми и отличаться от исходной")
        description = strip_seo_sections(inp.description)
        if not description.strip():
            return []
        emojis = list(dict.fromkeys(EMOJI.findall(description)))
        drafts = []
        for mc_id in dict.fromkeys(target_mc_ids):
            payload = json.dumps({
                "category": MICROCATEGORIES[mc_id],
                "description": description,
            }, ensure_ascii=False)
            answer = self.client.complete(self.prompt.format(
                payload=payload,
                category_rules=self.category_rules[mc_id],
                response_schema=self.response_schema,
            ))
            structured = GENERATED_RESPONSE.validate_json(answer)
            if structured is None:
                continue
            if emojis:
                structured = self._apply_style(structured, description, emojis)
            parts = [structured.intro, "", *(f"• {item}" for item in structured.items)]
            if structured.additional_info:
                parts.extend([
                    "", "Дополнительная информация:",
                    *(f"• {item}" for item in structured.additional_info),
                ])
            draft = Draft(mcId=mc_id, title=structured.title, text="\n".join(parts))
            drafts.append(draft.model_dump(by_alias=True))
        return drafts
