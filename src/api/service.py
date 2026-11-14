"""Логика веб-сервиса черновиков: классификация услуг и генерация черновиков.

Клиенты LLM (классификатор — Selectel Decisions API с вопросами типа noul,
генератор — Selectel chat/completions) и пул БД инжектируются из ``api.app``;
модуль не читает окружение сам.
"""

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import asyncpg
from pydantic import ValidationError

from api.clients.decisions import DecisionsClient
from api.clients.openai_compat import OpenAIChatClient
from api.db import get_catalog
from api.errors import LLMGatewayError
from api.schemas import (
    MICROCATEGORIES,
    ClassifierItem,
    Draft,
    GeneratedText,
    PipelineInput,
    PipelineOutput,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# Данные промпта идут в user-сообщение: всё от заголовка «## Данные...» до конца.
_DATA_HEADING_RE = re.compile(r"^##\s*Данные.*$", re.MULTILINE)
# Первая строка prompt-файла — HTML-комментарий для разработчиков, модели не нужен.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Определения и примеры услуг каждой микрокатегории для instructions-вопросов
# классификатора: docs/innoglobalhack_avito/microcategories-guide.md и
# CATEGORY_EXAMPLES из subcategory_extraction/google_decision/pipeline.py.
CATEGORY_EXAMPLES: dict[int, str] = {
    101: "комплексный ремонт квартиры, дома или помещения в несколько этапов "
    "и сдачей готового результата: косметический, капитальный, дизайнерский ремонт, "
    "полный цикл отделки, ремонт ванной и санузла",
    102: "штукатурка и выравнивание поверхностей штукатурными составами: "
    "штукатурка по маякам и без, механизированная штукатурка, углы и откосы",
    103: "укладка и облицовка плиткой, керамогранитом и мозаикой: стены и пол, "
    "кухонные фартуки, душевые зоны из плитки, затирка швов, запил под 45 градусов",
    104: "монтаж, замена, подключение и ремонт сантехники и трубопроводов: "
    "разводка водоснабжения и канализации, унитазы, раковины, ванны, инсталляции, "
    "смесители, радиаторы, фильтры, водонагреватели, стиральные и посудомоечные машины",
    105: "работы с электропроводкой, электроточками, освещением и распределительным "
    "оборудованием: замена проводки, розетки и выключатели, щиты и автоматы, "
    "светильники, штробление стен, электрический тёплый пол",
    106: "финишная отделка обоями или краской и подготовка к покрытию: поклейка "
    "и замена обоев, покраска стен и потолков, шпаклёвка, грунтование, шлифовка, "
    "стеклохолст, снятие старых обоев",
}
GENERATOR_FALLBACK_USER = (
    "Составь черновик объявления и верни JSON-объект с ключами title и text."
)


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    """Прочитать шаблон промпта из api/prompts/<name>.md и убрать HTML-комментарии."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return _HTML_COMMENT_RE.sub("", text).strip()


def _render(template: str, values: dict[str, str]) -> str:
    """Подставить {{ключи}} через str.replace: в шаблонах есть литеральные JSON-скобки."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _split_prompt(rendered: str, fallback_user: str) -> tuple[str, str]:
    """Разметить готовый промпт на system-инструкции и user-данные."""
    match = _DATA_HEADING_RE.search(rendered)
    if not match:
        logger.warning("В промпте нет секции «## Данные», отправляю его целиком в system")
        return rendered, fallback_user
    return rendered[: match.start()].strip(), rendered[match.start():].strip()


class DraftsService:
    """Дополнительные черновики: классификация mcId -> генерация по целевым категориям."""

    def __init__(
        self,
        classifier: DecisionsClient,
        generator: OpenAIChatClient,
        pool: asyncpg.Pool,
        threshold: float,
    ) -> None:
        self._classifier = classifier
        self._generator = generator
        self._pool = pool
        self._threshold = threshold

    async def run(self, inp: PipelineInput) -> PipelineOutput:
        """Полный цикл для одного объявления: классификатор -> генераторы."""
        catalog = await get_catalog(self._pool)
        self._check_catalog(catalog)

        target_mc_ids = await self._classify(inp, catalog)
        logger.info(
            "Классификатор: исходная mcId=%s, целевые %s", inp.mc_id, target_mc_ids or "—"
        )
        if not target_mc_ids:
            return PipelineOutput(drafts=[])

        drafts = await asyncio.gather(
            *(self._generate_draft(mc_id, inp, catalog) for mc_id in target_mc_ids)
        )
        return PipelineOutput(drafts=list(drafts))

    def _check_catalog(self, catalog: dict[int, str]) -> None:
        """Сверить справочник из БД с MICROCATEGORIES из main; расхождение — только лог."""
        if catalog == MICROCATEGORIES:
            return
        details: list[str] = []
        missing = sorted(set(MICROCATEGORIES) - set(catalog))
        extra = sorted(set(catalog) - set(MICROCATEGORIES))
        if missing:
            details.append(f"в БД нет mcId {missing}")
        if extra:
            details.append(f"в БД лишние mcId {extra}")
        changed = sorted(
            mc_id
            for mc_id in set(catalog) & set(MICROCATEGORIES)
            if catalog[mc_id] != MICROCATEGORIES[mc_id]
        )
        if changed:
            details.append(
                "названия отличаются: "
                + "; ".join(f"{mc_id}: БД «{catalog[mc_id]}» vs main «{MICROCATEGORIES[mc_id]}»" for mc_id in changed)
            )
        logger.error(
            "Справочник микрокатегорий в БД расходится с MICROCATEGORIES из main (%s); "
            "работаем с данными из БД",
            ", ".join(details),
        )

    async def _classify(self, inp: PipelineInput, catalog: dict[int, str]) -> list[int]:
        """Целевые mcId: p >= порога и не исходная категория объявления.

        Для каждой категории, кроме исходной, ставится вопрос типа noul
        с instructions из prompts/classifier.md; текст объявления уходит
        в state.text. Сбой или невалидный ответ шлюза пробрасываются
        как LLMGatewayError (→ HTTP 502), ретраи делает DecisionsClient.
        """
        instructions = _load_prompt("classifier")
        # Оцениваем контрактные микрокатегории 101..106, кроме исходной:
        # лишние строки справочника в БД не должны уходить в вопросы.
        candidates = [
            mc_id for mc_id in sorted(catalog)
            if mc_id in MICROCATEGORIES and mc_id != inp.mc_id
        ]
        questions = {
            f"mc{mc_id}": {
                "type": "noul",
                "instructions": _render(
                    instructions,
                    {
                        "category": catalog[mc_id],
                        "examples": CATEGORY_EXAMPLES[mc_id],
                    },
                ),
            }
            for mc_id in candidates
        }
        if not questions:
            return []
        scores = await self._classifier.score(inp.description, questions)
        items = [
            ClassifierItem(mc_id=mc_id, p=scores[f"mc{mc_id}"]) for mc_id in candidates
        ]
        return [
            item.mc_id
            for item in items
            if item.p >= self._threshold and item.mc_id != inp.mc_id
        ]

    async def _generate_draft(
        self, mc_id: int, inp: PipelineInput, catalog: dict[int, str]
    ) -> Draft:
        """Черновик для одной целевой категории; mcId берём из нашего списка целей."""
        title = catalog.get(mc_id) or MICROCATEGORIES.get(mc_id) or str(mc_id)
        system, user = _split_prompt(
            _render(
                _load_prompt("generator"),
                {"category_title": title, "description": inp.description},
            ),
            GENERATOR_FALLBACK_USER,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = await self._generate(messages, mc_id)
        if result is None:
            return Draft(mc_id=mc_id)
        return Draft(mc_id=mc_id, title=result.title, text=result.text)

    async def _generate(
        self, messages: list[dict[str, str]], mc_id: int
    ) -> GeneratedText | None:
        """Генерация с повторами: после generation_retries неудач — None (пустой черновик)."""
        attempts = self._generator.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                raw = await self._generator.chat(messages)
                data = json.loads(OpenAIChatClient.extract_json(raw))
                return GeneratedText.model_validate(data)
            except (LLMGatewayError, ValueError, ValidationError) as exc:
                logger.warning(
                    "Генерация для mcId %s не удалась (попытка %d/%d): %s",
                    mc_id,
                    attempt,
                    attempts,
                    exc,
                )
        logger.error("Генерация для mcId %s не удалась после %d попыток", mc_id, attempts)
        return None
