"""Логика веб-сервиса черновиков: классификация услуг и генерация черновиков.

Классификатор Selectel Decisions API и пул БД инжектируются из ``api.app``.
Генератор лениво загружается из общей конфигурации CLI и HTTP API.
"""

import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path

import asyncpg

from api.clients.decisions import DecisionsClient
from main import BasePipeline, load_config, load_pipeline
from api.db import get_catalog
from api.errors import LLMGatewayError
from api.schemas import (
    MICROCATEGORIES,
    ClassifierItem,
    Draft,
    PipelineInput,
    PipelineOutput,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
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


class DraftsService:
    """Дополнительные черновики: классификация mcId -> генерация по целевым категориям."""

    def __init__(
        self,
        classifier: DecisionsClient,
        pool: asyncpg.Pool,
        threshold: float,
    ) -> None:
        self._classifier = classifier
        self._pool = pool
        self._threshold = threshold

        self._generator: BasePipeline | None = None
        self._generation_lock = asyncio.Lock()

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

        async with self._generation_lock:
            try:
                drafts = await asyncio.to_thread(self._generate_drafts, inp, target_mc_ids)
            except Exception as exc:
                logger.error("Ошибка общего генератора: %s", type(exc).__name__)
                raise LLMGatewayError(
                    "Генератор черновиков недоступен или вернул некорректный результат"
                ) from exc
        return PipelineOutput(drafts=drafts)

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

    def _generate_drafts(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[Draft]:
        """Один блокирующий вызов общего генератора для всех целевых категорий."""
        if self._generator is None:
            self._generator = load_pipeline("draft_generation", load_config())
        result = self._generator.run(inp, target_mc_ids)
        drafts = [Draft.model_validate(item) for item in result]
        ids = [draft.mc_id for draft in drafts]
        if (
            len(ids) != len(set(ids))
            or not set(ids).issubset(target_mc_ids)
            or any(not draft.title or not draft.text for draft in drafts)
        ):
            raise ValueError("Генератор нарушил контракт целевых черновиков")
        return drafts
