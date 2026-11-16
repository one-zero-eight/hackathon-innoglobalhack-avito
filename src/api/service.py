"""Логика веб-сервиса: сверка справочника БД и вызов ядра пайплайна.

Классификатор и генератор берутся из ``config.json`` и выполняются через
``main.PipelineRuntime``; веб-слой не содержит собственной логики извлечения
подкатегорий и генерации черновиков. Ядро синхронное (urllib), поэтому вызов
уходит в threadpool и не блокирует event loop. Пул БД инжектируется из
``api.app``; модуль не читает окружение сам.
"""

import logging

import asyncpg
from starlette.concurrency import run_in_threadpool

from api.db import get_catalog
from api.schemas import MICROCATEGORIES, PipelineInput, PipelineOutput
from main import LLMCostLedger, PipelineConfig, PipelineRuntime, track_llm_costs

logger = logging.getLogger(__name__)


class DraftsService:
    """Дополнительные черновики: каталог микрокатегорий + полный пайплайн ядра."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        config: PipelineConfig,
        pool: asyncpg.Pool,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._pool = pool

    async def run(self, inp: PipelineInput) -> PipelineOutput:
        """Полный цикл для одного объявления: сверка каталога и вызов ядра."""
        catalog = await get_catalog(self._pool)
        self._check_catalog(catalog)
        # Ядро синхронное: выполняем в рабочем потоке, чтобы не блокировать event loop.
        return await run_in_threadpool(self._run_pipeline, inp)

    def _run_pipeline(self, inp: PipelineInput) -> PipelineOutput:
        """Выполнить ядро с изолированным журналом расходов одного запроса.

        Журнал создаётся на каждый запрос, чтобы usage не накапливался
        в переиспользуемых клиентах модулей. Ошибка учёта (например,
        неподходящий ``LLM_PRICING_FILE``) логируется и не роняет запрос.
        Контекст ставится здесь же, в потоке вызова: иначе ``ContextVar``
        не виден рабочему потоку threadpool. Пустые ``title``/``text``
        черновика — ошибка контракта генератора (502), а не тихий результат.
        """
        try:
            ledger = LLMCostLedger()
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(
                "Учёт расходов API не настроен (%s): запрос выполняется без него",
                type(exc).__name__,
            )
            return self._checked(self._runtime.run(inp, self._config))
        with track_llm_costs(ledger):
            return self._checked(self._runtime.run(inp, self._config))

    @staticmethod
    def _checked(output: PipelineOutput) -> PipelineOutput:
        """Отклонить черновики без заголовка или текста: контракт кейса требует оба поля."""
        if any(not draft.title or not draft.text for draft in output.drafts):
            raise ValueError("Генератор вернул черновик без заголовка или текста")
        return output

    def _check_catalog(self, catalog: dict[int, str]) -> None:
        """Сверить справочник из БД с MICROCATEGORIES из main; расхождение — только в лог."""
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
                + "; ".join(
                    f"{mc_id}: БД «{catalog[mc_id]}» vs main «{MICROCATEGORIES[mc_id]}»"
                    for mc_id in changed
                )
            )
        logger.error(
            "Справочник микрокатегорий в БД расходится с MICROCATEGORIES из main (%s); "
            "работаем с данными из БД",
            ", ".join(details),
        )
