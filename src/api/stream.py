"""Стриминг черновиков одного объявления: SSE-кадры по мере готовности.

Обслуживает только ``POST /api/drafts/stream`` — потоковый вариант одиночного
контракта ``POST /api/drafts``: вход тот же, черновики те же, но результат
дробится на кадры. Классификация отдаётся одним кадром ``categories``, затем
каждый черновик уходит отдельным кадром ``draft`` сразу после своего вызова
генератора — генераторы (sample_llm, local_llm) и так делают по одному
обращению на категорию, поэтому дополнительно вызываем генератор по одной
целевой категории за раз. Экземпляры модулей создаются на запрос через
``load_pipeline`` (как per-row runtime пакета): общий замок
``PipelineRuntime`` одиночного эндпоинта не удерживается на всё время
долгого потока.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from api.bulk import safe_detail, sse_frame
from api.schemas import StreamCategoriesEvent, StreamErrorEvent
from main import (
    VALID_MCIDS,
    BasePipeline,
    Draft,
    LLMCostLedger,
    MissingCredentialsError,
    PipelineConfig,
    PipelineInput,
    PipelineOutputError,
    PipelineUnavailableError,
    ServiceName,
    load_pipeline,
    track_llm_costs,
)

logger = logging.getLogger(__name__)

# Ошибки загрузки модуля — недоступность шага (как PipelineRuntime._load).
LOAD_EXCEPTIONS = (OSError, ImportError, ValueError, RuntimeError)
# Ошибки запуска модуля, означающие недоступность ресурсов или ключей.
RUN_EXCEPTIONS = (MissingCredentialsError, FileNotFoundError, PermissionError, ImportError)


def _load_step(service: ServiceName, config: PipelineConfig) -> BasePipeline:
    """Загрузить модуль шага; ошибка загрузки — недоступность (503-семантика)."""
    try:
        return load_pipeline(service, config)
    except LOAD_EXCEPTIONS as exc:
        raise PipelineUnavailableError(service, getattr(config, service)) from exc


def _extract_targets(inp: PipelineInput, config: PipelineConfig) -> list[int]:
    """Классифицировать объявление и проверить список целевых категорий.

    Проверки повторяют ``PipelineRuntime.run``: только int из 101–106, без
    исходной категории, без дублей.
    """
    extractor = _load_step("subcategory_extraction", config)
    try:
        targets = extractor.run(inp)
    except RUN_EXCEPTIONS as exc:
        raise PipelineUnavailableError(
            "subcategory_extraction", config.subcategory_extraction
        ) from exc
    if (
        not isinstance(targets, list)
        or any(
            type(cid) is not int or cid not in VALID_MCIDS or cid == inp.mc_id
            for cid in targets
        )
        or len(targets) != len(set(targets))
    ):
        raise PipelineOutputError("Некорректные дополнительные микрокатегории")
    return list(targets)


def _generate_one(
    generator: BasePipeline, inp: PipelineInput, mc_id: int, config: PipelineConfig
) -> dict[str, Any]:
    """Сгенерировать один черновик и проверить его контракт.

    Валидация как в ``PipelineRuntime.run`` и ``DraftsService._checked``:
    строгий ``Draft``, категория равна запрошенной, заголовок и текст на месте.
    """
    try:
        drafts = generator.run(inp, [mc_id])
    except RUN_EXCEPTIONS as exc:
        raise PipelineUnavailableError("draft_generation", config.draft_generation) from exc
    if not isinstance(drafts, list) or len(drafts) != 1:
        raise PipelineOutputError("Генератор должен вернуть один черновик на категорию")
    raw = drafts[0]
    try:
        draft = Draft.model_validate(
            raw.model_dump() if isinstance(raw, Draft) else raw, strict=True
        )
    except ValidationError as exc:
        raise PipelineOutputError("Некорректные черновики") from exc
    if draft.mc_id != mc_id or not draft.title or not draft.text:
        raise PipelineOutputError(
            "Черновик должен соответствовать целевой категории и иметь заголовок с текстом"
        )
    return draft.model_dump(by_alias=True)


def _make_ledger() -> LLMCostLedger | None:
    """Журнал расходов на запрос; ошибка настройки не роняет поток."""
    try:
        return LLMCostLedger()
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Учёт расходов API не настроен (%s): поток выполняется без него",
            type(exc).__name__,
        )
        return None


def _call_with_ledger(
    ledger: LLMCostLedger | None, fn: Callable[..., Any], *args: Any
) -> Any:
    """Выполнить шаг рабочего потока с журналом расходов этого запроса.

    Контекст ставится в потоке вызова (как в ``DraftsService``), иначе
    ``ContextVar`` не виден рабочему потоку threadpool.
    """
    if ledger is None:
        return fn(*args)
    with track_llm_costs(ledger):
        return fn(*args)


class DraftsStreamService:
    """Одиночный поток черновиков: ``categories`` → ``draft``… → ``done``/``error``."""

    async def stream(
        self,
        inp: PipelineInput,
        config: PipelineConfig,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать SSE-кадры одного объявления по мере готовности.

        Сбой любого шага завершает поток кадром ``error`` с безопасным
        detail (как в пакете). Пустой список целей отдаётся кадром
        ``categories`` с пустым массивом и сразу ``done``. При отключении
        клиента новые вызовы генератора не запускаются.
        """
        ledger = _make_ledger()

        try:
            targets = await run_in_threadpool(
                _call_with_ledger, ledger, _extract_targets, inp, config
            )
        except Exception as exc:  # noqa: BLE001 — уходим в кадр error с безопасным detail
            logger.warning("Классификация в потоке не удалась: %s", type(exc).__name__)
            yield sse_frame(
                "error", StreamErrorEvent(detail=safe_detail(exc)).model_dump()
            )
            return

        yield sse_frame(
            "categories", StreamCategoriesEvent(mc_ids=targets).model_dump(by_alias=True)
        )
        if not targets:
            yield sse_frame("done", {})
            return

        try:
            generator = await run_in_threadpool(_load_step, "draft_generation", config)
        except Exception as exc:  # noqa: BLE001 — уходим в кадр error с безопасным detail
            logger.warning("Генератор в потоке недоступен: %s", type(exc).__name__)
            yield sse_frame(
                "error", StreamErrorEvent(detail=safe_detail(exc)).model_dump()
            )
            return

        for mc_id in targets:
            if is_disconnected is not None and await is_disconnected():
                logger.info("Клиент отключился: поток черновиков остановлен")
                return
            try:
                draft = await run_in_threadpool(
                    _call_with_ledger, ledger, _generate_one, generator, inp, mc_id, config
                )
            except Exception as exc:  # noqa: BLE001 — уходим в кадр error с безопасным detail
                logger.warning(
                    "Генерация черновика %s в потоке не удалась: %s",
                    mc_id,
                    type(exc).__name__,
                )
                yield sse_frame(
                    "error", StreamErrorEvent(detail=safe_detail(exc)).model_dump()
                )
                return
            yield sse_frame("draft", draft)

        yield sse_frame("done", {})
