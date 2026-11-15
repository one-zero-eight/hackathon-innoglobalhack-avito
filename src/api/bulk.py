"""Пакетная обработка объявлений: параллелизм и SSE-стриминг результата.

Обслуживает только `POST /api/drafts/batch`; одиночный контракт
`POST /api/drafts` не меняется. Каждая строка пакета выполняется в своём
независимом экземпляре пайплайна (per-task `PipelineRuntime`): замки и
mutable-state модулей между строками не разделяются, в отличие от общего
runtime одиночного эндпоинта, который запросы сериализует. Параллелизм
ограничен `asyncio.Semaphore`, синхронное ядро выполняется в threadpool.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.error import URLError

from starlette.concurrency import run_in_threadpool

from api.schemas import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    BatchDoneEvent,
    BatchErrorEvent,
    BatchResultEvent,
    BatchStartEvent,
    PipelineInput,
)
from main import (
    LLMCostLedger,
    PipelineConfig,
    PipelineOutput,
    PipelineOutputError,
    PipelineRuntime,
    PipelineUnavailableError,
    track_llm_costs,
)

logger = logging.getLogger(__name__)

# Безопасные detail для событий error: без traceback, ключей и тела ответа провайдера.
DETAIL_UNAVAILABLE = "Модуль пайплайна недоступен: проверьте настройки и ресурсы сервера"
DETAIL_OUTPUT = "Ошибка провайдера или некорректный результат модели"
DETAIL_NETWORK = "Сетевая ошибка при обращении к модели, повторите запрос позже"
DETAIL_INTERNAL = "Внутренняя ошибка обработки объявления"


def sse_frame(event: str, payload: dict[str, Any]) -> str:
    """Собрать один SSE-кадр `event: <type>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def safe_detail(exc: BaseException | None) -> str:
    """Перевести исключение строки в безопасное русское сообщение."""
    if exc is None:
        return DETAIL_INTERNAL
    if isinstance(exc, PipelineUnavailableError):
        return DETAIL_UNAVAILABLE
    if isinstance(exc, (PipelineOutputError, ValueError)):
        return DETAIL_OUTPUT
    if isinstance(exc, (URLError, TimeoutError, ConnectionError)):
        return DETAIL_NETWORK
    return DETAIL_INTERNAL


def process_one(inp: PipelineInput, config: PipelineConfig) -> PipelineOutput:
    """Обработать одно объявление независимым пайплайном (рабочий поток).

    На каждую строку создаётся свой `PipelineRuntime`: тот, в свою очередь,
    при первом вызове загружает свежие экземпляры обоих шагов через
    `load_pipeline`, поэтому mutable-state модулей не смешивается между
    строками пакета. Журнал расходов изолирован на строку — тот же приём,
    что в одиночном `DraftsService`.
    """
    try:
        ledger = LLMCostLedger()
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Учёт расходов API не настроен (%s): строка выполняется без него",
            type(exc).__name__,
        )
        return PipelineRuntime().run(inp, config)
    with track_llm_costs(ledger):
        return PipelineRuntime().run(inp, config)


class BulkService:
    """Пакетная обработка: семафор, независимые пайплайны и SSE по готовности."""

    @staticmethod
    def clamp_concurrency(concurrency: int | None) -> int:
        """Привести параллелизм к диапазону [1, MAX_CONCURRENCY]; None — default."""
        if concurrency is None:
            return DEFAULT_CONCURRENCY
        return max(1, min(concurrency, MAX_CONCURRENCY))

    @staticmethod
    async def _run_row(
        index: int,
        inp: PipelineInput,
        config: PipelineConfig,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, PipelineOutput | None, BaseException | None]:
        """Выполнить строку под семафором; исключение возвращаем в кортеже.

        Строка никогда не бросает наверх: сбой одной позиции не должен
        ронять поток остальных. `CancelledError` (отключение клиента)
        проходит насквозь и не считается ошибкой строки.
        """
        async with semaphore:
            try:
                output = await run_in_threadpool(process_one, inp, config)
            except Exception as exc:  # noqa: BLE001 — per-row ошибка уходит в событие error
                return index, None, exc
            return index, output, None

    async def stream(
        self,
        ads: list[PipelineInput],
        concurrency: int | None,
        config: PipelineConfig,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        """Отдавать SSE-кадры пакета по мере готовности строк.

        Сначала `start`, затем `result`/`error` в порядке завершения задач
        (порядок может отличаться от входного — клиент ориентируется по
        `index`), в конце `done`. Сбой строки даёт событие `error` и не
        прерывает поток. При отключении клиента новые строки не запускаются.
        """
        limit = self.clamp_concurrency(concurrency)
        total = len(ads)
        yield sse_frame("start", BatchStartEvent(total=total, concurrency=limit).model_dump())

        semaphore = asyncio.Semaphore(limit)
        pending: set[asyncio.Task[tuple[int, PipelineOutput | None, BaseException | None]]] = {
            asyncio.create_task(self._run_row(index, inp, config, semaphore))
            for index, inp in enumerate(ads)
        }
        processed = succeeded = failed = 0
        try:
            while pending:
                if is_disconnected is not None and await is_disconnected():
                    logger.info(
                        "Клиент отключился: прекращаем пакет (%s из %s строк)",
                        processed,
                        total,
                    )
                    break
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    index, output, error = task.result()
                    processed += 1
                    if error is None and output is not None:
                        succeeded += 1
                        event = BatchResultEvent(
                            index=index,
                            mc_id=ads[index].mc_id,
                            drafts=output.drafts,
                        )
                        yield sse_frame("result", event.model_dump(by_alias=True))
                    else:
                        failed += 1
                        logger.warning(
                            "Пакетная строка %s завершилась ошибкой: %s",
                            index,
                            type(error).__name__ if error is not None else "без исключения",
                        )
                        event = BatchErrorEvent(index=index, detail=safe_detail(error))
                        yield sse_frame("error", event.model_dump())
        except Exception:  # noqa: BLE001 — поток закрываем кадром done, причину пишем в лог
            logger.exception("Пакетная обработка прервана непредвиденной ошибкой")
        finally:
            for task in pending:
                task.cancel()
        yield sse_frame(
            "done",
            BatchDoneEvent(
                processed=processed, succeeded=succeeded, failed=failed
            ).model_dump(),
        )
