"""Клиент Selectel Decisions API (вопросы типа noul) поверх httpx.

Протокол: POST ``{"model", "state": {"text": ...}, "questions": {...}}`` ->
``{"answers": {"<ключ вопроса>": {"type": "noul", "noul": <0..1>}}, ...}``.
Модуль входит в пакет ``api`` (запуск: ``uvicorn api.app:app --app-dir src``),
поэтому импорты внутри пакета — пакетные: ``from api.errors import ...``.
"""

import asyncio
import logging
from typing import Any

import httpx

from api.errors import LLMGatewayError

logger = logging.getLogger(__name__)

RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 30.0


class DecisionsClient:
    """Асинхронный клиент decisions-шлюза с ретраями и валидацией ответа.

    Повторяет запрос на HTTP 429 и 5xx с экспоненциальной паузой.
    Успешный ответ обязан быть JSON-объектом
    со словарём ``answers``, набор ключей которого совпадает с вопросами,
    и числовым полем ``noul`` в [0, 1] у каждого ответа; HTML-заглушка или
    любое другое нарушение схемы — LLMGatewayError с обрезком тела.
    Класс не делает ничего при импорте.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = max(retries, 0)

    async def score(
        self, text: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Задать вопросы о тексте и вернуть {ключ вопроса: вероятность noul}.

        Тело запроса: ``{"model", "state": {"text": text}, "questions"}``;
        вопросы — словарь вида ``{"mc104": {"type": "noul", "instructions": ...}}``.
        Сетевые сбои не повторяются и сразу превращаются в LLMGatewayError.
        """
        if not questions:
            raise ValueError("Нужен непустой словарь questions")
        payload: dict[str, Any] = {
            "model": self.model,
            "state": {"text": text},
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempts = self.retries + 1
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(attempts):
                if attempt > 0:
                    delay = min(
                        RETRY_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1),
                        RETRY_BACKOFF_MAX_SECONDS,
                    )
                    logger.warning(
                        "Decisions-шлюз: повтор %d/%d через %.1f с",
                        attempt,
                        self.retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                try:
                    response = await client.post(
                        self.base_url, json=payload, headers=headers
                    )
                except httpx.HTTPError as exc:
                    raise LLMGatewayError(
                        "Сетевой сбой при запросе к Decisions-шлюзу: "
                        f"{type(exc).__name__}"
                    ) from exc
                logger.debug(
                    "Decisions-шлюз: HTTP %s, попытка %d/%d",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
                if 200 <= response.status_code < 300:
                    return self._extract_answers(response, questions)
                error = LLMGatewayError(
                    f"Decisions-шлюз вернул HTTP {response.status_code}",
                    response.text,
                )
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    if attempt + 1 < attempts:
                        logger.warning(
                            "Decisions-шлюз: HTTP %d, осталось повторов %d",
                            response.status_code,
                            attempts - attempt - 1,
                        )
                        continue
                    logger.error(
                        "Decisions-шлюз: ретраи исчерпаны (%d повторов)", self.retries
                    )
                raise error

        # Достижимо только при attempts < 1, а retries >= 0 гарантирует попытку.
        raise LLMGatewayError("Decisions-шлюз: не выполнено ни одной попытки")

    @staticmethod
    def _extract_answers(
        response: httpx.Response, questions: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Проверить схему успешного ответа и вернуть {ключ: noul как float}.

        Не-JSON тело (например, HTML-заглушка шлюза), отсутствие словаря
        ``answers``, расхождение наборов ключей ответов и вопросов и некорректное
        поле ``noul`` — LLMGatewayError с пояснением и обрезком тела.
        """
        body = response.text
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMGatewayError(
                "Ответ Decisions-шлюза не является JSON (например, HTML-заглушка)", body
            ) from exc
        if not isinstance(data, dict):
            raise LLMGatewayError("Ожидался JSON-объект ответа Decisions-шлюза", body)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise LLMGatewayError("В ответе Decisions-шлюза нет словаря answers", body)
        if set(answers) != set(questions):
            raise LLMGatewayError(
                "Набор ключей answers не совпадает с набором вопросов: "
                f"ответы {sorted(answers)}, вопросы {sorted(questions)}",
                body,
            )
        scores: dict[str, float] = {}
        for key in sorted(answers):
            answer = answers[key]
            probability = answer.get("noul") if isinstance(answer, dict) else None
            # type() вместо isinstance отсекает bool: True/False — не вероятности.
            if type(probability) not in (float, int) or not 0 <= probability <= 1:
                raise LLMGatewayError(
                    f"Ответу {key!r} не хватает числа noul в [0, 1]", body
                )
            scores[key] = float(probability)
        return scores
