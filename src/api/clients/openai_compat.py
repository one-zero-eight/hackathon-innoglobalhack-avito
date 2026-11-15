"""Клиент OpenAI-совместимого chat/completions-шлюза поверх httpx.

Модуль входит в пакет ``api`` (запуск: ``uvicorn api.app:app --app-dir src``),
поэтому импорты внутри пакета — пакетные: ``from api.errors import ...``.
"""

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from api.errors import LLMGatewayError

logger = logging.getLogger(__name__)

# Пауза перед i-м повтором: RETRY_BACKOFF_BASE_SECONDS * 2**(i-1), но не больше MAX.
RETRY_BACKOFF_BASE_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 30.0
# HTTP-статусы, на которых запрос имеет смысл повторять: 429 и весь диапазон 5xx.
STATUS_TOO_MANY_REQUESTS = 429
STATUS_SERVER_ERROR_MIN = 500
STATUS_SERVER_ERROR_MAX = 599
# Markdown-ограждение ```json ... ``` или ``` ... ``` вокруг ответа модели.
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _is_retryable_status(status_code: int) -> bool:
    """True для HTTP 429 и любого 5xx — статусов, на которых повторяют запрос."""
    return status_code == STATUS_TOO_MANY_REQUESTS or (
        STATUS_SERVER_ERROR_MIN <= status_code <= STATUS_SERVER_ERROR_MAX
    )


class OpenAIChatClient:
    """Асинхронный клиент chat/completions с ретраями и валидацией ответа.

    Ответ шлюза обязан быть JSON-объектом с choices[0].message.content;
    HTML-заглушки, пустые choices и прочее нарушение схемы считаются ошибкой.
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

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Отправить сообщения модели и вернуть текст ответа.

        Повторяет запрос с экспоненциальной паузой на HTTP 429 и 5xx
        (не более ``self.retries`` повторов). При другом не-2xx статусе,
        после исчерпания ретраев, при не-JSON теле или нарушенной структуре
        ответа поднимает LLMGatewayError с пояснением и обрезком тела.
        Сетевые сбои не повторяются и сразу превращаются в LLMGatewayError.
        Ключ ``max_tokens`` добавляется в тело только если он задан.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
                        "LLM-шлюз: повтор %d/%d через %.1f с", attempt, self.retries, delay
                    )
                    await asyncio.sleep(delay)
                try:
                    response = await client.post(
                        self.base_url, json=payload, headers=headers
                    )
                except httpx.HTTPError as exc:
                    raise LLMGatewayError(
                        f"Сетевой сбой при запросе к LLM-шлюзу: {type(exc).__name__}"
                    ) from exc
                logger.debug(
                    "LLM-шлюз: HTTP %s, попытка %d/%d",
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
                if 200 <= response.status_code < 300:
                    return self._extract_content(response)
                error = LLMGatewayError(
                    f"LLM-шлюз вернул HTTP {response.status_code}",
                    response.text,
                )
                if _is_retryable_status(response.status_code):
                    if attempt + 1 < attempts:
                        logger.warning(
                            "LLM-шлюз: HTTP %d, осталось повторов %d",
                            response.status_code,
                            attempts - attempt - 1,
                        )
                        continue
                    logger.error("LLM-шлюз: ретраи исчерпаны (%d повторов)", self.retries)
                raise error

        # Достижимо только при attempts < 1, а retries >= 0 гарантирует попытку.
        raise LLMGatewayError("LLM-шлюз: не выполнено ни одной попытки")

    @staticmethod
    def _extract_content(response: httpx.Response) -> str:
        """Проверить схему успешного ответа и вернуть choices[0].message.content.

        Не-JSON тело (например, HTML-заглушка шлюза) и нарушенная структура —
        LLMGatewayError с пояснением и обрезком тела.
        """
        body = response.text
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMGatewayError(
                "Ответ LLM-шлюза не является JSON (например, HTML-заглушка)", body
            ) from exc
        if not isinstance(data, dict):
            raise LLMGatewayError("Ожидался JSON-объект ответа LLM-шлюза", body)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMGatewayError("В ответе LLM-шлюза нет choices", body)
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            raise LLMGatewayError("У choices[0] нет объекта message", body)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMGatewayError("choices[0].message.content отсутствует или пуст", body)
        return content

    @staticmethod
    def extract_json(raw: str) -> str:
        """Вырезать первый JSON-блок вида {...} или [...] из текста ответа модели.

        Снимает markdown-ограждения (```json ... ```). Блоком считается самая
        ранняя позиция ``{`` или ``[``, с которой текст действительно декодируется
        как единое JSON-значение; хвост после блока отбрасывается. Если JSON-блока
        нет, возвращает исходную строку без изменений — валидация (json.loads
        и схема) остаётся на вызывающем коде.
        """
        text = raw.strip()
        fenced = _CODE_FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                _, end = decoder.raw_decode(text, idx)
            except ValueError:
                continue
            return text[idx:end]
        return raw
