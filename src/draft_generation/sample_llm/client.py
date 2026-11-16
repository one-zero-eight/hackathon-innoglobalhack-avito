"""Клиент Selectel AI Router для генерации черновиков."""

import json
import logging
import os
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from main import LLMClient, MissingCredentialsError

DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_API_URL = "https://api.selectel.ru/aig/v1/chat/completions"
logger = logging.getLogger(__name__)


class SampleLLMClient(LLMClient):
    """Передать промпт модели и получить текст JSON-ответа."""

    def __init__(self) -> None:
        super().__init__(os.getenv("SAMPLE_LLM_MODEL_ID") or DEFAULT_MODEL)
        self.api_key = os.getenv("SELECTEL_API_KEY")
        self.api_url = os.getenv("SAMPLE_LLM_API_URL") or DEFAULT_API_URL
        self.timeout = float(os.getenv("SAMPLE_LLM_TIMEOUT", "60"))
        self.max_attempts = int(os.getenv("SAMPLE_LLM_MAX_ATTEMPTS", "4"))
        self.retry_delay = float(os.getenv("SAMPLE_LLM_RETRY_DELAY", "10"))
        if self.timeout <= 0:
            raise ValueError("SAMPLE_LLM_TIMEOUT должен быть положительным")
        if self.max_attempts < 1 or self.retry_delay < 0:
            raise ValueError("Нужно SAMPLE_LLM_MAX_ATTEMPTS >= 1 и SAMPLE_LLM_RETRY_DELAY >= 0")

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Запросить JSON, повторяя только временные ошибки HTTP и таймауты."""
        if not self.api_key:
            raise MissingCredentialsError("Укажите SELECTEL_API_KEY в .env")
        request = Request(
            self.api_url,
            data=json.dumps({
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.max_attempts):
            try:
                result = self.request_json(request, self.timeout, urlopen)
                break
            except (HTTPError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt + 1 == self.max_attempts:
                    raise
                logger.warning(
                    "Временная ошибка Selectel (%s), повтор %d/%d через %.0f с",
                    type(exc).__name__, attempt + 2, self.max_attempts, self.retry_delay,
                )
                sleep(self.retry_delay)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Selectel API: отсутствует текст в choices[0].message.content") from exc
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("Ответ Selectel API прерван до завершения")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM API вернул пустой или нетекстовый ответ")
        return content
