"""Клиент локального Chat Completions API без облачного ключа."""

import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request

from main import LLMClient

DEFAULT_MODEL_ID = str(
    Path(__file__).resolve().parents[3] / "out/draft-model-weights/qwen35-9b"
)


class LocalLLMClient(LLMClient):
    """Передать общий промпт локальной модели и учесть фактический usage."""

    def __init__(self) -> None:
        model_id = os.getenv("LOCAL_LLM_MODEL_ID", DEFAULT_MODEL_ID)
        if not model_id.strip():
            raise ValueError("LOCAL_LLM_MODEL_ID не должен быть пустым")
        super().__init__(model_id)
        self.api_key = None
        self.api_url = os.getenv(
            "LOCAL_LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions"
        )
        if urlparse(self.api_url).hostname not in ("127.0.0.1", "localhost", "::1", "host.docker.internal"):
            raise ValueError("local_llm использует только loopback или host.docker.internal endpoint")
        self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "180"))
        self.temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))
        self.revision = os.getenv("LOCAL_LLM_REVISION")
        self.seed = int(os.getenv("LOCAL_LLM_SEED", "43"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("LOCAL_LLM_TIMEOUT должен быть положительным конечным")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("LOCAL_LLM_TEMPERATURE должен быть от 0 до 2")
        if self.max_tokens < 1:
            raise ValueError("LOCAL_LLM_MAX_TOKENS должен быть положительным")

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Получить исходный ответ без исправлений и скрытых повторов."""
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        result = self.request_json(request, self.timeout)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Локальный сервер не вернул текст choices[0]") from exc
        if choice.get("finish_reason") != "stop":
            raise ValueError("Ответ локальной модели прерван до завершения")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Локальная модель вернула пустой или нетекстовый ответ")
        return content
