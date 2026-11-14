"""Клиент локального jevos без кэша ответов и повторов."""

import json
import math
import os
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from main import LLMClient

DEFAULT_BASE_URL = "http://127.0.0.1:8017"
DEFAULT_MODEL = "jevos-v5"


class LocalJevClient(LLMClient):
    """Проверить локальный сервер и получать вероятности всех вопросов за вызов."""

    def __init__(self) -> None:
        super().__init__(os.getenv("LOCAL_JEV_MODEL", DEFAULT_MODEL))
        base_url = os.getenv("LOCAL_JEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path):
            raise ValueError("LOCAL_JEV_BASE_URL должен указывать на локальный HTTP-сервер")
        self.endpoint = base_url + "/v1/systemone"
        self.timeout = float(os.getenv("LOCAL_JEV_TIMEOUT", "120"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("LOCAL_JEV_TIMEOUT должен быть конечным положительным числом")
        self.api_key = os.getenv("LOCAL_JEV_API_KEY")
        self.opener = build_opener(ProxyHandler({}))
        self.cache_enabled = False
        self.last_cache_hit = False
        self.last_attempts = 0
        self.last_usage: dict | None = None
        with self.opener.open(base_url + "/health", timeout=self.timeout) as response:
            self.health = json.load(response)
        if (not isinstance(self.health, dict) or self.health.get("status") != "ready"
                or self.health.get("model") != self.model_id):
            raise ValueError("Локальный Jev не готов или обслуживает другую модель")

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Отправить настоящий запрос и проверить точное соответствие ответов."""
        self.last_attempts = 0
        self.last_usage = None
        questions = kwargs.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Локальному Jev нужен непустой словарь questions")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=json.dumps({
            "model": self.model_id, "state": prompt, "questions": questions,
        }, ensure_ascii=False, allow_nan=False).encode(), headers=headers, method="POST")
        self.last_attempts = 1
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("model") != self.model_id:
            raise ValueError("Модель в ответе локального Jev не совпадает с запрошенной")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("Набор ответов локального Jev не совпадает с вопросами")
        for name, answer in answers.items():
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if (not isinstance(answer, dict) or answer.get("type") != "noul"
                    or type(probability) not in (int, float)
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError(f"Некорректная вероятность локального Jev: {name}")
        usage = result.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else None
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
