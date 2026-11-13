"""Клиент Google Decision и ограничение входных токенов."""

import json
import os
from collections import deque
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from main import LLMClient

DEFAULT_MODEL = "gemini-decision-1"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"


class TokenBudget:
    """Ограничить входные токены в скользящем окне."""

    limit = 45_000

    def __init__(self) -> None:
        self.last_actual: int | None = None
        self.requests: deque[tuple[float, int]] = deque()

    def reserve(self, estimate: int) -> None:
        """Зарезервировать бюджет до отправки запроса."""
        if estimate > self.limit:
            raise ValueError("Один запрос превышает бюджет входных токенов")
        while True:
            now = monotonic()
            while self.requests and now - self.requests[0][0] >= 60:
                self.requests.popleft()
            if sum(tokens for _, tokens in self.requests) + estimate <= self.limit:
                self.requests.append((now, estimate))
                return
            sleep(max(0, 60 - (now - self.requests[0][0])))

    def reconcile(self, actual: int) -> None:
        """Уточнить расход по ответу API."""
        started, _ = self.requests.pop()
        self.requests.append((started, actual))


class GoogleDecisionClient(LLMClient):
    """Получить структурированный JSON от Google Decision."""

    def __init__(self) -> None:
        super().__init__(os.getenv("LLM_MODEL_ID") or DEFAULT_MODEL)
        self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        self.endpoint = os.getenv("GOOGLE_DECISION_ENDPOINT", DEFAULT_ENDPOINT)
        self.budget = TokenBudget()

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Отправить описание и вопросы; вернуть JSON без интерпретации."""
        if not self.api_key:
            raise ValueError("Укажите LLM_API_KEY или GOOGLE_API_KEY в .env")
        questions = kwargs.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Нужен непустой словарь questions")
        payload = json.dumps({
            "model": self.model_id,
            "input": [{"type": "text", "text": prompt}],
            "questions": questions,
        }, ensure_ascii=False).encode()
        request = Request(
            self.endpoint, data=payload,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )
        for attempt in range(3):
            self.budget.reserve(len(payload) + 256)
            try:
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in (429, 503) or attempt == 2:
                    raise
                sleep(60)
        if not isinstance(result, dict):
            raise ValueError("Google Decision вернул не JSON-объект")
        usage = result.get("usage", {})
        actual = usage.get("total_input_tokens") if isinstance(usage, dict) else None
        self.budget.last_actual = actual if type(actual) is int and actual > 0 else None
        if self.budget.last_actual is not None:
            self.budget.reconcile(self.budget.last_actual)
        return json.dumps(result)
