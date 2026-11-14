"""Клиент нативного Qwen System One с валидацией и кэшем ответов."""

import hashlib
import json
import math
import os
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import LLMClient

DEFAULT_MODEL = "decision-model-preview"
DEFAULT_BASE_URL = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "out/qwen-decision-cache"
logger = logging.getLogger(__name__)


class QwenDecisionClient(LLMClient):
    """Получать проверенные P(yes), сохраняя только успешные ответы провайдера."""

    def __init__(self) -> None:
        super().__init__(os.getenv("QWEN_DECISION_MODEL") or DEFAULT_MODEL)
        self.api_key = os.getenv("QIANWEN_TOKEN_PLAN_API_KEY", "").strip()
        base_url = os.getenv("QWEN_DECISION_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.endpoint = base_url + "/systemone"
        self.timeout = float(os.getenv("QWEN_DECISION_TIMEOUT", "60"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("QWEN_DECISION_TIMEOUT должен быть конечным положительным")
        self.max_attempts = int(os.getenv("QWEN_DECISION_MAX_ATTEMPTS", "3"))
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("QWEN_DECISION_MAX_ATTEMPTS должен быть от 1 до 5")
        self.cache_dir = Path(os.getenv("QWEN_DECISION_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()
        self.cache_enabled = os.getenv("QWEN_DECISION_CACHE", "1") == "1"
        self.last_cache_hit = False
        self.last_usage: dict = {}
        self.last_attempts = 0

    def _request(self, request: Request) -> dict:
        """Повторить только временные HTTP/сетевые ошибки, учитывая каждую попытку."""
        for attempt in range(1, self.max_attempts + 1):
            self.last_attempts = attempt
            try:
                return self.request_json(request, self.timeout, urlopen)
            except HTTPError as exc:
                transient = exc.code in (429, 500, 502, 503, 504)
                if not transient or attempt == self.max_attempts:
                    raise RuntimeError(f"Qwen Decision HTTP {exc.code}; проверьте ключ, квоту и endpoint") from None
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError("Qwen Decision: сеть недоступна после ограниченных повторов") from exc
            logger.warning("Qwen Decision: временная ошибка, повтор %d/%d", attempt + 1, self.max_attempts)
            sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Qwen Decision: исчерпаны попытки")

    def _validate_response(self, result: object, questions: dict) -> dict:
        """Проверить модель, типы, полноту вопросов и конечность вероятностей."""
        if not isinstance(result, dict) or result.get("model") != self.model_id:
            raise ValueError("Qwen Decision: ответ другой модели или не JSON-объект")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("Qwen Decision: набор answers не совпадает с questions")
        for name, answer in answers.items():
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise ValueError(f"Qwen Decision: неверный тип ответа {name}")
            value = answer.get("noul")
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Qwen Decision: некорректная вероятность {name}")
        return result

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Запросить независимые Noul-вопросы без chat/completions и генерации текста."""
        self.last_cache_hit, self.last_usage, self.last_attempts = False, {}, 0
        questions = kwargs.get("questions")
        if not isinstance(questions, dict) or not questions or any(
            not isinstance(name, str) or not isinstance(question, dict)
            or question.get("type") != "noul" or not isinstance(question.get("instructions"), str)
            for name, question in questions.items()
        ):
            raise ValueError("Qwen Decision: нужен непустой словарь Noul-вопросов")
        payload = {"model": self.model_id, "state": prompt, "questions": questions}
        fingerprint = json.dumps(
            {"endpoint": self.endpoint, "payload": payload},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        path = self.cache_dir / (hashlib.sha256(fingerprint.encode()).hexdigest() + ".json")
        if self.cache_enabled and path.is_file():
            result = self._validate_response(json.loads(path.read_text(encoding="utf-8")), questions)
            self.last_cache_hit = True
            self.record_usage(result.get("usage"), cache_hit=True)
        else:
            if not self.api_key:
                raise ValueError("Задайте QIANWEN_TOKEN_PLAN_API_KEY в .env")
            request = Request(
                self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            result = self._validate_response(self._request(request), questions)
            if self.cache_enabled:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile("w", encoding="utf-8", dir=self.cache_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        json.dump(result, stream, ensure_ascii=False, allow_nan=False)
                        stream.flush()
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
        self.last_usage = result.get("usage", {})
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
