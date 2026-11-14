"""Нативный клиент Jev с проверкой ответов и дисковым кэшем."""

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from main import LLMClient

DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_ENDPOINT = "https://bestapi.fugue.pro/v1/systemone"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "out/jev-cache"


class JevClient(LLMClient):
    """Запрашивать вероятности Noul у выбранной модели Jev без подмены."""

    def __init__(self) -> None:
        super().__init__(os.getenv("JEV_MODEL") or DEFAULT_MODEL)
        self.api_key = os.getenv("BESTAPI_API_KEY")
        self.endpoint = os.getenv("JEV_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout = float(os.getenv("JEV_TIMEOUT", "60"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("JEV_TIMEOUT должен быть конечным положительным числом")
        self.cache_dir = Path(os.getenv("JEV_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()
        self.last_cache_hit = False
        self.last_usage: dict[str, object] | None = None

    def _validate_response(self, result: object, questions: dict[str, object]) -> dict:
        """Проверить модель, точный набор ответов и числовые вероятности."""
        if not isinstance(result, dict):
            raise ValueError("Jev вернул не JSON-объект")
        if result.get("model") != self.model_id:
            raise ValueError("Модель в ответе Jev не совпадает с запрошенной")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("Набор ответов Jev не совпадает с вопросами")
        for name, answer in answers.items():
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if (
                type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"Некорректная вероятность Jev для {name}")
        return result

    def _store_response(self, path: Path, result: dict) -> None:
        """Атомарно сохранить только проверенный фактический ответ API."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.cache_dir,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            try:
                json.dump(result, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Получить проверенный JSON для state/questions из API или кэша."""
        self.last_cache_hit = False
        self.last_usage = None
        questions = kwargs.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Jev нужен непустой словарь questions")
        if any(
            not isinstance(name, str)
            or not isinstance(question, dict)
            or question.get("type") != "noul"
            or not isinstance(question.get("instructions"), str)
            for name, question in questions.items()
        ):
            raise ValueError("Jev поддерживает вопросы Noul с текстом instructions")
        payload = {"model": self.model_id, "state": prompt, "questions": questions}
        cache_key = hashlib.sha256(json.dumps(
            {"endpoint": self.endpoint, "payload": payload},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{cache_key}.json"
        if path.is_file():
            result = self._validate_response(
                json.loads(path.read_text(encoding="utf-8")), questions,
            )
            self.last_cache_hit = True
            self.record_usage(result.get("usage"), cache_hit=True)
        else:
            if not self.api_key:
                raise ValueError("Укажите BESTAPI_API_KEY в .env для запросов Jev")
            request = Request(
                self.endpoint,
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            result = self._validate_response(self.request_json(request, self.timeout, urlopen), questions)
            self._store_response(path, result)
        usage = result.get("usage")
        self.last_usage = usage if isinstance(usage, dict) else None
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
