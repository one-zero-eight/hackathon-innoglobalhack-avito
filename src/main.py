"""CLI-ядро проекта.

Здесь живёт всё общее для сервисов: микрокатегории, контракт входа/выхода,
база пайплайнов и клиент LLM. Логика отдельных шагов — в папках сервисов,
которые импортируют общее из этого модуля.

Импорты модулей выполняются при загрузке пайплайна: сервисы импортируют main,
и модульный импорт на верхнем уровне дал бы циклическую зависимость.

Пример:
    python src/main.py --input data/input.json --output out/output.json
"""

import argparse
import json
import logging
import os
import math
import sys
from abc import ABC, abstractmethod
from email.utils import parsedate_to_datetime
from importlib import import_module
from pathlib import Path
from time import sleep, time
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import load_dotenv
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# --- Микрокатегории кейса «Ремонт и отделка» ---
# Таблица зафиксирована условиями кейса (docs/Innoglobalhack-Avito_2026.pdf):
# новые микрокатегории не создаются, выходить за таблицу нельзя.

MICROCATEGORIES: dict[int, str] = {
    101: "Ремонт под ключ",
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}

VALID_MCIDS: frozenset[int] = frozenset(MICROCATEGORIES)


def is_valid_mcid(mcid: int) -> bool:
    """Проверить, что mcId есть в таблице микрокатегорий кейса."""
    return mcid in VALID_MCIDS


def mcid_name(mcid: int) -> str | None:
    """Название микрокатегории по mcId или None, если такой категории нет."""
    return MICROCATEGORIES.get(mcid)


# --- Контракт входа/выхода ---
# Контракт зафиксирован условиями кейса и меняется только с согласования команды:
# вход — {"description": str, "mcId": int},
# выход — {"drafts": [{"mcId": int, "title": str|null, "text": str|null}]}.


def _validate_mcid(value: int) -> int:
    """Отбросить mcId, которого нет в таблице микрокатегорий кейса."""
    if not is_valid_mcid(value):
        raise ValueError(f"Неизвестная микрокатегория: {value}")
    return value


McId = Annotated[int, AfterValidator(_validate_mcid)]


class PipelineInput(BaseModel):
    """Входные данные пайплайна: текст объявления и его микрокатегория."""

    model_config = ConfigDict(populate_by_name=True)

    description: str
    mc_id: McId = Field(alias="mcId")


class Draft(BaseModel):
    """Черновик объявления для целевой микрокатегории."""

    model_config = ConfigDict(populate_by_name=True)

    mc_id: McId = Field(alias="mcId")
    title: str | None = None
    text: str | None = None


class PipelineOutput(BaseModel):
    """Выходные данные пайплайна: список дополнительных черновиков."""

    model_config = ConfigDict(populate_by_name=True)

    drafts: list[Draft] = Field(default_factory=list)


# --- База пайплайнов ---
# Один вариант сервиса — один класс с методом run().


class BasePipeline(ABC):
    """Оркестрируемый шаг пайплайна."""

    name: str = "base"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Выполнить шаг пайплайна и вернуть результат."""


# --- Клиент LLM ---
# Общий интерфейс клиента. Точный model id и ключ берём из окружения (.env):
# LLM_MODEL_ID — идентификатор модели (по умолчанию "jev"),
# LLM_API_KEY — ключ доступа (в репозиторий не коммитим).


class LLMClient(ABC):
    """Интерфейс клиента LLM для конкретных реализаций."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "jev")
        self.api_key = os.getenv("LLM_API_KEY")

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Получить ответ модели на промпт."""


# --- Общие API-клиенты нового пайплайна ---

GOOGLE_DECISIONS_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
GOOGLE_GENERATION_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_DECISION_MODEL = "gemini-decision-1"
DEFAULT_GENERATION_MODEL = "gemma-4-31b-it"
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY = 60.0
ResponseMode = Literal["prompt", "json_object", "json_schema"]


class LLMError(ValueError):
    """Явная ошибка API без тела ответа и секретов в сообщении."""


class APIClientSettings(BaseModel):
    """Проверенные настройки одного клиента, независимо от другого шага."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, hide_input_in_errors=True,
    )

    model_id: str = Field(min_length=1)
    api_key: str = Field(default="", repr=False, exclude=True)
    endpoint: str
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=5, strict=True)
    retry_delay: float = Field(default=1, ge=0, le=MAX_RETRY_DELAY, allow_inf_nan=False)
    response_mode: ResponseMode = "prompt"

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Не допускать управляющие символы и переносы в HTTP-заголовке."""
        if any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise ValueError("API-ключ должен содержать только печатные ASCII-символы без пробелов")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Разрешить HTTP(S) без встроенных секретов, query и fragment."""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(char.isspace() for char in value)
        ):
            raise ValueError("Нужен HTTP(S) endpoint без credentials, query и fragment")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("Некорректный порт endpoint") from exc
        return value

    @classmethod
    def from_env(
        cls, prefix: str, *, model_id: str, endpoint: str, api_key: str = "",
    ) -> "APIClientSettings":
        """Прочитать параметры с префиксом; пустые значения считать незаданными."""
        return cls(
            model_id=os.getenv(f"{prefix}_MODEL_ID") or model_id,
            api_key=os.getenv(f"{prefix}_API_KEY") or api_key,
            endpoint=os.getenv(f"{prefix}_API_URL") or endpoint,
            timeout=os.getenv(f"{prefix}_TIMEOUT") or 60,
            max_attempts=int(os.getenv(f"{prefix}_MAX_ATTEMPTS") or 3),
            retry_delay=os.getenv(f"{prefix}_RETRY_DELAY") or 1,
            response_mode=os.getenv(f"{prefix}_RESPONSE_MODE") or "prompt",
        )


class _NoRedirect(HTTPRedirectHandler):
    """Не пересылать ключ авторизации на адрес из HTTP redirect."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str,
    ) -> None:
        """Отклонить redirect как ошибку конфигурации API."""
        return None


def _retry_delay(settings: APIClientSettings, attempt: int, error: Exception) -> float:
    """Учесть Retry-After в секундах или HTTP-date, ограничив ожидание."""
    delay = settings.retry_delay * 2**attempt
    raw = error.headers.get("Retry-After") if isinstance(error, HTTPError) and error.headers else None
    if raw:
        try:
            parsed = float(raw)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw).timestamp() - time()
            except (TypeError, ValueError, OverflowError):
                parsed = delay
        if math.isfinite(parsed):
            delay = max(delay, parsed)
    return min(MAX_RETRY_DELAY, max(0, delay))


def _invalid_json_constant(value: str) -> None:
    """Отклонить NaN и Infinity, не являющиеся допустимым JSON."""
    raise ValueError("Недопустимая числовая константа JSON")


class JSONTransport:
    """POST JSON с ограниченными повторами и проверкой ответа сервера."""

    def __init__(self, settings: APIClientSettings) -> None:
        self.settings = settings
        self.opener = build_opener(_NoRedirect())

    def post(self, payload: dict[str, Any], auth_header: str) -> dict[str, Any]:
        """Получить JSON-объект; HTML, redirect и ошибки API не скрывать."""
        if not self.settings.api_key:
            raise LLMError("Не задан API-ключ выбранного клиента")
        authorization = (
            f"Bearer {self.settings.api_key}" if auth_header == "Authorization"
            else self.settings.api_key
        )
        request = Request(
            self.settings.endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={auth_header: authorization, "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        for attempt in range(self.settings.max_attempts):
            try:
                with self.opener.open(request, timeout=self.settings.timeout) as response:
                    content_type = response.headers.get_content_type()
                    if content_type != "application/json" and not content_type.endswith("+json"):
                        raise LLMError("API вернул не JSON (возможно HTML); проверьте endpoint и маршрут модели")
                    raw = response.read()
                try:
                    result = json.loads(raw, parse_constant=_invalid_json_constant)
                except (ValueError, UnicodeError):
                    raise LLMError("API вернул некорректный JSON") from None
                if not isinstance(result, dict):
                    raise LLMError("Ожидался JSON-объект в ответе API")
                if "error" in result:
                    raise LLMError("API вернул объект error вместо результата")
                return result
            except HTTPError as exc:
                exc.close()
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt + 1 == self.settings.max_attempts:
                    raise LLMError(f"API завершился с HTTP {exc.code}; попыток: {attempt + 1}") from None
                error: Exception = exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise LLMError(f"Сетевая ошибка API; попыток: {attempt + 1}") from None
                error = exc
            delay = _retry_delay(self.settings, attempt, error)
            logger.warning("Временная ошибка API, повтор %d/%d через %.1f с",
                           attempt + 2, self.settings.max_attempts, delay)
            sleep(delay)
        raise LLMError("Исчерпаны попытки API")


class _ConfiguredClient(LLMClient):
    """Общая инициализация конкретных API-клиентов без сетевых вызовов."""

    def __init__(self, settings: APIClientSettings) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.api_key = settings.api_key
        self.transport = JSONTransport(settings)


class GoogleDecisionClient(_ConfiguredClient):
    """Google Decisions API: описание и typed questions, без генерации текста."""

    def __init__(self, settings: APIClientSettings | None = None) -> None:
        if settings is None:
            settings = APIClientSettings.from_env(
                "EXTRACTION", model_id=os.getenv("LLM_MODEL_ID") or DEFAULT_DECISION_MODEL,
                endpoint=GOOGLE_DECISIONS_URL,
                api_key=os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
            )
        super().__init__(settings)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Вернуть JSON API; смысл и вероятности проверяет модуль выделения."""
        questions = kwargs.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Google Decision требует непустой словарь questions")
        result = self.transport.post(
            {"model": self.model_id, "input": prompt, "questions": questions}, "x-goog-api-key",
        )
        return json.dumps(result, ensure_ascii=False, allow_nan=False)


def _response_schema(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Получить схему только при явно включённом schema mode."""
    schema = kwargs.get("response_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("json_schema требует непустой response_schema")
    return schema


class GoogleGenerationClient(_ConfiguredClient):
    """Google GenerateContent API, включая размещённые модели Gemma."""

    def __init__(self, settings: APIClientSettings | None = None) -> None:
        if settings is None:
            model_id = os.getenv("GENERATION_MODEL_ID") or DEFAULT_GENERATION_MODEL
            settings = APIClientSettings.from_env(
                "GENERATION", model_id=model_id,
                endpoint=GOOGLE_GENERATION_URL.format(model=quote(model_id, safe="")),
                api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY") or "",
            )
        super().__init__(settings)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Получить завершённый текст ответа, исключая части с thinking."""
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if self.settings.response_mode != "prompt":
            config: dict[str, Any] = {"responseMimeType": "application/json"}
            if self.settings.response_mode == "json_schema":
                config["responseJsonSchema"] = _response_schema(kwargs)
            payload["generationConfig"] = config
        result = self.transport.post(payload, "x-goog-api-key")
        try:
            candidate = result["candidates"][0]
            if candidate.get("finishReason") != "STOP":
                raise LLMError("Google не завершил генерацию: блокировка или лимит токенов")
            parts = candidate["content"]["parts"]
            if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
                raise LLMError("Google вернул некорректные части ответа")
            texts = [part["text"] for part in parts if not part.get("thought") and "text" in part]
            if not texts or any(not isinstance(text, str) for text in texts):
                raise LLMError("Google не вернул текст генерации")
            text = "".join(texts)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("Google вернул некорректную оболочку генерации") from None
        if not text.strip():
            raise LLMError("Google вернул пустую генерацию")
        return text


class OpenAICompatibleClient(_ConfiguredClient):
    """Непотоковый Chat Completions с явно заданными endpoint и моделью."""

    def __init__(self, settings: APIClientSettings | None = None) -> None:
        if settings is None:
            settings = APIClientSettings.from_env("GENERATION", model_id="", endpoint="")
        super().__init__(settings)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Получить content; отказы, tool calls и обрыв текста считать ошибкой."""
        payload: dict[str, Any] = {
            "model": self.model_id, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.settings.response_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.settings.response_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "draft", "strict": True, "schema": _response_schema(kwargs)},
            }
        result = self.transport.post(payload, "Authorization")
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal") or message.get("tool_calls"):
                raise LLMError("Chat Completions не завершил текстовую генерацию")
            text = message["content"]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("Chat Completions вернул некорректную оболочку генерации") from None
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Chat Completions вернул пустую или нетекстовую генерацию")
        return text


def generation_client() -> LLMClient:
    """Выбрать провайдер генерации без переключения при ошибках API."""
    provider = os.getenv("GENERATION_PROVIDER") or "google"
    if provider == "google":
        return GoogleGenerationClient()
    if provider == "openai_compatible":
        return OpenAICompatibleClient()
    raise ValueError("GENERATION_PROVIDER: допустимы google и openai_compatible")


# --- Оркестрация ---


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
ServiceName = Literal["subcategory_extraction", "draft_generation"]
PIPELINE_CLASSES: dict[ServiceName, str] = {
    "subcategory_extraction": "SubcategoryExtractionPipeline",
    "draft_generation": "DraftGenerationPipeline",
}
ModuleName = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]*$")]


class PipelineConfig(BaseModel):
    """Выбранный модуль для каждого раздела проекта."""

    model_config = ConfigDict(extra="forbid")

    subcategory_extraction: ModuleName | None
    draft_generation: ModuleName | None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Прочитать и проверить конфигурацию выбора модулей."""
    return PipelineConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_pipeline(
    service: ServiceName, config: PipelineConfig | None = None
) -> BasePipeline:
    """Загрузить выбранный класс из <service>/<module>/pipeline.py."""
    if service not in PIPELINE_CLASSES:
        raise ValueError(f"Неизвестный раздел: {service}")
    config = config if config is not None else load_config()
    module_name = getattr(config, service)
    if module_name is None:
        raise ValueError(f"Для раздела {service} не выбран модуль в config.json")
    module_path = f"{service}.{module_name}.pipeline"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name in {service, f"{service}.{module_name}", module_path}:
            raise ValueError(f"Модуль раздела не найден: {module_path}") from exc
        raise
    class_name = PIPELINE_CLASSES[service]
    pipeline_class = getattr(module, class_name, None)
    if not isinstance(pipeline_class, type) or not issubclass(pipeline_class, BasePipeline):
        raise TypeError(f"{module_path}.{class_name} должен наследовать BasePipeline")
    return pipeline_class()


def run_pipeline(
    inp: PipelineInput, config: PipelineConfig | None = None
) -> PipelineOutput:
    """Прогнать выбранные модули: выделение -> генерация."""
    config = config if config is not None else load_config()
    extraction = load_pipeline("subcategory_extraction", config)
    generation = load_pipeline("draft_generation", config)

    target_mc_ids = extraction.run(inp)
    drafts = generation.run(inp, target_mc_ids)
    return PipelineOutput(drafts=drafts)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    """CLI: пути к файлу ввода и файлу вывода."""
    parser = argparse.ArgumentParser(description="Выделение услуг и черновики объявлений")
    parser.add_argument("--input", required=True, type=Path, help="путь к JSON на вход")
    parser.add_argument("--output", required=True, type=Path, help="путь к JSON на выход")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Прочитать входной файл, прогнать пайплайн, записать выходной файл."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()  # переменные из .env (LLM_MODEL_ID, LLM_API_KEY)
    args = build_parser().parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    inp = PipelineInput.model_validate(payload)

    output = run_pipeline(inp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output.model_dump(by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Готово: %s", args.output)
    return 0


if __name__ == "__main__":
    # Модули импортируют main: CLI должен использовать тот же BasePipeline.
    sys.modules["main"] = sys.modules[__name__]
    sys.exit(main())
