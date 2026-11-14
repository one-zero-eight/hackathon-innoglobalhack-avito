"""Офлайн-бенчмарк контрактов API-клиентов без запросов к провайдерам."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import subprocess
from contextlib import contextmanager
from email.message import Message
from email.utils import formatdate
from io import BytesIO
from time import perf_counter
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from benchmark_support import ROOT, metadata, save_report

EXPORTS = {
    "APIClientSettings", "JSONTransport", "LLMError", "GoogleDecisionClient",
    "GoogleGenerationClient", "OpenAICompatibleClient", "generation_client",
}
PLACEHOLDER_KEY = "offline-placeholder-key"
SCHEMA = {
    "type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
    "required": ["title", "text"], "additionalProperties": False,
}
GENERATED = '{"title":"Сантехника","text":"Устанавливаю раковины."}'


class Reply(BytesIO):
    """Ответ HTTP с настоящим разбором Content-Type, но без сети."""

    def __init__(self, body: Any, content_type: str = "application/json") -> None:
        super().__init__((body if isinstance(body, str) else json.dumps(body)).encode())
        self.headers = Message()
        self.headers["Content-Type"] = content_type


def require(condition: bool, message: str) -> None:
    """Остановить сценарий при нарушении проверяемого контракта."""
    if not condition:
        raise AssertionError(message)


@contextmanager
def rejected(error_type: type[Exception]) -> Iterator[None]:
    """Проверить явный отказ без утечки ключа в сообщение ошибки."""
    try:
        yield
    except error_type as exc:
        require(PLACEHOLDER_KEY not in str(exc), "Ключ попал в сообщение ошибки")
    else:
        raise AssertionError("Некорректный ответ или настройки приняты")


def settings(module: ModuleType, **overrides: Any) -> Any:
    """Создать независимые настройки контрольного клиента."""
    return module.APIClientSettings(**{
        "model_id": "offline-model", "api_key": PLACEHOLDER_KEY,
        "endpoint": "https://example.invalid/v1/chat/completions", **overrides,
    })


def google_reply() -> dict[str, Any]:
    """Контрольный ответ Google с отдельной частью thinking."""
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"thought": True, "text": "Внутреннее рассуждение"}, {"text": GENERATED},
    ]}}]}


def chat_reply() -> dict[str, Any]:
    """Контрольный завершённый ответ Chat Completions."""
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": GENERATED}}]}


def check_clients(module: ModuleType) -> dict[str, int]:
    """Проверить payload, секреты, режимы JSON и отказы трёх провайдеров."""
    counts = {"protocol_cases": 0, "invalid_settings": 0, "invalid_outputs": 0,
              "http_cases": 0, "environment_cases": 0}
    questions = {"mc104": {"type": "noul", "instructions": "Предлагается сантехника?"}}
    decision = module.GoogleDecisionClient(settings(module))
    with patch.object(decision.transport.opener, "open", return_value=Reply({"answers": {"mc104": {"noul": 0.9}}})) as opened:
        result = json.loads(decision.complete("Установка раковин", questions=questions))
        request = opened.call_args.args[0]
        require(json.loads(request.data) == {"model": "offline-model", "input": "Установка раковин", "questions": questions}, "Неверный протокол Decisions")
        require(request.get_header("X-goog-api-key") == PLACEHOLDER_KEY, "Неверная авторизация Google")
        require(result["answers"]["mc104"]["noul"] == 0.9, "Ответ Decisions изменён")
        require(opened.call_args.kwargs["timeout"] == 60, "Таймаут не передан")
    counts["protocol_cases"] += 1
    for questions_value in ({}, [], None):
        with patch.object(decision.transport.opener, "open") as opened, rejected(ValueError):
            decision.complete("Описание", questions=questions_value)
        require(not opened.called, "Неверные questions вызвали сеть")
        counts["invalid_outputs"] += 1

    for client_class, response in ((module.GoogleGenerationClient, google_reply), (module.OpenAICompatibleClient, chat_reply)):
        for mode in ("prompt", "json_object", "json_schema"):
            client = client_class(settings(module, response_mode=mode))
            with patch.object(client.transport.opener, "open", return_value=Reply(response())) as opened:
                require(client.complete("JSON, только исходные факты", response_schema=SCHEMA) == GENERATED, "Текст генерации изменён")
                request = opened.call_args.args[0]
                payload = json.loads(request.data)
                if client_class is module.GoogleGenerationClient:
                    require(payload["contents"] == [{"role": "user", "parts": [{"text": "JSON, только исходные факты"}]}], "Неверный payload Google")
                    require(request.get_header("X-goog-api-key") == PLACEHOLDER_KEY, "Неверный ключ Google")
                    require(("generationConfig" in payload) == (mode != "prompt"), "Неявное включение JSON mode Google")
                    if mode == "json_schema":
                        require(payload["generationConfig"]["responseJsonSchema"] == SCHEMA, "Схема Google изменена")
                else:
                    require(payload["stream"] is False and payload["model"] == "offline-model", "Неверный payload Chat Completions")
                    require(payload["messages"] == [{"role": "user", "content": "JSON, только исходные факты"}], "Потерян prompt Chat Completions")
                    require(request.get_header("Authorization") == f"Bearer {PLACEHOLDER_KEY}", "Неверная авторизация Bearer")
                    require(("response_format" in payload) == (mode != "prompt"), "Неявное включение JSON mode Chat Completions")
                    if mode == "json_schema":
                        require(payload["response_format"]["json_schema"]["schema"] == SCHEMA, "Схема Chat Completions изменена")
            counts["protocol_cases"] += 1
        client = client_class(settings(module, response_mode="json_schema"))
        with patch.object(client.transport.opener, "open") as opened, rejected(ValueError):
            client.complete("Описание")
        require(not opened.called, "Отсутствие схемы вызвало сеть")
        counts["invalid_outputs"] += 1

    bad_settings = [
        {"timeout": 0}, {"timeout": float("inf")}, {"timeout": float("nan")},
        {"max_attempts": 0}, {"max_attempts": True}, {"max_attempts": 6},
        {"retry_delay": -1}, {"retry_delay": float("inf")}, {"model_id": " "},
        {"response_mode": "auto"}, {"endpoint": ""}, {"endpoint": "file:///tmp/model"},
        {"endpoint": "https://user:password@example.invalid/v1"},
        {"endpoint": "https://example.invalid/v1?key=secret"},
        {"endpoint": "https://example.invalid:bad/v1"},
        {"api_key": f"{PLACEHOLDER_KEY}\r\nInjected: header"},
    ]
    for value in bad_settings:
        with rejected(ValueError):
            settings(module, **value)
        counts["invalid_settings"] += 1
    require(PLACEHOLDER_KEY not in repr(settings(module)), "Ключ виден в repr настроек")
    require("api_key" not in settings(module).model_dump(), "Ключ попал в сериализацию настроек")

    transport = module.JSONTransport(settings(module))
    for response in (
        Reply("<!doctype html><title>Vectrust</title>", "text/html; charset=utf-8"),
        Reply("<!doctype html>", "application/json"), Reply("not JSON"),
        Reply([]), Reply(None), Reply({"error": {"message": PLACEHOLDER_KEY}}),
        Reply('{"value": NaN}'), Reply('{"value": Infinity}'),
    ):
        with patch.object(transport.opener, "open", return_value=response) as opened, rejected(module.LLMError):
            transport.post({"input": "Описание"}, "Authorization")
        require(opened.call_count == 1, "Некорректный ответ повторно запрошен")
        counts["invalid_outputs"] += 1
    with patch.object(transport.opener, "open", return_value=Reply({"ok": True}, "application/problem+json")):
        require(transport.post({}, "Authorization") == {"ok": True}, "Не поддержан JSON media type")
    counts["protocol_cases"] += 1

    for status in (301, 302, 307, 308, 400, 401, 403, 404, 429, 500, 502, 503, 504):
        error = HTTPError("https://example.invalid", status, "upstream", Message(), None)
        with patch.object(transport.opener, "open", side_effect=error) as opened, patch.object(module, "sleep") as slept, rejected(module.LLMError):
            transport.post({}, "Authorization")
        attempts = 3 if status in {429, 500, 502, 503, 504} else 1
        require(opened.call_count == attempts and slept.call_count == attempts - 1, "Неверная политика повторов HTTP")
        counts["http_cases"] += 1
    for error in (TimeoutError(), URLError("temporary"), ConnectionError()):
        with patch.object(transport.opener, "open", side_effect=[error, Reply({"ok": True})]) as opened, patch.object(module, "sleep"):
            require(transport.post({}, "Authorization") == {"ok": True}, "Не выполнен повтор сетевой ошибки")
        require(opened.call_count == 2, "Неверное число попыток после сетевой ошибки")
        counts["http_cases"] += 1
    with patch.object(transport.opener, "open", side_effect=TimeoutError()) as opened, patch.object(module, "sleep"), rejected(module.LLMError):
        transport.post({}, "Authorization")
    require(opened.call_count == 3, "Исчерпание сетевых попыток не обработано")
    counts["http_cases"] += 1
    for retry_after, expected_delay in (("4", 4), ("1000", 60), ("garbage", 1), ("nan", 1), (formatdate(104, usegmt=True), 4)):
        headers = Message()
        headers["Retry-After"] = retry_after
        error = HTTPError("https://example.invalid", 429, "quota", headers, None)
        with patch.object(transport.opener, "open", side_effect=[error, Reply({"ok": True})]), patch.object(module, "sleep") as slept, patch.object(module, "time", return_value=100):
            transport.post({}, "Authorization")
        require(slept.call_args.args[0] == expected_delay, "Неверная обработка Retry-After")
        counts["http_cases"] += 1
    require(module._NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid") is None, "Разрешена пересылка ключа при redirect")
    counts["http_cases"] += 1

    invalid_google = [{}, {"candidates": []}, {"candidates": [None]}]
    for reason in ("MAX_TOKENS", "SAFETY", None):
        response = google_reply()
        response["candidates"][0]["finishReason"] = reason
        invalid_google.append(response)
    for parts in ([], [{"thought": True, "text": "reasoning"}], [{"text": None}], ["wrong"], [{"text": " "}]):
        response = google_reply()
        response["candidates"][0]["content"]["parts"] = parts
        invalid_google.append(response)
    invalid_chat = [{}, {"choices": []}, {"choices": [None]}]
    for update in ({"refusal": "blocked"}, {"tool_calls": [{}]}, {"content": None}, {"content": []}, {"content": " "}):
        response = chat_reply()
        response["choices"][0]["message"].update(update)
        invalid_chat.append(response)
    for reason in ("length", "content_filter", "tool_calls", None):
        response = chat_reply()
        response["choices"][0]["finish_reason"] = reason
        invalid_chat.append(response)
    for client_class, responses in ((module.GoogleGenerationClient, invalid_google), (module.OpenAICompatibleClient, invalid_chat)):
        client = client_class(settings(module))
        for response in responses:
            with patch.object(client.transport.opener, "open", return_value=Reply(response)), rejected(module.LLMError):
                client.complete("Описание")
            counts["invalid_outputs"] += 1

    with patch.dict("os.environ", {}, clear=True):
        google = module.generation_client()
        require(isinstance(google, module.GoogleGenerationClient), "Неверный провайдер по умолчанию")
        require(google.model_id == "gemma-4-31b-it", "Неверная модель по умолчанию")
        require(module.GoogleDecisionClient().model_id == "gemini-decision-1", "Неверная модель Decisions")
        with patch.object(google.transport.opener, "open") as opened, rejected(module.LLMError):
            google.complete("Описание")
        require(not opened.called, "Без ключа выполнен запрос")
    counts["environment_cases"] += 1
    with patch.dict("os.environ", {"LLM_API_KEY": "extraction-key", "LLM_MODEL_ID": "decision-model", "GENERATION_MODEL_ID": "my-gemma", "GENERATION_API_KEY": PLACEHOLDER_KEY}, clear=True):
        google = module.generation_client()
        require(google.model_id == "my-gemma" and google.api_key == PLACEHOLDER_KEY, "Настройки шагов смешаны")
        require(google.settings.endpoint.endswith("/my-gemma:generateContent"), "Модель не подставлена в URL Google")
        require(module.GoogleDecisionClient().api_key == "extraction-key", "Генерация изменила ключ Decisions")
    counts["environment_cases"] += 1
    with patch.dict("os.environ", {"GENERATION_PROVIDER": "openai_compatible", "GENERATION_MODEL_ID": "custom-model", "GENERATION_API_KEY": PLACEHOLDER_KEY, "GENERATION_API_URL": "https://example.invalid/v1/chat/completions"}, clear=True):
        require(isinstance(module.generation_client(), module.OpenAICompatibleClient), "Не выбран Chat Completions")
    counts["environment_cases"] += 1
    for environment in ({"GENERATION_PROVIDER": "unknown"}, {"GENERATION_PROVIDER": "openai_compatible"}):
        with patch.dict("os.environ", environment, clear=True), rejected(ValueError):
            module.generation_client()
        counts["environment_cases"] += 1
    return counts


def evaluate_revision(revision: str) -> dict[str, Any]:
    """Прогнать тот же сценарий на версии main.py или отметить отсутствие API."""
    if revision == "working_tree":
        source = (ROOT / "src/main.py").read_text(encoding="utf-8")
    else:
        source = subprocess.check_output(["git", "show", f"{revision}:src/main.py"], cwd=ROOT, text=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD" if revision == "working_tree" else revision], cwd=ROOT, text=True,
    ).strip()
    row = {"revision": revision, "commit": commit, "main_sha256": hashlib.sha256(source.encode()).hexdigest()}
    symbols = {node.name for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    missing = sorted(EXPORTS - symbols)
    if missing:
        return {**row, "status": "unavailable", "reason": "В этой версии нет новых общих API-клиентов", "missing": missing}
    spec = importlib.util.spec_from_loader("api_client_benchmark_revision", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / "src/main.py")
    sys.modules[module.__name__] = module
    started = perf_counter()
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        checks = check_clients(module)
    except Exception as exc:
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        del sys.modules[module.__name__]
    return {**row, "status": "complete", "checks": checks, "passed": sum(checks.values()),
            "offline_elapsed_seconds": perf_counter() - started}


def main() -> int:
    """Сохранить результаты офлайн-проверки; не измерять задержку модели."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-revisions", action="store_true", help="Все коммиты, менявшие src/main.py")
    parser.add_argument("--revision", default="working_tree")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    revisions = [args.revision]
    if args.all_revisions:
        revisions.append(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        revisions.extend(subprocess.check_output(["git", "log", "--format=%H", "--", "src/main.py"], cwd=ROOT, text=True).splitlines())
    rows = [evaluate_revision(revision) for revision in dict.fromkeys(revisions)]
    report = {**metadata(), "scenario": "offline_api_clients", "real_api_calls": 0,
              "status": "failed" if any(row["status"] == "failed" for row in rows) else "complete", "versions": rows}
    save_report(args.report, report)
    for row in rows:
        logging.info("%s: %s; %s", row["revision"][:12], row["status"], row.get("checks", row.get("reason", row.get("error"))))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
