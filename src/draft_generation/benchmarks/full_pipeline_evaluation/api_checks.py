"""Офлайн-проверки HTTP-контракта и переиспользования шагов в существующем сценарии."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
import gc
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import monotonic, sleep
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import weakref

from fastapi.testclient import TestClient

from api import create_app
from main import BasePipeline, Draft, MissingCredentialsError, PipelineConfig, PipelineInput, PipelineRuntime, discover_modules

ROOT = Path(__file__).resolve().parents[4]


class StubPipeline(BasePipeline):
    """Выполнить контролируемое действие без обращения к API моделей."""

    def __init__(self, action: Any) -> None:
        self.action = action

    def run(self, *args: Any) -> Any:
        return self.action(*args)


def check_api() -> list[str]:
    """Проверить выбор модулей, кэш, ошибки и параллельные запросы без сети и БД."""
    checks: list[str] = []

    def verify(condition: bool, name: str) -> None:
        """Сохранить только действительно пройденное условие."""
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    inp = {"description": "Установка розеток и смесителей.", "mcId": 101}
    targets = {"lexical.regex": [104], "lexical.learned_keywords": [105]}
    created: list[tuple[str, str]] = []

    def factory(service: str, config: PipelineConfig) -> BasePipeline:
        """Сохранить реальные запросы к общему loader и вернуть независимый шаг."""
        name = getattr(config, service)
        created.append((service, name))
        if service == "subcategory_extraction":
            return StubPipeline(lambda data: targets[name])
        return StubPipeline(lambda data, ids: [{"mcId": cid, "title": "Монтаж", "text": data.description} for cid in ids])

    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        config_path = directory / "config.json"
        config_path.write_text(json.dumps({"subcategory_extraction": "lexical.regex", "draft_generation": "sample_llm"}))
        with patch("main.load_pipeline", side_effect=AssertionError("Загрузка модели при импорте/старте")):
            importlib.reload(importlib.import_module("api"))
            app = create_app(config_path)
            with TestClient(app) as client:
                verify(client.get("/health").json() == {"status": "ok"}, "health_without_models")
                catalog = client.get("/modules").json()
                verify(catalog["extractors"] == list(discover_modules("subcategory_extraction"))
                       and catalog["generators"] == list(discover_modules("draft_generation")), "complete_catalog_without_imports")
                schema = client.get("/openapi.json").json()
                verify(schema["components"]["schemas"]["Extractor"]["enum"] == catalog["extractors"]
                       and schema["components"]["schemas"]["Generator"]["enum"] == catalog["generators"], "swagger_selector_enums")
                verify(client.get("/docs").status_code == 200, "swagger_without_models")
                verify(catalog["defaults"] == {"extractor": "lexical.regex", "generator": "sample_llm"}, "catalog_defaults")
                for selector in ("extractor=does_not_exist", "extractor=../../main", "generator=unknown"):
                    verify(client.post("/drafts?" + selector, json=inp).status_code == 422, "reject_" + selector)
                for body in ({}, {**inp, "mcId": 999}, {**inp, "description": []}):
                    verify(client.post("/drafts", json=body).status_code == 422, "reject_invalid_body_" + str(len(checks)))
                verify(client.post("/drafts", content="{", headers={"Content-Type": "application/json"}).status_code == 422,
                       "reject_invalid_json")
        with patch("main.load_pipeline", side_effect=factory), TestClient(app) as client:
            first = client.post("/drafts", json=inp)
            verify(first.status_code == 200 and first.json() == {"drafts": [{"mcId": 104, "title": "Монтаж", "text": inp["description"]}]},
                   "defaults_and_exact_output_aliases")
            verify(client.post("/drafts", json=inp).json() == first.json() and len(created) == 2, "repeat_reuses_both_instances")
            second = client.post("/drafts?extractor=lexical.learned_keywords&generator=sample_llm", json=inp)
            verify(second.json()["drafts"][0]["mcId"] == 105 and len(created) == 3, "override_replaces_only_extractor")
            verify(client.post("/drafts", json=inp).json() == first.json() and len(created) == 4, "override_does_not_change_defaults")
        # Эмулируем второй generator для проверки общей механики выбора без добавления модуля.
        with patch("api.discover_modules", side_effect=lambda service: tuple(targets) if service == "subcategory_extraction" else ("sample_llm", "other")):
            two_generators = create_app(config_path)
        created.clear()
        with patch("main.load_pipeline", side_effect=factory), TestClient(two_generators) as client:
            client.post("/drafts", json=inp)
            client.post("/drafts?generator=other", json=inp)
            verify(created == [("subcategory_extraction", "lexical.regex"), ("draft_generation", "sample_llm"), ("draft_generation", "other")],
                   "generator_uses_same_selector_and_cache_mechanism")
            verify(client.get("/modules").json()["defaults"]["generator"] == "sample_llm", "generator_override_is_request_local")
        # Недействительные defaults не маскируются запасной реализацией.
        for extractor in (None, "lexical.unknown"):
            config_path.write_text(json.dumps({"subcategory_extraction": extractor, "draft_generation": "sample_llm"}))
            try:
                create_app(config_path)
            except ValueError:
                checks.append("invalid_default_rejected_" + str(extractor))
            else:
                raise AssertionError("Недействительный default принят")
        config_path.write_text(json.dumps({"subcategory_extraction": "lexical.regex", "draft_generation": "sample_llm"}))
        # Отдельный app на каждый ошибочный ответ исключает влияние прежнего instance cache.
        for output in ([101], [999], [True], ["104"], [104, 104], [None], {104}, None):
            with patch("main.load_pipeline", return_value=StubPipeline(lambda data: output)), TestClient(create_app(config_path)) as client:
                verify(client.post("/drafts", json=inp).status_code == 502, "reject_extractor_output_" + repr(output))
        for output in ([], [{"mcId": 105}], [{"mcId": 104}, {"mcId": 104}], [{"mcId": 101}], [{"mcId": "104"}],
                       [{"mcId": 104, "title": []}], [Draft.model_construct(mc_id=104, title=[])], None):
            def malformed(service: str, config: PipelineConfig) -> BasePipeline:
                """Вернуть правильные цели и намеренно некорректный ответ generator."""
                return StubPipeline(lambda *args: [104] if service == "subcategory_extraction" else output)

            with patch("main.load_pipeline", side_effect=malformed), TestClient(create_app(config_path)) as client:
                response = client.post("/drafts", json=inp)
                verify(response.status_code == 502 and "drafts" not in response.json(), "reject_generator_output_" + repr(output))
        for error, expected in ((FileNotFoundError("private/path"), 503), (ImportError("private dependency"), 503), (ValueError("private settings"), 503)):
            with patch("main.load_pipeline", side_effect=error), TestClient(create_app(config_path)) as client:
                response = client.post("/drafts", json=inp)
                verify(response.status_code == expected and "private" not in response.text, "sanitize_load_" + type(error).__name__)
                verify(client.get("/health").status_code == 200, "health_after_unavailable_" + type(error).__name__)
        with patch("main.load_pipeline", side_effect=[FileNotFoundError("private/path"), StubPipeline(lambda data: [])]) as loader:
            with TestClient(create_app(config_path)) as client:
                verify(client.post("/drafts", json=inp).status_code == 503
                       and client.post("/drafts", json=inp).json() == {"drafts": []}
                       and loader.call_count == 2, "failed_load_can_be_retried")
        for stage in ("subcategory_extraction", "draft_generation"):
            for error in (MissingCredentialsError("private key"), URLError("private endpoint"), ValueError("private body"), RuntimeError("private provider")):
                def failing(service: str, config: PipelineConfig) -> BasePipeline:
                    """Поднять контролируемую ошибку выбранного шага."""
                    def action(*args: Any) -> Any:
                        if service == stage:
                            raise error
                        return [104]
                    return StubPipeline(action)

                with patch("main.load_pipeline", side_effect=failing), TestClient(create_app(config_path)) as client:
                    response = client.post("/drafts", json=inp)
                    verify(response.status_code == (503 if isinstance(error, MissingCredentialsError) else 502)
                           and "private" not in response.text, "sanitize_" + stage + "_" + type(error).__name__)
        with patch("main.load_pipeline", return_value=StubPipeline(lambda data: [])) as loader, TestClient(create_app(config_path)) as client:
            verify(client.post("/drafts", json=inp).json() == {"drafts": []} and loader.call_count == 1, "empty_targets_skip_generator")
        # Настоящие классы обоих шагов и общий run_pipeline; заменён только ответ провайдера.
        from main import run_pipeline

        config = PipelineConfig(subcategory_extraction="lexical.regex", draft_generation="sample_llm")
        with patch("draft_generation.sample_llm.client.SampleLLMClient.complete", return_value='{"title":"Монтаж","text":"Установка розеток и смесителей."}'):
            with TestClient(create_app(config_path)) as client:
                response = client.post("/drafts", json=inp)
                expected = run_pipeline(PipelineInput.model_validate(inp), config).model_dump(by_alias=True)
                verify(response.status_code == 200 and response.json() == expected and bool(expected["drafts"]), "real_modules_match_shared_orchestration")
        with patch.dict(os.environ, {"LLM_API_KEY": ""}), TestClient(create_app(config_path)) as client:
            verify(client.post("/drafts", json=inp).status_code == 503, "real_generator_missing_credentials_is_503")
        # Два перекрывающихся HTTP-запроса не должны одновременно трогать mutable state модулей.
        started, attempted, release = Event(), Event(), Event()
        guard = Lock()
        active = 0
        peak = 0

        def concurrent_factory(service: str, config: PipelineConfig) -> BasePipeline:
            """Сделать перекрытие запросов наблюдаемым без случайных задержек."""
            if service == "draft_generation":
                return factory(service, config)
            name = config.subcategory_extraction

            def extract(data: PipelineInput) -> list[int]:
                """Удержать первый запрос до запуска второго и проверить сериализацию."""
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                try:
                    if name == "lexical.regex":
                        started.set()
                        if not release.wait(5):
                            raise AssertionError("Второй запрос не освободил первый")
                    return targets[name]
                finally:
                    with guard:
                        active -= 1
            return StubPipeline(extract)

        original_run = PipelineRuntime.run

        def observed_run(runtime: PipelineRuntime, data: PipelineInput, config: PipelineConfig) -> Any:
            """Подтвердить, что второй HTTP-запрос уже дошёл до runtime первого."""
            if config.subcategory_extraction == "lexical.learned_keywords":
                attempted.set()
            return original_run(runtime, data, config)

        with (patch("main.load_pipeline", side_effect=concurrent_factory), patch.object(PipelineRuntime, "run", observed_run),
              TestClient(create_app(config_path)) as client):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(client.post, "/drafts?extractor=lexical.regex", json=inp)
                verify(started.wait(5), "first_concurrent_request_entered")
                second = pool.submit(client.post, "/drafts?extractor=lexical.learned_keywords", json=inp)
                verify(attempted.wait(5), "second_concurrent_request_reached_runtime")
                sleep(0.05)
                verify(not second.done(), "second_request_waits_for_first")
                verify(client.get("/health").json() == {"status": "ok"}, "health_remains_responsive_during_inference")
                release.set()
                responses = [first.result(timeout=10), second.result(timeout=10)]
            verify([response.json()["drafts"][0]["mcId"] for response in responses] == [104, 105] and peak == 1,
                   "concurrent_requests_keep_selection_and_serialize")
        # Ссылки на старые модели не остаются в runtime после переключения.
        references: list[Any] = []

        def referenced(service: str, config: PipelineConfig) -> BasePipeline:
            """Сохранить только слабую ссылку для проверки освобождения instance."""
            pipeline = factory(service, config)
            references.append(weakref.ref(pipeline))
            return pipeline

        with patch("main.load_pipeline", side_effect=referenced):
            runtime = PipelineRuntime()
            config = PipelineConfig(subcategory_extraction="lexical.regex", draft_generation="sample_llm")
            runtime.run(PipelineInput.model_validate(inp), config)
            runtime.run(PipelineInput.model_validate(inp), config.model_copy(update={"subcategory_extraction": "lexical.learned_keywords"}))
            gc.collect()
            verify(references[0]() is None and references[1]() is not None and references[2]() is not None, "replaced_instance_is_released")
        # CLI читает/пишет прежний JSON; реальный regex не запускает generator на пустых целях.
        from main import main as cli
        source = directory / "input.json"
        output = directory / "output.json"
        source.write_text(json.dumps({"description": "Здравствуйте! Работаю аккуратно.", "mcId": 101}))
        with patch("main.load_config", return_value=config), chdir(directory):
            verify(cli(["--input", str(source), "--output", str(output)]) == 0
                   and json.loads(output.read_text()) == {"drafts": []}, "cli_contract_and_other_working_directory")
    return checks


def check_server_startup() -> list[str]:
    """Запустить настоящий python -m src и проверить локальный HTTP без облачных вызовов."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    environment = os.environ | {"UVICORN_HOST": "127.0.0.1", "UVICORN_PORT": str(port), "PYTHONPATH": str(ROOT)}
    # Даже ошибочная попытка использовать cloud-клиент не получит реальный ключ.
    environment.update({name: "" for name in ("LLM_API_KEY", "GOOGLE_API_KEY", "SELECTEL_API_KEY", "BESTAPI_API_KEY", "QIANWEN_TOKEN_PLAN_API_KEY")})
    with TemporaryDirectory() as temporary:
        log_path = Path(temporary) / "server.log"
        with log_path.open("w+") as log:
            process = subprocess.Popen([sys.executable, "-m", "src"], cwd=temporary, env=environment, stdout=log, stderr=log)
            try:
                deadline = monotonic() + 15
                while True:
                    try:
                        with urlopen(base + "/health", timeout=1) as response:
                            assert json.load(response) == {"status": "ok"}
                        break
                    except (URLError, TimeoutError):
                        if process.poll() is not None or monotonic() >= deadline:
                            raise AssertionError("HTTP-сервер не запустился")
                        sleep(0.05)
                with urlopen(base + "/modules", timeout=2) as response:
                    assert json.load(response)["extractors"] == list(discover_modules("subcategory_extraction"))
                with urlopen(base + "/openapi.json", timeout=2) as response:
                    assert "/drafts" in json.load(response)["paths"]
                request = Request(base + "/drafts?extractor=lexical.regex", data=json.dumps({
                    "description": "Здравствуйте! Работаю аккуратно.", "mcId": 101,
                }).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=10) as response:
                    assert json.load(response) == {"drafts": []}
                request.data = b'{"description":"x","mcId":999}'
                try:
                    urlopen(request, timeout=2)
                except HTTPError as error:
                    assert error.code == 422
                else:
                    raise AssertionError("HTTP принял неверную категорию")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return ["module_startup_from_other_directory", "real_http_health_catalog_openapi", "real_http_empty_drafts", "real_http_validation"]
