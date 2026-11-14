"""Офлайн-проверка HTTP-контракта FastAPI без обращений к моделям."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import logging
import subprocess
from time import perf_counter
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from benchmark_support import ROOT, metadata, save_report
from main import Draft, LLMError, PipelineConfig, PipelineOutput


def main() -> None:
    """Проверить маршруты, схемы, ошибки и выполнение в thread pool."""
    cases = []
    started = perf_counter()
    payload = {"description": "Устанавливаю раковины.", "mcId": 101}
    output = PipelineOutput(drafts=[Draft(mcId=104, title="Сантехника", text="Устанавливаю раковины.")])
    with TestClient(api.app) as client, patch.object(api, "run_pipeline", return_value=output) as run:
        response = client.post("/drafts", json=payload)
        assert response.status_code == 200 and response.json() == output.model_dump(by_alias=True)
        assert run.call_args.args[0].model_dump(by_alias=True) == payload
        assert run.call_args.args[1] == api.app.state.pipeline_config
        cases.append("json_request_and_response")
        run.reset_mock()
        run.return_value = PipelineOutput(drafts=[])
        assert client.post("/drafts", json=payload).json() == {"drafts": []}
        cases.append("empty_drafts")
        run.reset_mock()
        invalid = [{}, {"description": "x"}, {"description": "x", "mcId": 999},
                   {"description": "x", "mcId": True}, {"mcId": 101}, [], None]
        for value in invalid:
            assert client.post("/drafts", json=value).status_code == 422
        assert client.post("/drafts", content="{", headers={"Content-Type": "application/json"}).status_code == 422
        run.assert_not_called()
        cases.extend(f"invalid_input_{index}" for index in range(len(invalid) + 1))
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert "mcId" in schema["components"]["schemas"]["PipelineInput"]["properties"]
        assert "502" in schema["paths"]["/drafts"]["post"]["responses"]
        run.assert_not_called()
        cases.extend(("health_without_models", "swagger_without_models", "openapi_aliases_and_errors"))
        assert client.get("/drafts").status_code == 405
        cases.append("post_only")
        run.side_effect = LLMError("API завершился с HTTP 500")
        response = client.post("/drafts", json=payload)
        assert response.status_code == 502 and "drafts" not in response.json()
        cases.append("upstream_error_502")
        run.side_effect = ValueError("PRIVATE-MODEL-OUTPUT")
        response = client.post("/drafts", json=payload)
        assert response.status_code == 502 and "PRIVATE-MODEL-OUTPUT" not in response.text
        cases.append("invalid_model_output_502")
        import threading
        handler_threads = []
        def execute(*args, **kwargs):
            """Запомнить поток выполнения синхронного обработчика."""
            handler_threads.append(threading.current_thread().name)
            return output
        run.side_effect = execute
        assert client.post("/drafts", json=payload).status_code == 200
        assert handler_threads and "worker" in handler_threads[0].lower()
        cases.append("blocking_pipeline_in_worker_thread")

    missing = PipelineConfig(subcategory_extraction=None, draft_generation="llm")
    with patch.object(api, "load_config", return_value=missing):
        try:
            with TestClient(api.app):
                raise AssertionError("Неполная конфигурация принята")
        except ValueError:
            pass
    cases.append("invalid_configuration_fails_startup")
    report = {**metadata(), "scenario": "offline_http_api_contract", "status": "complete",
              "real_model_calls": 0, "passed": len(cases), "cases": cases,
              "offline_elapsed_seconds": perf_counter() - started}
    revisions = subprocess.check_output(
        ["git", "log", "--format=%H", "--", "src/api.py", "src/main.py"], cwd=ROOT, text=True,
    ).splitlines()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["versions"] = []
    for revision in dict.fromkeys([head, *revisions]):
        present = subprocess.run(["git", "cat-file", "-e", f"{revision}:src/api.py"], cwd=ROOT, capture_output=True)
        report["versions"].append({
            "commit": revision, "status": "unavailable" if present.returncode else "not_run",
            "reason": "HTTP-адаптер отсутствует" if present.returncode else "Нужен отдельный прогон версии адаптера",
        })
    save_report(Path(__file__).with_name("results.json"), report)
    logging.info("HTTP API: %d checks passed; no model calls", len(cases))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
