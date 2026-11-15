"""Проверить метрики офлайн или запустить оценку полного решения."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing
from database import table_sha256


def check() -> None:
    """Повторить контрольные метрики, чтение case_test и аудит пересечений."""
    from run import audit_overlap, calculate_metrics, prepare_rows, read_records
    assert calculate_metrics([{102, 105}], [{102, 104}]) == {
        "micro_f1": 0.5, "classification_points": 30.0, "tp": 1, "fp": 1, "fn": 1,
    }
    assert calculate_metrics([set()], [set()])["micro_f1"] == 0
    assert calculate_metrics([{102}], [{101, 102}])["fp"] == 1
    assert calculate_metrics([{102}], [{102}])["micro_f1"] == 1
    assert calculate_metrics([{102}, set()], [{102}, {104, 105}])["micro_f1"] == 0.5
    rows = prepare_rows(read_records("case_test"), 101)
    assert len(rows) == 208
    assert set.union(*(row.expected for row in rows)) == {102, 103, 104, 105, 106}
    overlap = audit_overlap(rows, "case_train")
    assert overlap["overlapping_rows"] == 24
    logging.info("Формула и крайние случаи: OK; case_test: %d строк; пересечения: %d",
                 len(rows), overlap["overlapping_rows"])



def run_live(config_path: Path, report_path: Path, resume: bool) -> int:
    """Прогнать реальные строки с контрольными точками и ограниченным повтором сети."""
    import hashlib
    import json
    import os
    from urllib.error import HTTPError, URLError
    from time import sleep

    from main import PipelineOutput, load_config, run_pipeline
    from run import RunConfig, audit_overlap, calculate_metrics, prepare_rows, read_records, resolve_path

    config_path = config_path.resolve()
    config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    pipeline_path = resolve_path(base, config.pipeline_config)
    selected = load_config(pipeline_path)
    if any(value is None for value in selected.model_dump().values()):
        raise ValueError("Для полного прогона выберите оба модуля")
    dataset = config.dataset_table
    records = read_records(dataset)
    if config.limit is not None:
        records = records[:config.limit]
    rows = prepare_rows(records, config.source_mc_id)
    training = config.training_table
    fingerprint = hashlib.sha256(config_path.read_bytes() + pipeline_path.read_bytes())
    fingerprint.update(table_sha256(dataset).encode())
    # Кэш не должен пережить изменение кода, модели, промпта или артефакта.
    fingerprint.update(metadata()["code_sha256"].encode())
    public_settings = {key: value for key, value in os.environ.items()
                       if key.startswith(("GOOGLE_DECISION_", "SAMPLE_LLM_", "TFIDF_"))
                       and not any(word in key for word in ("KEY", "TOKEN", "SECRET"))}
    public_settings["LLM_MODEL_ID"] = os.getenv("LLM_MODEL_ID", "")
    fingerprint.update(json.dumps(public_settings, sort_keys=True).encode())
    artifact = Path(os.getenv("TFIDF_ARTIFACT_PATH", str(Path(__file__).resolve().parents[4] / "out/tfidf.json")))
    if selected.subcategory_extraction == "tfidf":
        fingerprint.update(artifact.read_bytes())
    signature = fingerprint.hexdigest()
    if resume:
        report = json.loads(report_path.read_text())
        if report.get("fingerprint") != signature:
            raise ValueError("Изменились данные, код, конфигурация, промпты или веса; нужен новый прогон")
        report.pop("metrics", None)
    else:
        report = {**metadata(), "scenario": "live", "fingerprint": signature,
                  "rows": len(rows), "limited": config.limit is not None,
                  "pipeline_config": selected.model_dump(), "run_config": config.model_dump(),
                  "overlap": audit_overlap(rows, training), "results": [], "sessions": []}
    report.update(status="running")
    report.pop("error", None)
    session = {**metadata(), "resumed": resume, "starting_row": len(report["results"]) + 1}
    report["sessions"].append(session)
    started = perf_counter()
    try:
        if len(report["results"]) > len(rows):
            raise ValueError("Контрольная точка длиннее исходного датасета")
        for index, row in enumerate(rows, 1):
            if index <= len(report["results"]):
                saved = report["results"][index - 1]
                PipelineOutput.model_validate(saved["output"])
                if saved["row_number"] != index or saved["expected"] != sorted(row.expected):
                    raise ValueError("Строка контрольной точки не совпала с эталоном")
                continue
            call_started = perf_counter()
            for attempt in range(3):
                try:
                    output = run_pipeline(row.inp, selected)
                    break
                except (ConnectionError, TimeoutError, URLError) as exc:
                    if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                        raise
                    if attempt == 2:
                        raise
                    logging.warning("Строка %d: %s, повтор полного вызова через 60 с", index, type(exc).__name__)
                    sleep(60)
            elapsed = perf_counter() - call_started
            report["results"].append({"row_number": index, "expected": sorted(row.expected),
                                      "predicted": sorted({draft.mc_id for draft in output.drafts}),
                                      "output": output.model_dump(by_alias=True), "seconds": elapsed,
                                      "attempts": attempt + 1})
            report.update(completed_rows=index, response_time=timing([item["seconds"] for item in report["results"]]))
            save_report(report_path, report)
            logging.info("Полный пайплайн: %d/%d, %.3f с", index, len(rows), elapsed)
        report.update(status="complete", metrics=calculate_metrics(
            [set(item["expected"]) for item in report["results"]],
            [set(item["predicted"]) for item in report["results"]],
        ))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      failed_row=len(report["results"]) + 1)
        logging.error("Прогон не завершён: %s", report["error"])
    finally:
        session["elapsed_seconds"] = perf_counter() - started
        report["elapsed_seconds"] = sum(item["elapsed_seconds"] for item in report["sessions"])
        save_report(report_path, report)
    return 0 if report["status"] == "complete" else 1


def main() -> int:
    """Проверить оценщик или выполнить реальный полный прогон."""
    from run import DEFAULT_RUN_CONFIG

    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--live", action="store_true")
    selection.add_argument("--api-check", action="store_true", help="Проверить HTTP API и runtime офлайн, без БД и моделей")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--resume", action="store_true", help="Продолжить live из контрольной точки без повторной генерации готовых строк")
    args = parser.parse_args()
    if args.live:
        return run_live(args.config, Path(__file__).with_name("live-results.json"), args.resume)
    if args.resume:
        parser.error("--resume требует --live")
    if args.api_check:
        from api_checks import check_api, check_server_startup
        from unittest.mock import patch

        path = Path(__file__).with_name("checks-results.json")
        report = json.loads(path.read_text()) if path.exists() else {"scenario": "offline"}
        if "results" not in report:
            report["results"] = {"evaluator": {key: value for key, value in report.items()}}
        result = {**metadata(), "status": "running", "api_calls": 0, "database_required": False}
        report["results"]["http_api"] = result
        started = perf_counter()
        try:
            with (patch.dict("os.environ", {"LLM_PRICING_FILE": "", "LLM_API_KEY": ""}),
                  patch("main.LLMClient.request_json", side_effect=AssertionError("Внешний API запрещён в offline проверке"))):
                result["checks"] = check_api() + check_server_startup()
            result.update(status="complete", checks_passed=len(result["checks"]))
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logging.exception("HTTP API: проверки не пройдены")
        result["elapsed_seconds"] = perf_counter() - started
        report["status"] = "complete" if all(item["status"] == "complete" for item in report["results"].values()) else "failed"
        save_report(path, report)
        logging.info("HTTP API: %s, %d проверок", result["status"], result.get("checks_passed", 0))
        return 0 if result["status"] == "complete" else 1
    started = perf_counter()
    report = {**metadata(), "scenario": "offline"}
    check()
    report.update(status="complete", rows=208, overlapping_rows=24, elapsed_seconds=perf_counter() - started)
    report["results"] = {"evaluator": dict(report)}
    save_report(Path(__file__).with_name("checks-results.json"), report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
