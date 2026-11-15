"""Диагностика экспортированных таблиц на полном корпусе или фиксированной выборке."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
from random import Random
import sys
from tempfile import TemporaryDirectory
from threading import local
from time import perf_counter, sleep
from typing import Any, Iterator
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_support import metadata, save_report, setup, timing
from database import read_records, table_sha256
from main import BasePipeline, MICROCATEGORIES, PipelineConfig, load_pipeline
from run import EvaluationRow, audit_overlap, calculate_metrics, prepare_rows
from subcategory_extraction.benchmarks.classifier_methods.run import METHODS
from subcategory_extraction.benchmarks.configurations import BY_KEY, client_observations, configuration_metadata, configured

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TABLE = "yandex_yoola"
SEED = 20261115
SIZE = 300
SCORED_CATEGORIES = set(MICROCATEGORIES) - {101}
logger = logging.getLogger(__name__)


def sample_records(table: str = TABLE, size: int = SIZE, seed: int = SEED) -> tuple[list[dict[str, Any]], list[int]]:
    """Отличить отсутствие разметки от размеченного пустого списка услуг."""
    eligible, excluded = [], []
    for record in read_records(table):
        extra = record.get("_extra", {})
        if table != TABLE or any(isinstance(extra.get(key), list) for key in ("found_microcategories", "target_mc_ids")):
            eligible.append(record)
        else:
            excluded.append(record["_row_id"])
    if size < 0 or size > len(eligible):
        raise ValueError(f"Размер выборки должен быть 0 (все строки) или 1–{len(eligible)}")
    return (Random(seed).sample(eligible, size) if size else eligible), excluded


def validate_output(output: object, row: EvaluationRow) -> list[int]:
    """Проверить полный контракт до ограничения классов диагностической метрики."""
    if not isinstance(output, list) or any(
        type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id for cid in output
    ) or len(output) != len(set(output)):
        raise ValueError("Неверный набор дополнительных микрокатегорий")
    return sorted(output)


def retryable(error: Exception) -> bool:
    """Повторять только транспортные сбои и временные ошибки провайдера."""
    if isinstance(error, HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, (URLError, TimeoutError, ConnectionError))


@contextmanager
def fresh_caches() -> Iterator[None]:
    """Создать новые кэши до создания всех вложенных API-клиентов latency-прогона."""
    root = ROOT / "out/benchmark-cache"
    root.mkdir(parents=True, exist_ok=True)
    names = ("JEV_CACHE_DIR", "QWEN_DECISION_CACHE_DIR", "STRUCTURED_LLM_CACHE_DIR", "QWEN_DECISION_CACHE")
    previous = {key: os.environ.get(key) for key in names}
    try:
        with TemporaryDirectory(dir=root) as temporary:
            for key in names[:-1]:
                os.environ[key] = str(Path(temporary) / key.lower())
            os.environ["QWEN_DECISION_CACHE"] = "0"
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load(method: str) -> BasePipeline:
    """Создать выбранный модуль без изменения его опубликованных параметров."""
    return load_pipeline("subcategory_extraction", PipelineConfig(subcategory_extraction=method, draft_generation=None))


def inference_fingerprint() -> str:
    """Не позволить продолжить ответы с другими весами, промптами или параметрами."""
    digest = hashlib.sha256()
    paths = [ROOT / "src/main.py", ROOT / "out/tfidf.json"]
    paths.extend(path for path in (ROOT / "src/subcategory_extraction").rglob("*")
                 if path.is_file() and "benchmarks" not in path.parts and "__pycache__" not in path.parts
                 and (path.suffix == ".py" or "artifacts" in path.parts or "prompts" in path.parts
                      or path.name == "artifact-manifest.json"))
    prefixes = ("JEV_", "LOCAL_JEV_", "SEMANTIC_", "RUBERT_F1_", "TFIDF_", "LEARNED_",
                "SNIPPET_", "MORPH_LOGREG_", "STRUCTURED_LLM_", "GOOGLE_DECISION_", "QWEN_DECISION_", "LLM_MODEL_ID")
    settings = {key: value for key, value in os.environ.items() if key.startswith(prefixes)
                and not key.endswith(("_KEY", "_PASSWORD", "_TOKEN")) and "CACHE" not in key}
    for key, value in settings.items():
        if key.endswith(("_PATH", "_ARTIFACT", "_DATASET")):
            candidate = Path(value).expanduser().resolve()
            if candidate.is_dir():
                paths.extend(path for path in candidate.rglob("*") if path.is_file())
            else:
                paths.append(candidate)
    for path in sorted(set(path.resolve() for path in paths)):
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).encode())
        if path.is_file():
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        else:
            digest.update(b"missing")
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def measure_latency(method: str, rows: list[EvaluationRow]) -> dict[str, Any]:
    """Измерить десять последовательных вызовов на входах текущей provider-выборки."""
    result: dict[str, Any] = {
        "status": "running", "rows": [],
        "scope": {"loading_included": False, "first_call_included": True, "response_cache_hits": 0,
                  "fresh_api_caches": True, "quota_wait_and_retries_included": True,
                  "sample_seed": 44, "warmup": False,
                  "other_activity": "F1 jobs for other modules may overlap; CPU and provider load not isolated"},
    }
    durations = []
    try:
        with fresh_caches():
            pipeline = load(method)
            for row in Random(44).sample(rows, 10):
                started = perf_counter()
                output = validate_output(pipeline.run(row.inp), row)
                elapsed = perf_counter() - started
                clients = client_observations(pipeline)
                if any(client.get("cache_hit") for client in clients):
                    raise ValueError("Latency получил ответ из кэша")
                durations.append(elapsed)
                result["rows"].append({"description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                                       "source_mc_id": row.inp.mc_id, "predicted": output,
                                       "seconds": elapsed, "api_clients": clients})
                logger.info("%s latency: %d/10, %.3f с", method, len(durations), elapsed)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["response_time"] = timing(durations)
    return result


def evaluate_method(method: str, rows: list[EvaluationRow], report: dict[str, Any], path: Path, workers: int,
                    reuse: dict[str, Any] | None = None, configuration: dict[str, Any] | None = None) -> None:
    """Сохранять каждый успешный ответ и продолжать только недостающие строки."""
    key = configuration["key"] if configuration else "tfidf::legacy" if method == "lexical.tfidf" else method.rsplit(".", 1)[-1]
    result = report["results"].setdefault(key, {
        "module": method, "status": "running", "predicted": [None] * len(rows),
        "inference": [None] * len(rows), "row_errors": {}, "workers": workers,
        "measurement_metadata": {key: report[key] for key in ("date", "commit", "code_sha256", "machine", "environment")},
    })
    if configuration is not None:
        if result.get("configuration", configuration) != configuration:
            raise ValueError("Настройки конфигурации изменились при продолжении")
        result["configuration"] = configuration
    if reuse is not None:
        saved = reuse["results"].get(key, {})
        if saved.get("module") == method and saved.get("configuration") == configuration:
            indexed = {(r["description_sha256"], r["source_mc_id"]): (r, p, observation)
                       for r, p, observation in zip(reuse["rows"], saved["predicted"], saved["inference"], strict=True)}
            for index, identity in enumerate(report["rows"]):
                match = indexed.get((identity["description_sha256"], identity["source_mc_id"]))
                if match and match[1] is not None and result["predicted"][index] is None:
                    old_row, output, observation = match
                    if old_row["expected"] != identity["expected"]:
                        raise ValueError("Эталон изменился при переносе предсказаний")
                    result["predicted"][index] = validate_output(output, rows[index])
                    result["inference"][index] = {**observation, "reused_from_full_dataset": True}
            result["reuse_measurement_metadata"] = saved.get("measurement_metadata")
            result["reuse_origin_status"] = saved.get("status")
    pending = [i for i, prediction in enumerate(result["predicted"]) if prediction is None]
    scored_categories = set(report["protocol"]["scored_categories"])
    result["resumed_predictions"] = len(rows) - len(pending)
    try:
        pipeline = load(method) if pending else None
        describe = getattr(pipeline, "benchmark_metadata", None)
        if callable(describe):
            result["model_provenance"] = describe()
        state = local()

        def predict(index: int) -> tuple[int, list[int] | None, dict[str, Any]]:
            """Изолировать клиент потока и записать повторы без замены ошибочного ответа."""
            for attempt in range(1, 3):
                started = perf_counter()
                try:
                    if workers == 1:
                        active = pipeline
                    else:
                        if not hasattr(state, "pipeline"):
                            state.pipeline = load(method)
                        active = state.pipeline
                    output = validate_output(active.run(rows[index].inp), rows[index])
                    return index, output, {"api_clients": client_observations(active),
                                           "seconds_including_cache_if_any": perf_counter() - started,
                                           "benchmark_attempts": attempt,
                                           "probabilities": getattr(active, "last_probabilities", None)}
                except Exception as error:
                    if attempt == 1 and retryable(error):
                        sleep(1)
                        continue
                    return index, None, {"error": f"{type(error).__name__}: {error}", "benchmark_attempts": attempt}
            raise RuntimeError("Недостижимая ветка повтора")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(predict, index) for index in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                index, output, observation = future.result()
                result["inference"][index] = observation
                result["predicted"][index] = output
                if output is None:
                    result["row_errors"][str(index)] = observation["error"]
                else:
                    result["row_errors"].pop(str(index), None)
                if completed % (100 if len(rows) > 1000 else 10) == 0 or output is None or completed == len(pending):
                    save_report(path, report)
                if completed % 20 == 0:
                    logger.info("%s: %d/%d", method, len(rows) - len(pending) + completed, len(rows))
        result["completed_rows"] = sum(output is not None for output in result["predicted"])
        if result["completed_rows"] == len(rows):
            result["metrics"] = calculate_metrics(
                [row.expected & scored_categories for row in rows],
                [set(output) & scored_categories for output in result["predicted"]],
            )
            result.update(status="completed")
            result.pop("error", None)
            logger.info("%s: N=%d, micro-F1=%.6f", method, len(rows), result["metrics"]["micro_f1"])
        else:
            result.update(status="failed", error=f"Нет ответа для {len(rows) - result['completed_rows']} строк; F1 не рассчитан")
            result.pop("metrics", None)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result.pop("metrics", None)
        logger.exception("%s: недоступен", method)
    save_report(path, report)
    if result.get("latency", {}).get("status") != "complete":
        result["latency"] = measure_latency(method, rows)
        save_report(path, report)


def main() -> int:
    """Выполнить текущие модули на одном фиксированном сэмпле из PostgreSQL."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--methods", nargs="+", choices=METHODS)
    selection.add_argument("--configurations", nargs="+", choices=BY_KEY)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--report", type=Path, default=HERE / "export-diagnostic-results.json")
    parser.add_argument("--dataset", choices=(TABLE, "igh_deduped_no_conflict"), default=TABLE)
    parser.add_argument("--size", type=int, default=SIZE, help="0 — весь датасет; иначе фиксированная выборка")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reuse-report", type=Path, help="Переиспользовать совпадающие входы той же модели из полного набора")
    args = parser.parse_args()
    setup()
    if args.dataset != TABLE and args.report.resolve() == (HERE / "export-diagnostic-results.json").resolve():
        parser.error("Для другого датасета нужен отдельный --report")
    records, excluded = sample_records(args.dataset, args.size, args.seed)
    scored_categories = SCORED_CATEGORIES if args.dataset == TABLE else set(MICROCATEGORIES)
    rows = prepare_rows(records, 101)
    identities = [{"row_id": record["_row_id"], "source": record["_source"],
                   "description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                   "source_mc_id": row.inp.mc_id, "expected": sorted(row.expected & scored_categories)}
                  for record, row in zip(records, rows, strict=True)]
    protocol = {"dataset_table": args.dataset, "table_sha256": table_sha256(args.dataset),
                "sample_size": len(rows), "sample_seed": args.seed if args.size else None, "excluded_unlabeled_row_ids": excluded,
                "sample_sha256": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
                "scored_categories": sorted(scored_categories), "source_mc_id": "per-row original value",
                "label_interpretation": ("agreement with automatic labels; not independent human ground truth" if args.dataset == TABLE else "supplied combined corpus labels; predominantly overlaps training data"),
                "category_101": "unscored: table loader drops 101 from target labels" if args.dataset == TABLE else "source mcId=101 assumed because the JSONL has no source category",
                "source_category": "excluded from expected and predicted sets by contract"}
    report = {**metadata(), "scenario": "export_diagnostic" if args.dataset == TABLE else "igh_deduped_diagnostic", "storage": "postgresql",
              "inference_fingerprint": inference_fingerprint(),
              "protocol": protocol, "rows": identities, "results": {},
              "overlap": {table: audit_overlap(rows, table) for table in ("case_train", "manual_train")}}
    if args.report.exists():
        previous = json.loads(args.report.read_text())
        if (previous["protocol"] != protocol or previous["rows"] != identities
                or previous.get("inference_fingerprint") != report["inference_fingerprint"]):
            raise ValueError("Нельзя продолжить отчёт с другими входами, весами, настройками или протоколом")
        report = previous
    report["status"] = "running"
    reuse = json.loads(args.reuse_report.read_text()) if args.reuse_report else None
    if reuse is not None:
        if (reuse.get("inference_fingerprint") != report["inference_fingerprint"]
                or any(reuse["protocol"][key] != protocol[key] for key in ("dataset_table", "table_sha256", "scored_categories"))):
            raise ValueError("Нельзя переносить ответы другой модели, разметки или таблицы")
        report["reuse_report"] = str(args.reuse_report)
    if args.configurations:
        for key in args.configurations:
            configuration = BY_KEY[key]
            with configured(configuration):
                evaluate_method(configuration.module, rows, report, args.report, args.workers, reuse,
                                configuration_metadata(configuration))
    else:
        for method in args.methods or METHODS:
            evaluate_method(method, rows, report, args.report, args.workers, reuse)
    report["status"] = "complete" if all(r["status"] == "completed" for r in report["results"].values()) else "failed"
    save_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
