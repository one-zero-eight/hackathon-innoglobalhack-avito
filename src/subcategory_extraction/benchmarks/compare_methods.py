"""Единое сравнение заранее фиксированных методов на выбранном наборе."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from statistics import median
import threading
from time import perf_counter

from dotenv import load_dotenv

from main import LLMCostLedger, MICROCATEGORIES, PipelineConfig, load_pipeline, track_llm_costs
from run import audit_overlap, calculate_metrics, prepare_rows, read_records
from subcategory_extraction.google_decision.pipeline import PROMPT_DIR

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "docs/innoglobalhack_avito/public-test-dataset.parquet"
GOOGLE_RECORD = HERE / "results/google-guided-public.json"
METHODS = (
    "google_decision", "jev", "tfidf", "jev_tfidf", "jev_retrieval",
    "evidence_jev", "jev_arbiter", "semantic_encoder", "jev_boundaries",
)
logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Зафиксировать версию файла без чтения секретов и git diff."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def google_replay(path: Path, dataset: Path, rows: list, costs: LLMCostLedger) -> tuple[list[set[int]], dict]:
    """Применить реальные сохранённые вероятности только к совпавшим входам."""
    record = json.loads(path.read_text(encoding="utf-8"))
    prompt = PROMPT_DIR / "google_decision_guided.md"
    original_hash = digest(DATASET)
    if not isinstance(record, dict) or (
        record.get("model") != "gemini-decision-1" or record.get("threshold") != 0.5
        or record.get("dataset_sha256") != original_hash
        or record.get("prompt_sha256") != digest(prompt)
        or type(record.get("source_mc_id")) is not int or record["source_mc_id"] != 101
    ):
        raise ValueError("Google replay не соответствует исходным данным, модели или guided-промпту")
    original_rows = read_records(DATASET)
    saved_rows = record.get("rows")
    if not isinstance(saved_rows, list) or len(saved_rows) != len(original_rows):
        raise ValueError("Google replay: неверное число исходных записей")
    indexed = {}
    record_numbers = {}
    expected_categories = {str(cid) for cid in MICROCATEGORIES if cid != 101}
    for number, (original, saved) in enumerate(zip(original_rows, saved_rows, strict=True), start=1):
        description = original.get("description")
        source = original.get("mcId", 101)
        if not isinstance(description, str) or type(source) is not int or source != 101:
            raise ValueError("Google replay: неверный исходный вход")
        key = (hashlib.sha256(description.encode("utf-8")).hexdigest(), source)
        if not isinstance(saved, dict) or saved.get("description_sha256") != key[0]:
            raise ValueError("Google replay: изменился порядок/текст исходных объявлений")
        probabilities = saved.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected_categories or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in probabilities.values()
        ):
            raise ValueError("Google replay: неверные категории или вероятности")
        if key in indexed and indexed[key]["probabilities"] != probabilities:
            raise ValueError("Google replay: противоречивые вероятности одинаковых исходных входов")
        indexed.setdefault(key, saved)
        record_numbers.setdefault(key, []).append(number)
    keys = [
        (hashlib.sha256(row.inp.description.encode("utf-8")).hexdigest(), row.inp.mc_id)
        for row in rows
    ]
    if any(key not in indexed for key in keys):
        raise ValueError("Google replay: запрошенный точный текст/source отсутствует в исходной записи")
    predictions = []
    for key in keys:
        saved = indexed[key]
        predictions.append({int(cid) for cid, value in saved["probabilities"].items() if value >= 0.5})
        costs.record(record["model"], "https://generativelanguage.googleapis.com/v1beta/decisions",
                     saved.get("usage"), cache_hit=True)
    return predictions, {
        "mode": "genuine_recorded_api_replay", "record_sha256": digest(path),
        "model": record["model"], "prompt_sha256": record["prompt_sha256"],
        "original_dataset_sha256": original_hash, "evaluation_dataset_sha256": digest(dataset),
        "source_mc_id": record["source_mc_id"], "threshold": record["threshold"],
        "fresh_api_requests": 0, "recorded_rows": len(saved_rows), "evaluation_rows": len(rows),
        "unique_inputs": len(set(keys)), "cache_hits": len(rows),
        "matched_record_row_numbers": [record_numbers[key] for key in keys],
        "provenance": "genuine recorded API probabilities; exact input/source replay, not fresh calls; labels not reused",
    }


def observed_clients(pipeline: object) -> dict:
    """Учитывать только клиентов, реально вызванных на текущем объявлении."""
    clients = {}
    for name in ("client", "extractor"):
        client = getattr(pipeline, name, None)
        if client is not None:
            clients[name] = client
    for name in ("baseline", "jev"):
        parent = getattr(pipeline, name, None)
        client = getattr(parent, "client", None)
        if client is not None:
            clients[name] = client
    if getattr(pipeline, "last_arbiter_used", False):
        clients["arbiter"] = pipeline.arbiter
    return {
        name: {
            "model": client.model_id,
            "cache_hit": bool(getattr(client, "last_cache_hit", False)),
            "usage": getattr(client, "last_usage", None) or {},
        }
        for name, client in clients.items()
    }


def describe(pipeline: object) -> dict:
    """Сохранить существенные параметры реальной реализации, не веса и ключи."""
    metadata = {}
    for name in (
        "threshold", "weight", "uncertainty_low", "uncertainty_high",
        "reference_metadata", "model_metadata", "trained_categories", "unsupported_categories",
    ):
        value = getattr(pipeline, name, None)
        if value is not None:
            metadata[name] = sorted(value) if isinstance(value, (set, frozenset)) else value
    for name in ("client", "extractor", "arbiter"):
        client = getattr(pipeline, name, None)
        if client is not None:
            metadata[name] = {"model": client.model_id, "endpoint": client.endpoint}
    if hasattr(pipeline, "device"):
        metadata["device"] = str(pipeline.device)
    if isinstance(getattr(pipeline, "prompt", None), str):
        metadata["prompt_sha256"] = hashlib.sha256(pipeline.prompt.encode("utf-8")).hexdigest()
    return metadata


def evaluate_method(method: str, rows: list, workers: int, costs: LLMCostLedger) -> dict:
    """Предсказать уникальные входы один раз, но оценить каждую исходную строку."""
    config = PipelineConfig(subcategory_extraction=method, draft_generation=None)
    load_started = perf_counter()
    prototype = load_pipeline("subcategory_extraction", config)
    metadata = describe(prototype)
    model_load_seconds = perf_counter() - load_started
    unique = {}
    for row in rows:
        unique.setdefault((row.inp.description, row.inp.mc_id), row.inp)
    thread_state = threading.local()
    worker_count = 1 if method in ("semantic_encoder", "google_decision") else workers

    def infer(item: tuple) -> tuple:
        """Использовать отдельные mutable-счётчики клиентов на каждом потоке."""
        key, inp = item
        if worker_count == 1:
            pipeline = prototype
        else:
            if not hasattr(thread_state, "pipeline"):
                thread_state.pipeline = load_pipeline("subcategory_extraction", config)
            pipeline = thread_state.pipeline
        started = perf_counter()
        with track_llm_costs(costs):
            output = pipeline.run(inp)
        if not isinstance(output, list) or any(
            type(cid) is not int or cid not in MICROCATEGORIES or cid == inp.mc_id for cid in output
        ):
            raise ValueError(f"{method}: неверный выходной набор категорий")
        details = {
            "seconds": perf_counter() - started,
            "clients": observed_clients(pipeline),
            "routed_categories": getattr(pipeline, "last_routed_categories", []),
            "evidence_count": len(getattr(pipeline, "last_evidence", [])),
        }
        return key, (set(output), details)

    started = perf_counter()
    outputs = {}
    if worker_count == 1:
        for index, item in enumerate(unique.items(), start=1):
            key, value = infer(item)
            outputs[key] = value
            if index % 10 == 0 or index == len(unique):
                logger.info("%s: %d/%d уникальных входов", method, index, len(unique))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(infer, item) for item in unique.items()]
            try:
                for index, future in enumerate(as_completed(futures), start=1):
                    key, value = future.result()
                    outputs[key] = value
                    if index % 10 == 0 or index == len(unique):
                        logger.info("%s: %d/%d уникальных входов", method, index, len(unique))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    predictions = [outputs[(row.inp.description, row.inp.mc_id)][0] for row in rows]
    unique_details = [value[1] for value in outputs.values()]
    fresh = sum(not client["cache_hit"] for detail in unique_details for client in detail["clients"].values())
    cache_hits = sum(client["cache_hit"] for detail in unique_details for client in detail["clients"].values())
    durations = sorted(detail["seconds"] for detail in unique_details)
    return {
        "status": "completed", "metadata": metadata, "unique_inputs": len(unique),
        "workers": worker_count, "elapsed_seconds": perf_counter() - started,
        "fresh_api_requests": fresh, "cache_hits": cache_hits,
        "model_load_seconds": model_load_seconds,
        "latency_seconds": {
            "median": median(durations), "p95_nearest_rank": durations[math.ceil(0.95 * len(durations)) - 1],
            "mean": sum(durations) / len(durations), "max": durations[-1],
            "scope": "single unique input; includes first inference; excludes pipeline loading",
        },
        "metrics": calculate_metrics([row.expected for row in rows], predictions),
        "predicted": [sorted(value) for value in predictions],
        "unique_observations": [
            {"description_sha256": hashlib.sha256(key[0].encode()).hexdigest(),
             "source_mc_id": key[1], **value[1]}
            for key, value in outputs.items()
        ],
    }


def main() -> None:
    """Сравнить выбранные модули без создания validation или поиска параметров."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--training-dataset", type=Path, default=ROOT / "out/hybrid-training.jsonl")
    parser.add_argument("--google-record", type=Path, default=GOOGLE_RECORD)
    parser.add_argument("--google-live", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=ROOT / "out/classifier-methods.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers должен быть от 1 до 8")
    load_dotenv(ROOT / ".env")
    for name in ("JEV_THRESHOLD", "GOOGLE_DECISION_THRESHOLD", "JEV_TFIDF_THRESHOLD"):
        os.environ[name] = "0.5"
    os.environ["GOOGLE_DECISION_PROMPT"] = "guided"
    os.environ["JEV_PROMPT_PATH"] = str(PROMPT_DIR / "google_decision_guided.md")
    os.environ["JEV_TFIDF_WEIGHT"] = "0.5"
    os.environ["JEV_REFERENCE_DATASET"] = str(args.training_dataset.resolve())
    os.environ["JEV_REFERENCE_MAX_CHARS"] = "1800"
    rows = prepare_rows(read_records(args.dataset), 101)
    if args.report.resolve() in {args.dataset.resolve(), args.training_dataset.resolve(), args.google_record.resolve()}:
        raise ValueError("Отчёт не должен перезаписывать входные данные")
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset_path": str(args.dataset.resolve()),
            "training_dataset_path": str(args.training_dataset.resolve()),
            "dataset_sha256": digest(args.dataset), "training_sha256": digest(args.training_dataset),
            "rows": len(rows), "source_mc_id_assumption": 101,
            "threshold": 0.5, "validation_used": False, "test_parameter_search_used": False,
            "scope": "subcategory_extraction only; not a full 100-point competition result",
            "duplicate_policy": "identical inputs inferred once; every reference row scored",
        },
        "overlap": audit_overlap(rows, args.training_dataset),
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src").rglob("*.py"))},
        "rows": [{"row_number": index, "description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                  "source_mc_id": row.inp.mc_id, "expected": sorted(row.expected)}
                 for index, row in enumerate(rows, start=1)],
        "results": {},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        costs = LLMCostLedger()
        try:
            if method == "google_decision" and not args.google_live:
                predictions, metadata = google_replay(args.google_record, args.dataset, rows, costs)
                result = {"status": "completed", "metadata": metadata,
                          "metrics": calculate_metrics([row.expected for row in rows], predictions),
                          "predicted": [sorted(value) for value in predictions]}
            else:
                result = evaluate_method(method, rows, args.workers, costs)
            result["costs"] = costs.summary()
            logger.info("%s: API USD fresh=%s, cold=%s, попытки=%d, кэш=%d", method,
                        result["costs"]["fresh_estimated_usd"], result["costs"]["cold_estimated_usd"],
                        result["costs"]["api_attempts"], result["costs"]["local_cache_hits"])
            report["results"][method] = result
            metrics = result["metrics"]
            logger.info("%s: micro-F1 %.6f, баллы %.3f/60, TP=%d FP=%d FN=%d", method,
                        metrics["micro_f1"], metrics["classification_points"], metrics["tp"], metrics["fp"], metrics["fn"])
        except Exception as exc:
            report["results"][method] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "costs": costs.summary(),
            }
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.exception("%s: ошибка; фиктивная метрика не рассчитывается", method)
            raise
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
