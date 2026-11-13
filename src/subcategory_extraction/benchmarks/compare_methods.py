"""Единое сравнение заранее фиксированных методов на полном официальном наборе."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
from time import perf_counter

from dotenv import load_dotenv

from main import MICROCATEGORIES, PipelineConfig, load_pipeline
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


def google_replay(path: Path, dataset: Path, rows: list) -> tuple[list[set[int]], dict]:
    """Применить реальные сохранённые вероятности только к совпавшим входам."""
    record = json.loads(path.read_text(encoding="utf-8"))
    prompt = PROMPT_DIR / "google_decision_guided.md"
    if (
        record["model"] != "gemini-decision-1" or record["threshold"] != 0.5
        or record["dataset_sha256"] != digest(dataset)
        or record["prompt_sha256"] != digest(prompt)
        or record["source_mc_id"] != 101 or len(record["rows"]) != len(rows)
    ):
        raise ValueError("Google replay не соответствует данным, модели или guided-промпту")
    predictions = []
    for row, saved in zip(rows, record["rows"], strict=True):
        if saved["description_sha256"] != hashlib.sha256(row.inp.description.encode()).hexdigest():
            raise ValueError("Google replay: изменился порядок/текст объявлений")
        probabilities = {int(cid): value for cid, value in saved["probabilities"].items()}
        if set(probabilities) != set(MICROCATEGORIES) - {row.inp.mc_id} or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in probabilities.values()
        ):
            raise ValueError("Google replay: неверные категории или вероятности")
        predictions.append({cid for cid, value in probabilities.items() if value >= 0.5})
    return predictions, {
        "mode": "genuine_recorded_api_replay", "record_sha256": digest(path),
        "model": record["model"], "prompt_sha256": record["prompt_sha256"],
        "fresh_api_requests": 0, "recorded_rows": len(rows),
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
    return metadata


def evaluate_method(method: str, rows: list, workers: int) -> dict:
    """Предсказать уникальные входы один раз, но оценить каждую исходную строку."""
    config = PipelineConfig(subcategory_extraction=method, draft_generation=None)
    prototype = load_pipeline("subcategory_extraction", config)
    metadata = describe(prototype)
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
    return {
        "status": "completed", "metadata": metadata, "unique_inputs": len(unique),
        "workers": worker_count, "elapsed_seconds": perf_counter() - started,
        "fresh_api_requests": fresh, "cache_hits": cache_hits,
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
    os.environ["JEV_TFIDF_WEIGHT"] = "0.5"
    os.environ["JEV_REFERENCE_DATASET"] = str(args.training_dataset.resolve())
    os.environ["JEV_REFERENCE_MAX_CHARS"] = "1800"
    rows = prepare_rows(read_records(args.dataset), 101)
    if args.report.resolve() in {args.dataset.resolve(), args.training_dataset.resolve(), args.google_record.resolve()}:
        raise ValueError("Отчёт не должен перезаписывать входные данные")
    report = {
        "protocol": {
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
        try:
            if method == "google_decision" and not args.google_live:
                predictions, metadata = google_replay(args.google_record, args.dataset, rows)
                result = {"status": "completed", "metadata": metadata,
                          "metrics": calculate_metrics([row.expected for row in rows], predictions),
                          "predicted": [sorted(value) for value in predictions]}
            else:
                result = evaluate_method(method, rows, args.workers)
            report["results"][method] = result
            metrics = result["metrics"]
            logger.info("%s: micro-F1 %.6f, баллы %.3f/60, TP=%d FP=%d FN=%d", method,
                        metrics["micro_f1"], metrics["classification_points"], metrics["tp"], metrics["fp"], metrics["fn"])
        except Exception as exc:
            report["results"][method] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.exception("%s: ошибка; фиктивная метрика не рассчитывается", method)
            raise
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
