"""Micro-F1: сохранённые ответы по умолчанию, новые вызовы — через --methods."""

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_support import metadata, replay_recorded_report, save_report, setup
from database import table_sha256
from main import BasePipeline, MICROCATEGORIES, PipelineConfig, load_pipeline
from run import EvaluationRow, audit_overlap, calculate_metrics, prepare_rows, read_records

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
METHODS = tuple(sorted(path.parent.name for path in HERE.parent.parent.glob("*/pipeline.py")))
LOCAL_METHODS = {"tfidf", "learned_keywords", "semantic_encoder", "semantic_encoder_calibrated", "semantic_encoder_f1", "snippet_logistic_regression"}
DATASETS = {
    # Имена таблиц PostgreSQL (единый формат в src/database.py).
    "classifier_methods": ("case_test", "case_train"),
    "improved_dataset": ("manual_test", "manual_train"),
}
logger = logging.getLogger(__name__)


def predictions(pipeline: BasePipeline, method: str, rows: list[EvaluationRow], workers: int) -> Iterator[tuple[list[int], dict]]:
    """Вызвать run() с отдельным экземпляром на поток и сохранить порядок входов."""
    state = local()

    def predict(row: EvaluationRow) -> tuple[list[int], dict]:
        """Не смешивать вероятности и usage параллельных клиентов."""
        if workers == 1:
            active = pipeline
        else:
            if not hasattr(state, "pipeline"):
                state.pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
                    subcategory_extraction=method, draft_generation=None,
                ))
            active = state.pipeline
        output = active.run(row.inp)
        client = getattr(active, "client", None)
        return output, {
            "cache_hit": getattr(client, "last_cache_hit", None),
            "http_attempts": getattr(client, "last_attempts", None),
            "usage": getattr(client, "last_usage", None),
            "probabilities": getattr(active, "last_probabilities", None),
        }

    if workers == 1:
        yield from map(predict, rows)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(predict, rows)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main(directory: Path = HERE, scenario: str = "classifier_methods") -> int:
    """Проверить старые ответы либо оценить модули с явным числом потоков."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--replay", action="store_true")
    selection.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--offline", action="store_true", help="Запретить вызовы API-модулей")
    parser.add_argument("--append", action="store_true", help="Сохранить результаты других модулей с их датами")
    parser.add_argument("--report", type=Path, default=directory / "results.json")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if not args.methods:
        replay_recorded_report(directory, scenario, args.report)
        return 0
    dataset, training = DATASETS[scenario]
    if not dataset or not training:
        parser.error("Не заданы таблицы датасета и обучения для сценария")
    if args.report.resolve() == (directory / "recorded-results.json").resolve():
        parser.error("Отчёт не должен перезаписывать сохранённые ответы")
    setup()
    rows = prepare_rows(read_records(dataset), 101)
    report = {
        **metadata(), "status": "running", "scenario": scenario, "mode": "pipeline_run",
        "storage": "postgresql",
        "protocol": {"dataset_table": dataset, "training_table": training,
                     "dataset_sha256": table_sha256(dataset),
                     "training_sha256": table_sha256(training),
                     "rows": len(rows), "source_mc_id_assumption": 101},
        "overlap": audit_overlap(rows, training),
        "rows": [{"description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                  "source_mc_id": row.inp.mc_id, "expected": sorted(row.expected)} for row in rows],
        "results": {},
    }
    if args.append and args.report.is_file():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if previous.get("protocol", {}).get("dataset_sha256") != report["protocol"]["dataset_sha256"] or previous.get("rows") != report["rows"]:
            raise ValueError("Нельзя объединить отчёты с разными входами/метками")
        for method, item in previous["results"].items():
            item.setdefault("measurement_metadata", {
                key: previous.get(key) for key in ("date", "commit", "code_sha256", "mode", "protocol", "overlap")
            })
            report["results"][method] = item
    for method in args.methods:
        result = {"status": "running", "predicted": [], "inference": [], "workers": args.workers}
        report["results"][method] = result
        try:
            if args.offline and method not in LOCAL_METHODS:
                raise ValueError("API-модуль недоступен с --offline; используйте --replay")
            pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
                subcategory_extraction=method, draft_generation=None,
            ))
            describe = getattr(pipeline, "benchmark_metadata", None)
            if callable(describe):
                result["model_provenance"] = describe()
                actual_train = result["model_provenance"].get("training_dataset")
                if actual_train:
                    result["model_training_overlap"] = audit_overlap(rows, actual_train)
            for index, (row, (output, inference)) in enumerate(zip(
                rows, predictions(pipeline, method, rows, args.workers), strict=True,
            ), start=1):
                if not isinstance(output, list) or any(
                    type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id
                    for cid in output
                ) or len(output) != len(set(output)):
                    raise ValueError("Неверный набор выходных микрокатегорий")
                result["predicted"].append(sorted(output))
                result["inference"].append(inference)
                if index % 20 == 0:
                    logger.info("%s: %d/%d", method, index, len(rows))
            result.update(status="completed", metrics=calculate_metrics(
                [row.expected for row in rows], [set(ids) for ids in result["predicted"]],
            ))
            if all(isinstance(item["probabilities"], dict) for item in result["inference"]):
                result["fixed_threshold_metrics"] = {
                    "mode": "same_recorded_probabilities_with_fixed_0.5_not_a_new_model_run",
                    "metrics": calculate_metrics([row.expected for row in rows], [
                        {int(cid) for cid, probability in item["probabilities"].items() if probability >= 0.5}
                        for item in result["inference"]
                    ]),
                }
            logger.info("%s: N=%d, micro-F1=%.6f", method, len(rows), result["metrics"]["micro_f1"])
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logger.exception("%s: прогон не завершён, F1 не рассчитан", method)
        result["measurement_metadata"] = {
            key: report[key] for key in ("date", "commit", "code_sha256", "mode", "protocol", "overlap")
        }
        save_report(args.report, report)
    report["status"] = "complete" if all(
        result["status"] in {"completed", "complete"} for result in report["results"].values()
    ) else "failed"
    save_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
