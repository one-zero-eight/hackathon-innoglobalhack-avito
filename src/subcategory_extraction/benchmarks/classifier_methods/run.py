"""Micro-F1: сохранённые ответы по умолчанию, новые вызовы — через --methods."""

import argparse
from contextlib import nullcontext
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
from subcategory_extraction.benchmarks.configurations import (
    BY_KEY, CONFIGURATIONS, LOCAL_METHODS, Configuration, client_observations,
    configuration_metadata, configured,
)
from main import BasePipeline, MICROCATEGORIES, PipelineConfig, load_pipeline
from run import EvaluationRow, audit_overlap, calculate_metrics, prepare_rows, read_records

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
METHODS = tuple(sorted(".".join(path.parent.relative_to(HERE.parent.parent).parts) for path in HERE.parent.parent.glob("*/*/pipeline.py")))
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
            "api_clients": client_observations(active),
        }

    if workers == 1:
        yield from map(predict, rows)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(predict, rows)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def reusable_predictions(path: Path | None, configuration: dict, rows: list[EvaluationRow]) -> dict[int, tuple[list[int], dict]]:
    """Переиспользовать совпавшие входы той же конфигурации без переноса меток."""
    if path is None:
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    result = source["results"].get(configuration["key"], {})
    if result.get("status") not in {"complete", "completed"} or result.get("configuration") != configuration:
        return {}
    source_rows = source.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != len(result.get("predicted", [])):
        return {}
    saved = {}
    for row, output, inference in zip(source_rows, result["predicted"], result["inference"], strict=True):
        identity = (row["description_sha256"], row["source_mc_id"])
        saved[identity] = (output, {**inference, "reused_prediction": True, "reuse_report": str(path)})
    identities = [(hashlib.sha256(row.inp.description.encode()).hexdigest(), row.inp.mc_id) for row in rows]
    return {index: saved[identity] for index, identity in enumerate(identities) if identity in saved}


def main(directory: Path = HERE, scenario: str = "classifier_methods") -> int:
    """Проверить старые ответы либо оценить модули с явным числом потоков."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--replay", action="store_true")
    selection.add_argument("--methods", nargs="+", choices=METHODS)
    selection.add_argument("--configuration", choices=BY_KEY)
    selection.add_argument("--all-configurations", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Запретить вызовы API-модулей")
    parser.add_argument("--append", action="store_true", help="Сохранить результаты других модулей с их датами")
    parser.add_argument("--report", type=Path, default=directory / "results.json")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--reuse-report", type=Path, help="Пересчитать F1 по совпадающим входам и конфигурации; не замер latency")
    args = parser.parse_args()
    if not (args.methods or args.configuration or args.all_configurations):
        replay_recorded_report(directory, scenario, args.report)
        return 0
    configurations = (list(CONFIGURATIONS) if args.all_configurations else
                      [BY_KEY[args.configuration]] if args.configuration else
                      [Configuration(method.rsplit(".", 1)[-1], method) for method in args.methods])
    matrix_mode = bool(args.configuration or args.all_configurations)
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
                     "dataset_sha256": table_sha256(dataset), "table_sha256": table_sha256(dataset),
                     "training_sha256": table_sha256(training),
                     "rows": len(rows), "source_mc_id_assumption": 101},
        "overlap": audit_overlap(rows, training),
        "rows": [{"description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                  "source_mc_id": row.inp.mc_id, "expected": sorted(row.expected)} for row in rows],
        "results": {},
    }
    if args.append and args.report.is_file():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        previous_protocol = previous.get("protocol", {})
        if previous_protocol.get("table_sha256", previous_protocol.get("dataset_sha256")) != report["protocol"]["dataset_sha256"] or previous.get("rows") not in (report["rows"], len(rows)):
            raise ValueError("Нельзя объединить отчёты с разными входами/метками")
        for method, item in previous["results"].items():
            item.setdefault("measurement_metadata", {
                key: previous.get(key) for key in ("date", "commit", "code_sha256", "mode", "protocol", "overlap")
            })
            report["results"][method] = item
    for configuration in configurations:
        method = configuration.module
        result = {"status": "running", "predicted": [], "inference": [], "workers": args.workers}
        report["results"][configuration.key] = result
        try:
            with configured(configuration) if matrix_mode else nullcontext():
                if matrix_mode:
                    result["configuration"] = configuration_metadata(configuration)
                if configuration.training_table:
                    result["model_training_overlap"] = audit_overlap(rows, configuration.training_table)
                if args.offline and method not in LOCAL_METHODS:
                    raise ValueError("API-модуль недоступен с --offline; используйте --replay")
                reused = reusable_predictions(args.reuse_report, result.get("configuration", {}), rows) if matrix_mode else {}
                pending = [row for index, row in enumerate(rows) if index not in reused]
                pipeline = None if not pending else load_pipeline("subcategory_extraction", PipelineConfig(
                    subcategory_extraction=method, draft_generation=None,
                ))
                result["execution"] = "saved_predictions_rescored" if not pending else "mixed_reuse_and_pipeline_run" if reused else "pipeline_run"
                result["reused_predictions"] = len(reused)
                fresh = iter(predictions(pipeline, method, pending, args.workers)) if pending else iter(())
                outputs = (reused[index] if index in reused else next(fresh) for index in range(len(rows)))
                describe = getattr(pipeline, "benchmark_metadata", None)
                if callable(describe):
                    result["model_provenance"] = describe()
                    actual_train = result["model_provenance"].get("training_dataset")
                    if actual_train:
                        result["model_training_overlap"] = audit_overlap(rows, actual_train)
                for index, (row, (output, inference)) in enumerate(zip(
                    rows, outputs, strict=True,
                ), start=1):
                    if not isinstance(output, list) or any(
                        type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id
                        for cid in output
                    ) or len(output) != len(set(output)):
                        raise ValueError("Неверный набор выходных микрокатегорий")
                    result["predicted"].append(sorted(output))
                    result["inference"].append(inference)
                    if index % 20 == 0:
                        logger.info("%s: %d/%d", configuration.key, index, len(rows))
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
                logger.info("%s: N=%d, micro-F1=%.6f", configuration.key, len(rows), result["metrics"]["micro_f1"])
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
