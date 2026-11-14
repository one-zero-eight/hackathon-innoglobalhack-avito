"""Micro-F1: сохранённые ответы по умолчанию, новые вызовы — через --methods."""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_support import metadata, replay_recorded_report, save_report, setup
from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from run import audit_overlap, calculate_metrics, prepare_rows, read_records

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
METHODS = tuple(sorted(path.parent.name for path in HERE.parent.parent.glob("*/pipeline.py")))
LOCAL_METHODS = {"tfidf", "semantic_encoder", "semantic_encoder_calibrated", "semantic_encoder_f1"}
DATASETS = {
    "classifier_methods": ("docs/innoglobalhack_avito/public-test-dataset.parquet",
                           "docs/innoglobalhack_avito/public-train-annotated.jsonl"),
    "improved_dataset": ("docs/improved_dataset/reviewed_test.parquet", "docs/improved_dataset/train.jsonl"),
}
logger = logging.getLogger(__name__)


def main(directory: Path = HERE, scenario: str = "classifier_methods") -> int:
    """Проверить старые ответы либо последовательно оценить выбранные модули."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--replay", action="store_true")
    selection.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--offline", action="store_true", help="Запретить вызовы API-модулей")
    parser.add_argument("--report", type=Path, default=directory / "results.json")
    args = parser.parse_args()
    if not args.methods:
        replay_recorded_report(directory, scenario, args.report)
        return 0
    dataset, training = (ROOT / name for name in DATASETS[scenario])
    if args.report.resolve() in {dataset.resolve(), training.resolve(), (directory / "recorded-results.json").resolve()}:
        parser.error("Отчёт не должен перезаписывать данные или сохранённые ответы")
    setup()
    rows = prepare_rows(read_records(dataset), 101)
    report = {
        **metadata(), "status": "running", "scenario": scenario, "mode": "pipeline_run",
        "protocol": {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                     "training_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                     "rows": len(rows), "source_mc_id_assumption": 101},
        "overlap": audit_overlap(rows, training),
        "rows": [{"description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                  "source_mc_id": row.inp.mc_id, "expected": sorted(row.expected)} for row in rows],
        "results": {},
    }
    for method in args.methods:
        result = {"status": "running", "predicted": []}
        report["results"][method] = result
        try:
            if args.offline and method not in LOCAL_METHODS:
                raise ValueError("API-модуль недоступен с --offline; используйте --replay")
            pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
                subcategory_extraction=method, draft_generation=None,
            ))
            for row in rows:
                output = pipeline.run(row.inp)
                if not isinstance(output, list) or any(
                    type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id
                    for cid in output
                ) or len(output) != len(set(output)):
                    raise ValueError("Неверный набор выходных микрокатегорий")
                result["predicted"].append(sorted(output))
            result.update(status="completed", metrics=calculate_metrics(
                [row.expected for row in rows], [set(ids) for ids in result["predicted"]],
            ))
            logger.info("%s: N=%d, micro-F1=%.6f", method, len(rows), result["metrics"]["micro_f1"])
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logger.exception("%s: прогон не завершён, F1 не рассчитан", method)
        save_report(args.report, report)
    report["status"] = "complete" if all(
        result["status"] == "completed" for result in report["results"].values()
    ) else "failed"
    save_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
