"""Micro-F1: сохранённые ответы по умолчанию, новые вызовы — через --methods."""

import argparse
import hashlib
import json
import os
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_support import metadata, replay_recorded_report, save_report, setup
from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from run import audit_overlap, calculate_metrics, normalize_description, prepare_rows, read_records

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
METHODS = tuple(sorted(path.parent.name for path in HERE.parent.parent.glob("*/pipeline.py")))
LOCAL_METHODS = {"tfidf", "semantic_encoder", "semantic_encoder_calibrated", "semantic_encoder_f1", "semantic_evidence"}
DATASETS = {
    "classifier_methods": ("docs/innoglobalhack_avito/public-test-dataset.parquet",
                           "docs/innoglobalhack_avito/public-train-annotated.jsonl"),
    "improved_dataset": ("docs/improved_dataset/reviewed_test.parquet", "docs/improved_dataset/train.jsonl"),
}
logger = logging.getLogger(__name__)


def _actual_training_audit(model_metadata: dict, rows: list) -> dict:
    """Проверить фактические source-строки модели, а не весь train сценария."""
    training = model_metadata.get("training", {})
    if "source" not in training or "source_row_numbers" not in training:
        return {"checked": False, "reason": "artifact does not record exact source rows"}
    source = (ROOT / training["source"]).resolve()
    if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest() != training["source_sha256"]:
        raise ValueError("Источник обучения изменился относительно артефакта")
    records = read_records(source)
    numbers = training["source_row_numbers"]
    if len(set(numbers)) != len(numbers) or any(type(n) is not int or not 1 <= n <= len(records) for n in numbers):
        raise ValueError("Некорректные source_row_numbers артефакта")
    texts = [records[n - 1]["description"] for n in numbers]
    known = {normalize_description(text) for text in texts}
    exact = [index for index, row in enumerate(rows, start=1) if normalize_description(row.inp.description) in known]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                max_features=60000, sublinear_tf=True)
    train_features = vectorizer.fit_transform(texts)
    test_features = vectorizer.transform([row.inp.description for row in rows])
    nearest = (test_features @ train_features.T).max(axis=1).toarray().ravel()
    return {
        "checked": True, "source": str(source.relative_to(ROOT)), "source_sha256": training["source_sha256"],
        "actual_training_rows": len(numbers), "normalized_exact_rows": exact,
        "nearest_train_cosine": nearest.tolist(), "near_threshold": 0.92,
        "near_rows": [index + 1 for index, value in enumerate(nearest) if value >= 0.92],
        "method": "exact normalized plus char_wb(3,5) cosine; not a semantic leakage guarantee",
        "evaluation_rows_removed": 0,
    }


def main(directory: Path = HERE, scenario: str = "classifier_methods") -> int:
    """Проверить старые ответы либо последовательно оценить выбранные модули."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--replay", action="store_true")
    selection.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--offline", action="store_true", help="Запретить вызовы API-модулей")
    parser.add_argument("--report", type=Path, default=directory / "results.json")
    parser.add_argument("--variant", help="Метка заранее зафиксированного варианта одного модуля")
    args = parser.parse_args()
    if args.variant and (not args.methods or len(args.methods) != 1):
        parser.error("--variant требует ровно один --methods")
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
    if args.report.exists():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if (previous.get("scenario") == scenario
                and previous.get("protocol", {}).get("dataset_sha256") == report["protocol"]["dataset_sha256"]
                and previous.get("rows") == report["rows"]):
            for name, result in previous.get("results", {}).items():
                result.setdefault("measurement_metadata", {
                    key: previous.get(key) for key in ("date", "commit", "code_sha256", "mode")
                })
                report["results"][name] = result
    for method in args.methods:
        key = f"{method}/{args.variant}" if args.variant else method
        result = {"status": "running", "module": method, "variant": args.variant, "predicted": [],
                  "measurement_metadata": {name: report[name] for name in ("date", "commit", "code_sha256", "mode")}}
        report["results"][key] = result
        try:
            if args.offline and method not in LOCAL_METHODS:
                raise ValueError("API-модуль недоступен с --offline; используйте --replay")
            pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
                subcategory_extraction=method, draft_generation=None,
            ))
            model_metadata = getattr(pipeline, "model_metadata", {})
            if method == "tfidf":
                from subcategory_extraction.tfidf.pipeline import DEFAULT_ARTIFACT
                artifact = Path(os.getenv("TFIDF_ARTIFACT_PATH", str(DEFAULT_ARTIFACT)))
                model_metadata = json.loads(artifact.read_text(encoding="utf-8"))
                model_metadata = {"training": model_metadata.get("training", {}),
                                  "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            result["model_metadata"] = model_metadata
            result["actual_training_overlap"] = _actual_training_audit(model_metadata, rows)
            durations = []
            for row in rows:
                started = perf_counter()
                output = pipeline.run(row.inp)
                durations.append(perf_counter() - started)
                if not isinstance(output, list) or any(
                    type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id
                    for cid in output
                ) or len(output) != len(set(output)):
                    raise ValueError("Неверный набор выходных микрокатегорий")
                result["predicted"].append(sorted(output))
            result.update(status="completed", metrics=calculate_metrics(
                [row.expected for row in rows], [set(ids) for ids in result["predicted"]],
            ))
            result["run_calls"] = len(durations)
            result["run_seconds"] = sum(durations)
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
