"""Измерить задержку настоящего run() у всех конфигураций классификатора."""

import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import json
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing, validation_sample
from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from subcategory_extraction.benchmarks.classifier_methods.run import METHODS
from subcategory_extraction.benchmarks.configurations import (
    BY_KEY, CONFIGURATIONS, Configuration, client_observations,
    configuration_metadata, configured,
)


def main() -> int:
    """Измерить одинаковые входы без прогрева, исключая создание экземпляра."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--module", choices=METHODS)
    selection.add_argument("--configuration", choices=BY_KEY)
    selection.add_argument("--all-configurations", action="store_true")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--append", action="store_true", help="Сохранить другие замеры с исходной датой")
    args = parser.parse_args()
    rows = validation_sample(args.size, seed=44)
    report = {**metadata(), "sample_seed": 44, "sample_size": len(rows), "results": {}}
    if args.report.name == "recorded-results.json":
        parser.error("Нельзя перезаписать исторические ответы")
    if args.append and args.report.is_file():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if previous.get("sample_seed") != report["sample_seed"] or previous.get("sample_size") != report["sample_size"]:
            raise ValueError("Нельзя объединить замеры на разных выборках")
        for module, item in previous["results"].items():
            item.setdefault("measurement_metadata", {
                key: previous.get(key) for key in ("date", "commit", "code_sha256", "timing_scope")
            })
            report["results"][module] = item
    configurations = (list(CONFIGURATIONS) if args.all_configurations else
                      [BY_KEY[args.configuration]] if args.configuration else
                      [Configuration(args.module or "tfidf", args.module or "tfidf")])
    matrix_mode = bool(args.configuration or args.all_configurations)
    for configuration in configurations:
        module = configuration.module
        with configured(configuration, fresh=True) if matrix_mode else nullcontext():
            result = {"status": "running", "model": module, "rows": []}
            if matrix_mode:
                result["configuration"] = configuration_metadata(configuration)
            result["measurement_metadata"] = {key: report[key] for key in ("date", "commit", "code_sha256", "machine", "environment")}
            report["results"][configuration.key] = result
            try:
                pipeline = load_pipeline("subcategory_extraction", PipelineConfig(subcategory_extraction=module, draft_generation=None))
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}", response_time={"calls": 0})
                save_report(args.report, report)
                continue
            result["model"] = getattr(getattr(pipeline, "client", None), "model_id", module)
            describe = getattr(pipeline, "benchmark_metadata", None)
            if callable(describe):
                result["model_provenance"] = describe()
            result["timing_scope"] = {
                "loading_included": False, "first_call_included": True,
                "cache_enabled": getattr(getattr(pipeline, "client", None), "cache_enabled", None),
                "quota_wait_and_retries_included_if_any": True,
                "warmup": False, "fresh_client_caches": matrix_mode,
                "p95_method": "numpy.percentile linear interpolation",
            }
            durations = []
            for row in rows:
                started = perf_counter()
                try:
                    output = pipeline.run(row["inp"])
                    if not isinstance(output, list) or len(output) != len(set(output)) or any(
                        type(cid) is not int or cid not in MICROCATEGORIES or cid == row["inp"].mc_id
                        for cid in output
                    ):
                        raise ValueError("Неверный набор выходных микрокатегорий")
                except Exception as exc:
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_seconds=perf_counter() - started)
                    break
                elapsed = perf_counter() - started
                durations.append(elapsed)
                client = getattr(pipeline, "client", None)
                result["rows"].append({"group_id": row["group_id"], "seconds": elapsed, "predicted": output,
                                       "cache_hit": getattr(client, "last_cache_hit", None),
                                       "http_attempts": getattr(client, "last_attempts", None),
                                       "usage": getattr(client, "last_usage", None),
                                       "api_clients": client_observations(pipeline)})
                save_report(args.report, report)
                logging.info("%s: %d/%d, %.3f с", configuration.key, len(durations), len(rows), elapsed)
            result["response_time"] = timing(durations)
            if len(durations) == len(rows):
                result["status"] = "complete"
            save_report(args.report, report)
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"].values()) else "failed"
    save_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
