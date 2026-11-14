"""Измерить задержку настоящего run() у TF-IDF и Google Decision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing, validation_sample
from main import PipelineConfig, load_pipeline


def main() -> int:
    """Измерить одинаковые входы без прогрева, исключая создание экземпляра."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--module", choices=("all", "tfidf", "google_decision"), default="all")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    rows = validation_sample(args.size, seed=44)
    report = {**metadata(), "sample_seed": 44, "sample_size": len(rows), "results": {}}
    modules = ("tfidf", "google_decision") if args.module == "all" else (args.module,)
    for module in modules:
        pipeline = load_pipeline("subcategory_extraction", PipelineConfig(subcategory_extraction=module, draft_generation=None))
        result = {"status": "running", "model": getattr(getattr(pipeline, "client", None), "model_id", module), "rows": []}
        report["results"][module] = result
        durations = []
        for row in rows:
            started = perf_counter()
            try:
                output = pipeline.run(row["inp"])
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_seconds=perf_counter() - started)
                break
            elapsed = perf_counter() - started
            durations.append(elapsed)
            result["rows"].append({"group_id": row["group_id"], "seconds": elapsed, "predicted": output})
            save_report(args.report, report)
            logging.info("%s: %d/%d, %.3f с", module, len(durations), len(rows), elapsed)
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
