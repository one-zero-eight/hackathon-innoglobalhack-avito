"""Измерить время настоящего run() выбранного модуля выделения."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
from time import perf_counter

from benchmark_support import TRAIN, metadata, save_report, setup, timing, validation_sample
from main import PipelineConfig, load_pipeline


def main() -> int:
    """Измерить общую выборку без прогрева и загрузки экземпляра в таймере."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--module", default="google_decision")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("extraction-results.json"))
    args = parser.parse_args()
    report = {**metadata(), "scenario": "live_extraction_latency", "module": args.module,
              "sample_seed": 44, "sample_size": args.size, "status": "running", "rows": []}
    durations = []
    try:
        rows = validation_sample(args.size, seed=44)
        report["dataset_sha256"] = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
        report["sample_groups"] = [row["group_id"] for row in rows]
        pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
            subcategory_extraction=args.module, draft_generation=None,
        ))
        report["model"] = pipeline.client.model_id
        report["threshold"] = pipeline.threshold
        report["prompt_sha256"] = hashlib.sha256(pipeline.prompt_path.read_bytes()).hexdigest()
        report["api_settings"] = pipeline.client.settings.model_dump()
        if not pipeline.client.api_key:
            raise ValueError("Не задан API-ключ; настоящие вызовы не выполнялись")
        for index, row in enumerate(rows, 1):
            started = perf_counter()
            try:
                predicted = pipeline.run(row["inp"])
            except Exception:
                report.update(failed_row=index, failed_seconds=perf_counter() - started)
                raise
            seconds = perf_counter() - started
            durations.append(seconds)
            report["rows"].append({"group_id": row["group_id"], "predicted": predicted, "seconds": seconds})
            report["response_time"] = timing(durations)
            save_report(args.report, report)
            logging.info("%s: %d/%d, %.3f с", args.module, index, len(rows), seconds)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        report.update(status="failed", error="Прогон прерван")
    report["response_time"] = timing(durations)
    report["timing_includes"] = "run(), API retries and quota waits; excludes instance construction and data loading"
    save_report(args.report, report)
    logging.info("Выделение: %s, %s", report["status"], report.get("error", report["response_time"]))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
