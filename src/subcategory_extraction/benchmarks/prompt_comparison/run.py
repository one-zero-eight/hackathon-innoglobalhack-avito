"""Сравнить актуальные модули и оба промпта Google на одной validation-выборке."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
from time import perf_counter

from benchmark_support import TRAIN, metadata, save_report, setup, timing, validation_sample
from main import PipelineConfig, load_pipeline
from subcategory_extraction.benchmarks.metrics import score


def main() -> int:
    """Выполнить настоящий инференс; неполные сценарии не получают итоговый F1."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    rows = validation_sample(args.size)
    report = {**metadata(), "dataset": str(TRAIN.relative_to(TRAIN.parents[2])),
              "dataset_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
              "split": "validation", "seed": 43, "rows": len(rows), "results": {}}
    config = PipelineConfig(subcategory_extraction="tfidf", draft_generation=None)
    for name in ("tfidf", "baseline", "guided"):
        durations, predictions, outputs = [], [], []
        result = {"status": "running", "rows": outputs}
        report["results"][name] = result
        config.subcategory_extraction = "tfidf" if name == "tfidf" else "google_decision"
        pipeline = load_pipeline("subcategory_extraction", config)
        if name != "tfidf":
            prompt_path = Path(__file__).resolve().parents[2] / f"google_decision/prompts/google_decision_{name}.md"
            pipeline.prompt = prompt_path.read_text(encoding="utf-8")
            result["prompt_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            result["model"] = pipeline.client.model_id
        else:
            result["model"] = "tfidf"
        started = perf_counter()
        for index, row in enumerate(rows, 1):
            call_started = perf_counter()
            try:
                predicted = set(pipeline.run(row["inp"]))
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              failed_row=index, failed_seconds=perf_counter() - call_started)
                logging.error("%s: ошибка на %d/%d: %s", name, index, len(rows), result["error"])
                break
            elapsed = perf_counter() - call_started
            durations.append(elapsed)
            predictions.append(predicted)
            outputs.append({"group_id": row["group_id"], "input": row["inp"].model_dump(by_alias=True),
                            "expected": sorted(row["targets"]), "predicted": sorted(predicted), "seconds": elapsed})
            result["response_time"] = timing(durations)
            save_report(args.report, report)
            logging.info("%s: %d/%d, %.3f с", name, index, len(rows), elapsed)
        result.update(response_time=timing(durations), elapsed_seconds=perf_counter() - started)
        if len(predictions) == len(rows):
            result.update(status="complete", metrics=score([row["targets"] for row in rows], predictions))
        save_report(args.report, report)
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"].values()) else "failed"
    save_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
