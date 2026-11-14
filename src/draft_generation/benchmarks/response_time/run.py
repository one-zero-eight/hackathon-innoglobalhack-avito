"""Измерить время реальной генерации одного черновика на объявление."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing, validation_sample
from main import PipelineConfig, PipelineOutput, load_pipeline
from draft_generation.sample_llm.pipeline import PROMPT_PATH


def main() -> int:
    """Создать по одному черновику для минимального эталонного mcId на 10 входах."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--offline", action="store_true", help="Записать отсутствие свежего API-замера")
    args = parser.parse_args()
    if args.offline:
        save_report(args.report, {
            **metadata(), "status": "unavailable", "module": "sample_llm",
            "reason": "Внешний API отключён: свежие вызовы и latency не измерялись",
            "fresh_api_requests": 0, "response_time": {"calls": 0},
        })
        return 1
    rows = validation_sample(args.size, seed=44, nonempty=True)
    pipeline = load_pipeline("draft_generation", PipelineConfig(subcategory_extraction=None, draft_generation="sample_llm"))
    report = {**metadata(), "sample_seed": 44, "sample_size": len(rows), "module": "sample_llm",
              "model": pipeline.client.model_id, "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
              "status": "running", "rows": []}
    durations = []
    for row in rows:
        targets = [min(row["targets"])]
        started = perf_counter()
        try:
            drafts = pipeline.run(row["inp"], targets)
            validated = PipelineOutput(drafts=drafts)
            if [draft.mc_id for draft in validated.drafts] != targets:
                raise ValueError("Неверное число или категории черновиков")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_seconds=perf_counter() - started)
            break
        elapsed = perf_counter() - started
        durations.append(elapsed)
        report["rows"].append({"group_id": row["group_id"], "input": row["inp"].model_dump(by_alias=True),
                               "target_mc_ids": targets, "drafts": drafts, "seconds": elapsed})
        save_report(args.report, report)
        logging.info("sample_llm: %d/%d, %.3f с", len(durations), len(rows), elapsed)
    report["response_time"] = timing(durations)
    if len(durations) == len(rows):
        report["status"] = "complete"
    save_report(args.report, report)
    logging.info("Генерация: %s, %s", report["status"], report["response_time"])
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
