"""Настоящая генерация и время run() на общей выборке исходных объявлений."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
from time import perf_counter

from benchmark_support import TRAIN, metadata, save_report, setup, timing, validation_sample
from main import PipelineConfig, PipelineOutput, load_pipeline


def main() -> int:
    """Создать один черновик на вход; сохранить полный текст для ручной оценки."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="llm")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("generation-results.json"))
    args = parser.parse_args()
    report = {**metadata(), "scenario": "live_generation", "module": args.module,
              "sample_seed": 44, "sample_size": args.size, "status": "running", "rows": []}
    durations = []
    try:
        rows = validation_sample(args.size, seed=44, nonempty=True)
        report["dataset_sha256"] = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
        report["sample_groups"] = [row["group_id"] for row in rows]
        pipeline = load_pipeline("draft_generation", PipelineConfig(
            subcategory_extraction=None, draft_generation=args.module,
        ))
        report["model"] = pipeline.client.model_id
        settings = getattr(pipeline.client, "settings", None)
        if settings is not None:
            report["api_settings"] = settings.model_dump()
        prompt_path = Path(__file__).resolve().parents[2] / args.module / "prompts/draft.md"
        report["prompt_path"] = str(prompt_path.relative_to(TRAIN.parents[2]))
        report["prompt_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if not pipeline.client.api_key:
            raise ValueError("Не задан API-ключ; настоящие вызовы не выполнялись")
        for index, row in enumerate(rows, 1):
            targets = [min(row["targets"])]
            started = perf_counter()
            try:
                drafts = pipeline.run(row["inp"], targets)
                validated = PipelineOutput(drafts=drafts)
                if [draft.mc_id for draft in validated.drafts] != targets:
                    raise ValueError("Неверное число или категории черновиков")
            except Exception:
                report.update(failed_row=index, failed_seconds=perf_counter() - started)
                raise
            seconds = perf_counter() - started
            durations.append(seconds)
            report["rows"].append({
                "group_id": row["group_id"], "input": row["inp"].model_dump(by_alias=True),
                "target_mc_ids": targets, "drafts": drafts, "seconds": seconds,
            })
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
    logging.info("Генерация: %s, %s", report["status"], report.get("error", report["response_time"]))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
