"""Итоговый classification-only F1 текущего extractor на официальном test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing
from main import Draft, PipelineOutput, load_config, load_pipeline
from run import CLASSIFICATION_POINTS, DEFAULT_RUN_CONFIG, RunConfig, audit_overlap, calculate_metrics, prepare_rows, read_records, resolve_path


def main() -> int:
    """Оценить только выделение; не выдавать этот сценарий за полный пайплайн."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = RunConfig.model_validate_json(config_path.read_text())
    base = config_path.parent
    pipeline_path = resolve_path(base, config.pipeline_config)
    selected = load_config(pipeline_path)
    dataset = resolve_path(base, config.dataset)
    training = resolve_path(base, config.training_dataset) if config.training_dataset else None
    records = read_records(dataset)
    if config.limit is not None:
        records = records[:config.limit]
    rows = prepare_rows(records, config.source_mc_id)
    pipeline = load_pipeline("subcategory_extraction", selected)
    settings = {
        "module": selected.subcategory_extraction, **pipeline.client.settings.model_dump(),
        "prompt": pipeline.settings.prompt, "threshold": pipeline.threshold,
        "prompt_sha256": hashlib.sha256(pipeline.prompt.encode()).hexdigest(),
        "guidance_sha256": hashlib.sha256(json.dumps(pipeline.guidance, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }
    run_metadata = metadata()
    fingerprint = hashlib.sha256(
        dataset.read_bytes() + json.dumps(settings, sort_keys=True).encode()
        + run_metadata["code_sha256"].encode() + str(config.source_mc_id).encode()
        + str(config.limit).encode()
    ).hexdigest()
    if args.resume:
        report = json.loads(args.report.read_text())
        if report.get("fingerprint") != fingerprint:
            raise ValueError("Нельзя resume: изменились код, данные или настройки выделения")
        report.pop("metrics", None)
    else:
        report = {
            **run_metadata, "scenario": "classification_only_official_test", "fingerprint": fingerprint,
            "dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "rows_total": len(rows), "limited": config.limit is not None,
            "model_settings": settings, "overlap": audit_overlap(rows, training),
            "scoring": {"source": "docs/Критерии авито.pdf", "average": "micro", "zero_division": 0,
                        "classification_max_points": CLASSIFICATION_POINTS},
            "generator_executed": False, "end_to_end_pipeline_score": False,
            "results": [], "sessions": [],
        }
    report.update(status="running")
    report.pop("error", None)
    session = {**run_metadata, "resumed": args.resume, "starting_row": len(report["results"]) + 1}
    report["sessions"].append(session)
    save_report(args.report, report)
    started = perf_counter()
    try:
        if len(report["results"]) > len(rows):
            raise ValueError("Контрольная точка длиннее датасета")
        for index, row in enumerate(rows, 1):
            if index <= len(report["results"]):
                saved = report["results"][index - 1]
                output = PipelineOutput.model_validate(saved["output"])
                if saved["row_number"] != index or saved["expected"] != sorted(row.expected):
                    raise ValueError("Контрольная точка не соответствует исходной строке")
                if saved["predicted"] != sorted({draft.mc_id for draft in output.drafts}):
                    raise ValueError("Предсказания не соответствуют сохранённым drafts")
                continue
            call_started = perf_counter()
            try:
                targets = pipeline.run(row.inp)
                if any(type(cid) is not int for cid in targets):
                    raise ValueError("Неверный тип mcId")
                # Нулевые тексты обозначают отсутствие генерации, а не её успешный вызов.
                output = PipelineOutput(drafts=[Draft(mcId=cid) for cid in targets])
            except Exception:
                report["failed_seconds"] = perf_counter() - call_started
                raise
            seconds = perf_counter() - call_started
            report["results"].append({
                "row_number": index, "expected": sorted(row.expected),
                "predicted": sorted({draft.mc_id for draft in output.drafts}),
                "output": output.model_dump(by_alias=True), "seconds": seconds,
            })
            report.update(completed_rows=index, response_time=timing([item["seconds"] for item in report["results"]]))
            save_report(args.report, report)
            logging.info("Classification-only: %d/%d, %.3f с", index, len(rows), seconds)
        report.update(status="complete", metrics=calculate_metrics(
            [set(item["expected"]) for item in report["results"]],
            [set(item["predicted"]) for item in report["results"]],
        ))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_row=len(report["results"]) + 1)
        logging.error("Classification-only failed: %s", report["error"])
    except KeyboardInterrupt:
        report.update(status="failed", error="Прогон прерван; итогового F1 нет")
    finally:
        session["elapsed_seconds"] = perf_counter() - started
        report["elapsed_seconds"] = sum(item["elapsed_seconds"] for item in report["sessions"])
        report["timing_includes"] = "extractor.run(), retries and quota waits; excludes instance construction and dataset loading"
        save_report(args.report, report)
    if report["status"] == "complete":
        logging.info("Classification-only: %s", report["metrics"])
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
