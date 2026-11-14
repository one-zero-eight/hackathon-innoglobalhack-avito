"""Проверить метрики офлайн или запустить оценку полного решения."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import logging
from time import perf_counter

from benchmark_support import metadata, save_report, setup, timing


def check() -> None:
    """Повторить контрольные метрики, чтение Parquet и аудит пересечений."""
    from run import ROOT, audit_overlap, calculate_metrics, prepare_rows, read_records
    assert calculate_metrics([{102, 105}], [{102, 104}]) == {
        "micro_f1": 0.5, "classification_points": 30.0, "tp": 1, "fp": 1, "fn": 1,
    }
    assert calculate_metrics([set()], [set()])["micro_f1"] == 0
    assert calculate_metrics([{102}], [{101, 102}])["fp"] == 1
    assert calculate_metrics([{102}], [{102}])["micro_f1"] == 1
    assert calculate_metrics([{102}, set()], [{102}, {104, 105}])["micro_f1"] == 0.5
    data = ROOT / "docs/innoglobalhack_avito"
    rows = prepare_rows(read_records(data / "public-test-dataset.parquet"), 101)
    assert len(rows) == 208
    assert set.union(*(row.expected for row in rows)) == {102, 103, 104, 105, 106}
    overlap = audit_overlap(rows, data / "public-train-annotated.jsonl")
    assert overlap["overlapping_rows"] == 24
    logging.info("Формула и крайние случаи: OK; Parquet: %d строк; пересечения: %d",
                 len(rows), overlap["overlapping_rows"])
    check_checkpoint()


def check_checkpoint() -> None:
    """Проверить resume и отказ на повреждённый кэш без вызовов модели."""
    import json
    import os
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from main import MICROCATEGORIES, Draft, PipelineOutput

    with TemporaryDirectory(prefix="evaluation-checkpoint-") as folder:
        root = Path(folder)
        (root / "pipeline.json").write_text(json.dumps({
            "subcategory_extraction": "google_decision", "draft_generation": "llm",
        }))
        row = {"description": "Устанавливаю раковины.", "categories": [MICROCATEGORIES[104]]}
        (root / "dataset.jsonl").write_text((json.dumps(row, ensure_ascii=False) + "\n") * 2)
        config_path = root / "run.json"
        config_path.write_text(json.dumps({
            "pipeline_config": "pipeline.json", "dataset": "dataset.jsonl", "report": "unused.json",
            "source_mc_id": 101,
        }))
        report_path = root / "results.json"
        output = PipelineOutput(drafts=[Draft(mcId=104, title="Сантехника", text="Устанавливаю раковины.")])
        with patch("main.run_pipeline", return_value=output) as execute:
            assert run_live(config_path, report_path, resume=False) == 0
        assert execute.call_count == 2
        original = json.loads(report_path.read_text())
        with patch("main.run_pipeline") as execute:
            assert run_live(config_path, report_path, resume=True) == 0
        assert execute.call_count == 0
        changed = json.loads(json.dumps(original))
        changed["results"][0]["predicted"] = [105]
        report_path.write_text(json.dumps(changed))
        with patch("main.run_pipeline") as execute:
            assert run_live(config_path, report_path, resume=True) == 1
        failed = json.loads(report_path.read_text())
        assert failed["status"] == "failed" and "metrics" not in failed
        assert execute.call_count == 0
        report_path.write_text(json.dumps(original))
        with patch.dict(os.environ, {"EXTRACTION_MODEL_ID": "changed-offline-model"}), patch("main.run_pipeline") as execute:
            try:
                run_live(config_path, report_path, resume=True)
            except ValueError:
                pass
            else:
                raise AssertionError("Изменение модели не запретило resume")
        assert execute.call_count == 0
    logging.info("Контрольные точки: сохранение, resume, повреждённый кэш и смена модели — OK")



def run_live(config_path: Path, report_path: Path, resume: bool) -> int:
    """Прогнать реальные строки с контрольными точками и ограниченным повтором сети."""
    import hashlib
    import json
    import os

    from main import PipelineOutput, load_config, run_pipeline
    from run import RunConfig, audit_overlap, calculate_metrics, prepare_rows, read_records, resolve_path

    config_path = config_path.resolve()
    config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    pipeline_path = resolve_path(base, config.pipeline_config)
    selected = load_config(pipeline_path)
    if any(value is None for value in selected.model_dump().values()):
        raise ValueError("Для полного прогона выберите оба модуля")
    dataset = resolve_path(base, config.dataset)
    records = read_records(dataset)
    if config.limit is not None:
        records = records[:config.limit]
    rows = prepare_rows(records, config.source_mc_id)
    training = resolve_path(base, config.training_dataset) if config.training_dataset else None
    fingerprint = hashlib.sha256(config_path.read_bytes() + pipeline_path.read_bytes() + dataset.read_bytes())
    # Кэш не должен пережить изменение кода, модели, промпта или артефакта.
    fingerprint.update(metadata()["code_sha256"].encode())
    public_settings = {key: value for key, value in os.environ.items()
                       if key.startswith(("EXTRACTION_", "GENERATION_"))
                       and not any(word in key for word in ("KEY", "TOKEN", "SECRET"))}
    public_settings["LLM_MODEL_ID"] = os.getenv("LLM_MODEL_ID", "")
    fingerprint.update(json.dumps(public_settings, sort_keys=True).encode())
    signature = fingerprint.hexdigest()
    if resume:
        report = json.loads(report_path.read_text())
        if report.get("fingerprint") != signature:
            raise ValueError("Изменились данные, код, конфигурация, промпты или веса; нужен новый прогон")
        report.pop("metrics", None)
    else:
        report = {**metadata(), "scenario": "live", "fingerprint": signature,
                  "rows": len(rows), "limited": config.limit is not None,
                  "pipeline_config": selected.model_dump(), "run_config": config.model_dump(),
                  "overlap": audit_overlap(rows, training), "results": [], "sessions": []}
    report.update(status="running")
    report.pop("error", None)
    session = {**metadata(), "resumed": resume, "starting_row": len(report["results"]) + 1}
    report["sessions"].append(session)
    started = perf_counter()
    try:
        if len(report["results"]) > len(rows):
            raise ValueError("Контрольная точка длиннее исходного датасета")
        for index, row in enumerate(rows, 1):
            if index <= len(report["results"]):
                saved = report["results"][index - 1]
                saved_output = PipelineOutput.model_validate(saved["output"])
                if saved["predicted"] != sorted({draft.mc_id for draft in saved_output.drafts}):
                    raise ValueError("Предсказания контрольной точки не совпали с её черновиками")
                if saved["row_number"] != index or saved["expected"] != sorted(row.expected):
                    raise ValueError("Строка контрольной точки не совпала с эталоном")
                continue
            call_started = perf_counter()
            output = run_pipeline(row.inp, selected)
            elapsed = perf_counter() - call_started
            report["results"].append({"row_number": index, "expected": sorted(row.expected),
                                      "predicted": sorted({draft.mc_id for draft in output.drafts}),
                                      "output": output.model_dump(by_alias=True), "seconds": elapsed,
                                      "pipeline_calls": 1})
            report.update(completed_rows=index, response_time=timing([item["seconds"] for item in report["results"]]))
            save_report(report_path, report)
            logging.info("Полный пайплайн: %d/%d, %.3f с", index, len(rows), elapsed)
        report.update(status="complete", metrics=calculate_metrics(
            [set(item["expected"]) for item in report["results"]],
            [set(item["predicted"]) for item in report["results"]],
        ))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      failed_row=len(report["results"]) + 1)
        logging.error("Прогон не завершён: %s", report["error"])
    finally:
        session["elapsed_seconds"] = perf_counter() - started
        report["elapsed_seconds"] = sum(item["elapsed_seconds"] for item in report["sessions"])
        save_report(report_path, report)
    return 0 if report["status"] == "complete" else 1


def main() -> int:
    """Проверить оценщик или выполнить реальный полный прогон."""
    from run import DEFAULT_RUN_CONFIG

    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--report", type=Path, help="Переопределить путь текущего отчёта")
    parser.add_argument("--resume", action="store_true", help="Продолжить live из контрольной точки без повторной генерации готовых строк")
    args = parser.parse_args()
    if args.live:
        return run_live(args.config, args.report or Path(__file__).with_name("pipeline-results.json"), args.resume)
    if args.resume:
        parser.error("--resume требует --live")
    started = perf_counter()
    report = {**metadata(), "scenario": "offline"}
    check()
    report.update(status="complete", rows=208, overlapping_rows=24, checkpoint_checks=4,
                  elapsed_seconds=perf_counter() - started)
    save_report(args.report or Path(__file__).with_name("checks-results.json"), report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
