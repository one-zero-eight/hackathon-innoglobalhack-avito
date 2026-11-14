"""Воспроизведение полного протокола Leo и публикация локального артефакта."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import hashlib
import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone
from benchmark_support import metadata

import numpy as np

from subcategory_extraction.semantic_encoder_f1.train_plan import (
    ROOT, evaluate_plan, release_memory, save_json, train_plan,
)
from subcategory_extraction.semantic_encoder.train import (
    load_training_rows, train,
)
from subcategory_extraction.semantic_encoder_calibrated.train import (
    run, write_split,
)

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts"
REPORT = ROOT / "out/rubert-f1-plan/reproduced-results.json"
SUMMARY = ROOT / "out/rubert-f1-plan/summary-results.json"
logger = logging.getLogger(__name__)


def main() -> None:
    """Обучить исходные baseline и все варианты, затем измерить F1 и размер."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Явно разрешить длительное полное обучение")
    args = parser.parse_args()
    if not args.train:
        if REPORT.exists():
            logger.info("Существующий отчёт сохранён; обучение без --train не запускается.")
            return
        save_json(REPORT, {
            **metadata(), "status": "cancelled",
            "reason": "Полное обучение остановлено пользователем; не возобновлять без --train",
            "completed_stage": "initial split baseline trained; full protocol not completed",
            "log": "out/reproduce-weights.log", "exit_code": 143,
            "final_f1": None,
        })
        logger.info("Обучение отменено пользователем. Для нового запуска требуется --train.")
        return
    if ARTIFACT.exists() or REPORT.exists() or SUMMARY.exists():
        raise FileExistsError("Итоговые артефакты уже существуют; перезапись запрещена")
    started = time.perf_counter()
    source = ROOT / "docs/improved_dataset/train.jsonl"
    threshold_output = ROOT / "out/rubert-thresholds"
    baseline_output = ROOT / "out/improved-dataset"
    output = ROOT / "out/rubert-f1-plan"
    try:
        if not (threshold_output / "encoder/train-report.json").exists():
            run(source, threshold_output, ROOT / "out/reproduced-thresholds.json", "auto")
        release_memory()
        baseline = baseline_output / "semantic-encoder"
        if not (baseline / "train-report.json").exists():
            baseline_output.mkdir(parents=True, exist_ok=True)
            rows, _ = load_training_rows(source)
            dataset = baseline_output / "hybrid-training.jsonl"
            if dataset.exists():
                raise FileExistsError(dataset)
            write_split(dataset, rows, np.arange(len(rows)))
            train(dataset, baseline, "auto")
        release_memory()
        train_plan(output, "auto")
        release_memory()
        report = evaluate_plan(output, REPORT)
        shutil.copytree(output / "selected", ARTIFACT)
        files = [
            {
                "path": str(path.relative_to(ARTIFACT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(ARTIFACT.rglob("*")) if path.is_file()
        ]
        summary = {
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    HERE / "train_plan.py",
                    HERE / "train.py",
                    HERE.parent / "semantic_encoder/train.py",
                    HERE.parent / "semantic_encoder_calibrated/train.py",
                    HERE / "robust_thresholds.py",
                    ROOT / "src/dataset_annotation/rubert_review/run.py",
                    HERE / "model.py",
                    HERE / "pipeline.py",
                )
            },
            "selected": report["selected"],
            "cpu": report["cpu"],
            "artifact": str(ARTIFACT.relative_to(ROOT)),
            "files": files,
            "weights_bytes": sum(
                item["bytes"] for item in files if item["path"].endswith(".safetensors")
            ),
            "artifact_bytes": sum(item["bytes"] for item in files),
            "total_seconds": time.perf_counter() - started,
            "cost": "бесплатно; локальное обучение, без внешних LLM API",
            "limitation": "119 diagnostic rows are not independent manually verified gold",
        }
        save_json(SUMMARY, summary)
        logger.info("micro-F1 %.6f; веса %d байт; артефакт %d байт",
                    report["selected"]["diagnostic"]["micro_f1"],
                    summary["weights_bytes"], summary["artifact_bytes"])
    except Exception as error:
        save_json(ROOT / "out/reproduced-weights-failure.json", {
            "status": "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "total_seconds": time.perf_counter() - started,
        })
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
