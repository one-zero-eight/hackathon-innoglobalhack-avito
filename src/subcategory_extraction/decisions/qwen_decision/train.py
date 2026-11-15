"""Выбор промпта и порогов Qwen только на группах исправленного train."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from random import Random

import numpy as np

from benchmark_support import metadata, save_report, setup
from database import table_sha256
from main import MICROCATEGORIES
from run import EvaluationRow, read_records, prepare_rows
from subcategory_extraction.decisions.qwen_decision.pipeline import (
    DEFAULT_CALIBRATION, PROMPTS, SubcategoryExtractionPipeline,
)
from subcategory_extraction.semantic.semantic_encoder_f1.robust_thresholds import (
    binary_metrics, select_robust_thresholds, validation_groups,
)

ROOT = Path(__file__).resolve().parents[4]
# Подбор промпта и порогов на ручной смысловой разметке train из PostgreSQL.
TRAIN_TABLE = "manual_train"
REPORT = ROOT / "src/subcategory_extraction/benchmarks/improved_dataset/qwen-training-results.json"
SEED = 20261114
CATEGORIES = [cid for cid in MICROCATEGORIES if cid != 101]
logger = logging.getLogger(__name__)


def collect(variant: str, rows: list[EvaluationRow], workers: int) -> tuple[np.ndarray, list[dict]]:
    """Получить реальные ответы с ограниченной параллельностью и дисковым кэшем."""
    os.environ["QWEN_DECISION_PROMPT"] = variant

    def request(row: EvaluationRow) -> tuple[list[float], dict]:
        """Создать отдельный клиент, чтобы usage разных потоков не смешивался."""
        pipeline = SubcategoryExtractionPipeline()
        values = pipeline.probabilities(row.inp)
        return [values[cid] for cid in CATEGORIES], {
            "description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
            "probabilities": {str(cid): values[cid] for cid in CATEGORIES},
            "expected": sorted(row.expected), "cache_hit": pipeline.client.last_cache_hit,
            "usage": pipeline.client.last_usage,
            "http_attempts": pipeline.client.last_attempts,
        }

    outputs, records = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for values, record in pool.map(request, rows):
            outputs.append(values)
            records.append(record)
            if len(records) % 20 == 0 or len(records) == len(rows):
                logger.info("%s: %d/%d", variant, len(records), len(rows))
    return np.asarray(outputs), records


def main() -> None:
    """Зафиксировать решение до test; контроль не участвует в выборе параметров."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-size", type=int, default=240)
    parser.add_argument("--control-size", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.fit_size < 30 or args.control_size < 20 or not 1 <= args.workers <= 8:
        parser.error("Нужны fit >= 30, control >= 20 и workers 1..8")
    protected = {DEFAULT_CALIBRATION.resolve()}
    if args.report.resolve() in protected or args.report.name == "recorded-results.json":
        parser.error("Отчёт не должен перезаписывать калибровку или исторические ответы")
    setup()
    records = read_records(TRAIN_TABLE)
    rows = prepare_rows(records, 101)
    groups = validation_groups(records)
    representatives = {}
    for index, group in enumerate(groups):
        representatives.setdefault(int(group), index)
    selected = sorted(representatives)
    Random(SEED).shuffle(selected)
    if len(selected) < args.fit_size + args.control_size:
        parser.error(f"Недостаточно групп: {len(selected)}")
    fit_groups = selected[:args.fit_size]
    control_groups = selected[args.fit_size:args.fit_size + args.control_size]
    fit = [rows[representatives[group]] for group in fit_groups]
    control = [rows[representatives[group]] for group in control_groups]
    labels = np.array([[cid in row.expected for cid in CATEGORIES] for row in fit], dtype=bool)
    report = {
        **metadata(), "status": "running", "mode": "train_only_prompt_and_threshold_selection",
        "train": TRAIN_TABLE,
        "train_sha256": table_sha256(TRAIN_TABLE),
        "model": None, "endpoint": None, "seed": SEED,
        "grouping": {"method": "char_wb_3_5_tfidf_cosine_components", "similarity": 0.85,
                     "total_rows": len(rows), "total_groups": len(representatives),
                     "fit_groups": fit_groups, "control_groups": control_groups,
                     "selection": "one first representative per seeded sampled group",
                     "cross_split_groups": 0,
                     "limitation": "heuristic near-duplicates, not full semantic equivalence"},
        "test_used_for_selection": False, "candidates": {},
    }
    for variant in PROMPTS:
        probabilities, responses = collect(variant, fit, args.workers)
        tuning = select_robust_thresholds(probabilities, labels, CATEGORIES, np.asarray(fit_groups))
        report["candidates"][variant] = {"tuning": tuning, "responses": responses}
        save_report(args.report, report)
        logger.info("%s: fixed F1=%.6f; crossfit F1=%.6f (%s)", variant,
                    binary_metrics(labels, probabilities >= 0.5)["micro_f1"],
                    tuning["selection_metric"]["micro_f1"], tuning["selected_method"])
    variant = max(PROMPTS, key=lambda name: report["candidates"][name]["tuning"]["selection_metric"]["micro_f1"])
    tuning = report["candidates"][variant]["tuning"]
    os.environ["QWEN_DECISION_PROMPT"] = variant
    pipeline = SubcategoryExtractionPipeline()
    report.update(model=pipeline.client.model_id, endpoint=pipeline.client.endpoint, selected_prompt=variant)
    artifact = {
        **metadata(), "version": 1, "model": pipeline.client.model_id,
        "endpoint": pipeline.client.endpoint, "prompt": variant,
        "prompt_sha256": pipeline.prompt_sha256, "thresholds": tuning["thresholds"],
        "train_sha256": report["train_sha256"], "fit_rows": len(fit),
        "fit_group_ids": fit_groups, "control_group_ids": control_groups,
        "fit_description_sha256": [hashlib.sha256(row.inp.description.encode()).hexdigest() for row in fit],
        "selection": "best prompt by threshold-only 3-fold crossfit on train groups; control/test unused",
        "crossfit_metric": tuning["selection_metric"], "threshold_method": tuning["selected_method"],
        "source_mc_id": 101, "category_101_threshold": "untrained, fixed at 0.5",
    }
    save_report(DEFAULT_CALIBRATION, artifact)
    report["frozen_calibration_sha256"] = hashlib.sha256(DEFAULT_CALIBRATION.read_bytes()).hexdigest()
    save_report(args.report, report)
    # Выбранный промпт и пороги уже сохранены; контроль не меняет артефакт.
    probabilities, responses = collect(variant, control, args.workers)
    truth = np.array([[cid in row.expected for cid in CATEGORIES] for row in control], dtype=bool)
    thresholds = np.array([tuning["thresholds"][str(cid)] for cid in CATEGORIES])
    report["control"] = {
        "rows": len(control), "responses": responses,
        "fixed": binary_metrics(truth, probabilities >= 0.5),
        "calibrated": binary_metrics(truth, probabilities >= thresholds),
        "used_for_selection": False,
    }
    report["status"] = "complete"
    save_report(args.report, report)
    logger.info("Frozen %s %s; control fixed=%.6f calibrated=%.6f", variant, tuning["selected_method"],
                report["control"]["fixed"]["micro_f1"], report["control"]["calibrated"]["micro_f1"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
