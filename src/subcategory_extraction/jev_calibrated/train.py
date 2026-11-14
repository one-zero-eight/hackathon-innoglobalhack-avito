"""Выбор промпта и порогов Jev только на группах официального train."""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from benchmark_support import ROOT, metadata, save_report, setup
from database import table_sha256
from main import MICROCATEGORIES
from run import EvaluationRow, normalize_description, prepare_rows, read_records
from subcategory_extraction.jev_calibrated.pipeline import (
    DEFAULT_CALIBRATION, PROMPTS, SubcategoryExtractionPipeline,
)
from subcategory_extraction.semantic_encoder_f1.robust_thresholds import (
    binary_metrics, select_robust_thresholds, validation_groups,
)

TRAIN_TABLE = "case_train"
REPORT = ROOT / "src/subcategory_extraction/benchmarks/classifier_methods/jev-training-results.json"
TEST_TABLES = ("case_test", "manual_test")
SEED = 20261115
SOURCE_MC_ID = 101
CATEGORIES = [cid for cid in MICROCATEGORIES if cid != SOURCE_MC_ID]
logger = logging.getLogger(__name__)


def split_groups(fit_size: int, control_size: int) -> tuple[list[EvaluationRow], list[EvaluationRow], dict]:
    """Разделить точные и близкие дубликаты, исключив группы с текстами test."""
    original = read_records(TRAIN_TABLE)
    unique: dict[str, dict] = {}
    for record in original:
        key = normalize_description(record["description"])
        if key in unique:
            if unique[key]["found_microcategories"] != record["found_microcategories"]:
                raise ValueError("Конфликт меток одинаковых train-описаний")
        else:
            unique[key] = record
    records = list(unique.values())
    rows = prepare_rows(records, SOURCE_MC_ID)
    groups = validation_groups(records)
    # Метки test не используются: нужны только тексты для исключения пересечений.
    protected = {normalize_description(row["description"])
                 for table in TEST_TABLES for row in read_records(table)}
    excluded = {int(group) for row, group in zip(rows, groups, strict=True)
                if normalize_description(row.inp.description) in protected}
    representatives: dict[int, int] = {}
    for index, group in enumerate(groups):
        if int(group) not in excluded:
            representatives.setdefault(int(group), index)
    selected = sorted(representatives)
    Random(SEED).shuffle(selected)
    if len(selected) < fit_size + control_size:
        raise ValueError("Недостаточно независимых train-групп")
    fitting = selected[:fit_size]
    control = selected[fit_size:fit_size + control_size]
    return (
        [rows[representatives[group]] for group in fitting],
        [rows[representatives[group]] for group in control],
        {"method": "normalized_exact_then_char_wb_3_5_cosine_components",
         "near_duplicate_similarity": 0.85, "seed": SEED,
         "original_rows": len(original), "unique_rows": len(rows),
         "eligible_groups": len(selected), "excluded_test_touching_groups": len(excluded),
         "fit_groups": fitting, "control_groups": control, "cross_split_groups": 0,
         "limitation": "heuristic near-duplicate grouping, not semantic equivalence"},
    )


def collect(
    variant: str, rows: list[EvaluationRow], records: list[dict], report: dict, path: Path,
) -> np.ndarray:
    """Сохранять каждый реальный ответ; повторный запуск использует проверенный кэш."""
    os.environ["JEV_CALIBRATED_PROMPT"] = variant
    pipeline = SubcategoryExtractionPipeline()
    values = []
    for index, row in enumerate(rows, start=1):
        probabilities = pipeline.probabilities(row.inp)
        records.append({
            "description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
            "expected": sorted(row.expected),
            "probabilities": {str(cid): probabilities[cid] for cid in CATEGORIES},
            "cache_hit": pipeline.client.last_cache_hit, "usage": pipeline.client.last_usage,
        })
        values.append([probabilities[cid] for cid in CATEGORIES])
        save_report(path, report)
        if index % 12 == 0 or index == len(rows):
            logger.info("%s: %d/%d train-групп", variant, index, len(rows))
    return np.asarray(values)


def main() -> int:
    """Сравнить два промпта и девять схем порогов без выбора по test/control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-size", type=int, default=48)
    parser.add_argument("--control-size", type=int, default=24)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.fit_size < 30 or args.control_size < 20:
        parser.error("Нужны fit >= 30 и control >= 20")
    if args.report.resolve() == DEFAULT_CALIBRATION.resolve():
        parser.error("Отчёт не должен перезаписывать данные или калибровку")
    if args.report.name == "recorded-results.json":
        parser.error("Нельзя перезаписать исторические ответы")
    setup()
    fit, control, grouping = split_groups(args.fit_size, args.control_size)
    fit_labels = np.array([[cid in row.expected for cid in CATEGORIES] for row in fit], dtype=bool)
    control_labels = np.array([[cid in row.expected for cid in CATEGORIES] for row in control], dtype=bool)
    report = {
        **metadata(), "status": "running", "mode": "train_only_prompt_threshold_selection",
        "training_dataset": TRAIN_TABLE, "training_storage": "postgresql",
        "training_sha256": table_sha256(TRAIN_TABLE),
        "grouping": grouping, "fit_rows": len(fit), "control_rows": len(control),
        "fit_positive_counts": dict(zip(CATEGORIES, fit_labels.sum(axis=0).tolist(), strict=True)),
        "test_labels_used_for_selection": False, "control_used_for_selection": False,
        "promotion_rule": "crossfit-selected candidate must beat guided@0.5 on untouched control by >=0.002",
        "candidates": {},
    }
    try:
        for variant in PROMPTS:
            candidate = {"fit_responses": []}
            report["candidates"][variant] = candidate
            probabilities = collect(variant, fit, candidate["fit_responses"], report, args.report)
            tuning = select_robust_thresholds(
                probabilities, fit_labels, CATEGORIES, np.asarray(grouping["fit_groups"]),
            )
            candidate.update(tuning=tuning, fixed=binary_metrics(fit_labels, probabilities >= 0.5))
            save_report(args.report, report)
            logger.info("%s: fixed=%.6f crossfit=%.6f (%s)", variant,
                        candidate["fixed"]["micro_f1"], tuning["selection_metric"]["micro_f1"],
                        tuning["selected_method"])
        selected = max(PROMPTS, key=lambda name: report["candidates"][name]["tuning"]["selection_metric"]["micro_f1"])
        tuning = report["candidates"][selected]["tuning"]
        report.update(selected_prompt=selected, frozen_thresholds=tuning["thresholds"])
        # Параметры уже выбраны. Control не меняет победителя или пороги.
        for variant in PROMPTS:
            candidate = report["candidates"][variant]
            candidate["control_responses"] = []
            probabilities = collect(variant, control, candidate["control_responses"], report, args.report)
            thresholds = np.array([candidate["tuning"]["thresholds"][str(cid)] for cid in CATEGORIES])
            candidate["control"] = {
                "fixed": binary_metrics(control_labels, probabilities >= 0.5),
                "calibrated": binary_metrics(control_labels, probabilities >= thresholds),
                "used_for_selection": False,
            }
        baseline = report["candidates"]["guided"]["control"]["fixed"]
        proposed = report["candidates"][selected]["control"]["calibrated"]
        confirmed = proposed["micro_f1"] >= baseline["micro_f1"] + 0.002
        report.update(control_baseline=baseline, control_selected=proposed,
                      control_confirmed=confirmed, artifact_saved=False)
        if confirmed:
            os.environ["JEV_CALIBRATED_PROMPT"] = selected
            pipeline = SubcategoryExtractionPipeline()
            artifact = {
                **metadata(), "version": 1, "provider": pipeline.client.provider,
                "endpoint": pipeline.client.endpoint, "model": pipeline.client.model_id,
                "prompt": selected, "prompt_sha256": pipeline.prompt_sha256,
                "thresholds": tuning["thresholds"], "threshold_method": tuning["selected_method"],
                "training_dataset": report["training_dataset"],
                "training_sha256": report["training_sha256"], "grouping": grouping,
                "fit_description_sha256": [hashlib.sha256(row.inp.description.encode()).hexdigest() for row in fit],
                "control_description_sha256": [hashlib.sha256(row.inp.description.encode()).hexdigest() for row in control],
                "fit_crossfit": tuning["selection_metric"], "control_confirmed": True,
                "control_baseline": baseline, "control_selected": proposed,
                "source_mc_id": SOURCE_MC_ID, "category_101_threshold": "untrained, fixed 0.5",
            }
            save_report(DEFAULT_CALIBRATION, artifact)
            report.update(artifact_saved=True, calibration_sha256=hashlib.sha256(DEFAULT_CALIBRATION.read_bytes()).hexdigest())
        report["status"] = "complete"
        save_report(args.report, report)
        logger.info("Frozen %s %s; control baseline=%.6f selected=%.6f; confirmed=%s", selected,
                    tuning["selected_method"], baseline["micro_f1"], proposed["micro_f1"], confirmed)
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", artifact_saved=False)
        save_report(args.report, report)
        logger.error("Обучение не завершено; калибровка не заменена: %s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
