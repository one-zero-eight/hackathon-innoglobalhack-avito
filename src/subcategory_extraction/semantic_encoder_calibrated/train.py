"""Честный подбор порогов RuBERT: групповой train/validation/control, inference CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
import torch
from benchmark_support import replay_recorded_report

from main import MICROCATEGORIES, PipelineInput
from run import normalize_description
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.semantic_encoder_f1.threshold_search import pick_thresholds
from database import DATABASE_TABLES, read_training_rows as database_read_training_rows
from subcategory_extraction.semantic_encoder.train import load_training_rows, train
from subcategory_extraction.semantic_encoder.pipeline import SubcategoryExtractionPipeline
from subcategory_extraction.semantic_encoder_calibrated.pipeline import (
    SubcategoryExtractionPipeline as CalibratedPipeline,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_CATEGORY = 101
SPLIT_SEED = 42
SIMILARITY = 0.85
logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Зафиксировать точные байты входа или артефакта."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    """Сохранить воспроизводимые числа без JSON NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_rows(rows: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    """Разделить компоненты близких текстов, не используя целевые метки."""
    texts = [normalize_description(row["description"]) for row in rows]
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        """Найти корень компоненты с сокращением пути."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    features = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=40_000, dtype=np.float32,
    ).fit_transform(texts)
    neighbors = NearestNeighbors(
        metric="cosine", algorithm="brute", radius=1 - SIMILARITY, n_jobs=1,
    ).fit(features).radius_neighbors(features, return_distance=False)
    exact = {}
    for index, text in enumerate(texts):
        previous = exact.setdefault(text, index)
        first, second = find(index), find(previous)
        if first != second:
            parents[max(first, second)] = min(first, second)
        for other in neighbors[index]:
            first, second = find(index), find(int(other))
            if first != second:
                parents[max(first, second)] = min(first, second)
    groups = np.array([find(index) for index in range(len(rows))])
    indices = np.arange(len(rows))
    remaining, control = next(GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SPLIT_SEED,
    ).split(indices, groups=groups))
    training_local, validation_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.15 / 0.85, random_state=SPLIT_SEED + 1,
    ).split(remaining, groups=groups[remaining]))
    splits = {
        "train": np.sort(remaining[training_local]),
        "validation": np.sort(remaining[validation_local]),
        "control": np.sort(control),
    }
    assignment = np.empty(len(rows), dtype=int)
    for number, selected in enumerate(splits.values()):
        assignment[selected] = number
    cross_edges = sum(
        int(assignment[index] != assignment[int(other)])
        for index, related in enumerate(neighbors) for other in related
    )
    if cross_edges or len(set(np.concatenate(list(splits.values())))) != len(rows):
        raise RuntimeError("Нарушена изоляция компонент или полнота разбиения")
    return splits, {
        "method": "connected_components_char_wb_tfidf_cosine_then_GroupShuffleSplit",
        "normalization": "NFKC/casefold/ё→е/whitespace",
        "similarity_threshold": SIMILARITY,
        "vectorizer": {"ngram_range": [3, 5], "min_df": 2, "max_features": 40_000},
        "group_count": len(set(groups.tolist())),
        "largest_group_rows": int(np.unique(groups, return_counts=True)[1].max()),
        "seed_control": SPLIT_SEED, "seed_validation": SPLIT_SEED + 1,
        "requested_group_fractions": {"train": 0.70, "validation": 0.15, "control": 0.15},
        "cross_split_similarity_edges": cross_edges,
        "labels_used_for_grouping_or_split": False,
        "limitation": "No known exact/TF-IDF edges cross splits; not a guarantee against all semantic/template leakage.",
    }


def write_split(path: Path, rows: list[dict], indices: np.ndarray) -> None:
    """Сохранить полные исходные описания и наборы меток выбранной части."""
    with path.open("w", encoding="utf-8") as stream:
        for index in indices:
            row = rows[int(index)]
            stream.write(json.dumps({
                "description": row["description"],
                "found_microcategories": [{"mcId": category} for category in sorted(row["targets"])],
            }, ensure_ascii=False) + "\n")


def probabilities(pipeline: SubcategoryExtractionPipeline, rows: list[dict], categories: list[int]) -> np.ndarray:
    """Вычислить настоящие вероятности локальной модели последовательно на CPU."""
    outputs = []
    for row in rows:
        values = pipeline.probabilities(PipelineInput(description=row["description"], mcId=SOURCE_CATEGORY))
        outputs.append([values[category] for category in categories])
    return np.asarray(outputs, dtype=np.float64)


def predicted_sets(values: np.ndarray, thresholds: dict, categories: list[int]) -> list[set[int]]:
    """Применить заранее зафиксированные пороги к сохранённым вероятностям."""
    boundary = np.array([thresholds[str(category)] for category in categories])
    return [
        {category for category, keep in zip(categories, record, strict=True) if keep}
        for record in values >= boundary
    ]


def metrics(expected: list[set[int]], predicted: list[set[int]]) -> dict:
    """Дополнить общую micro-F1 диагностикой precision/recall и пустых наборов."""
    result = score(expected, predicted)
    result.update({
        "precision": result["tp"] / (result["tp"] + result["fp"]) if result["tp"] + result["fp"] else 0.0,
        "recall": result["tp"] / (result["tp"] + result["fn"]) if result["tp"] + result["fn"] else 0.0,
        "exact_match_rows": sum(truth == output for truth, output in zip(expected, predicted, strict=True)),
        "empty_target_rows": sum(not truth for truth in expected),
        "empty_target_false_positive_rows": sum(not truth and bool(output) for truth, output in zip(expected, predicted, strict=True)),
    })
    return result


def run(dataset: str | Path, output: Path, report_path: Path, device: str) -> dict:
    """Обучить один фиксированный encoder и выбрать пороги без доступа к control.

    ``dataset`` может быть именем таблицы PostgreSQL (по умолчанию
    ``manual_train`` — ручная смысловая разметка) или путём к файлу
    (для воспроизведения прежних зафиксированных протоколов).
    """
    if output.exists() or report_path.exists():
        raise ValueError("Для нового опыта нужны новые output/report: существующие результаты не перезаписываются")
    started = time.perf_counter()
    if isinstance(dataset, str) and dataset in DATABASE_TABLES:
        rows, source_hash = database_read_training_rows(dataset)
        dataset_label = f"postgresql:{dataset}"
    else:
        path = Path(dataset)
        rows, source_hash = load_training_rows(path)
        dataset_label = str(path.resolve())
    splits, grouping = split_rows(rows)
    output.mkdir(parents=True)
    categories = [category for category in MICROCATEGORIES if category != SOURCE_CATEGORY]
    split_manifest = {}
    for name, indices in splits.items():
        path = output / f"{name}.jsonl"
        write_split(path, rows, indices)
        split_manifest[name] = {
            "path": str(path.resolve()), "sha256": digest(path), "rows": len(indices),
            "source_row_numbers": (indices + 1).tolist(),
            "positive_counts": {str(category): sum(category in rows[int(index)]["targets"] for index in indices) for category in categories},
            "empty_target_rows": sum(not rows[int(index)]["targets"] for index in indices),
        }
    save_json(output / "split-manifest.json", {
        "source_sha256": source_hash, "grouping": grouping, "splits": split_manifest,
    })
    logger.info("Разбиение: %s; компонент: %d", {key: value["rows"] for key, value in split_manifest.items()}, grouping["group_count"])
    artifact = output / "encoder"
    training = train(output / "train.jsonl", artifact, device)
    os.environ["SEMANTIC_ENCODER_DEVICE"] = "cpu"
    baseline = SubcategoryExtractionPipeline(artifact_path=artifact)
    validation = [rows[int(index)] for index in splits["validation"]]
    validation_values = probabilities(baseline, validation, categories)
    validation_labels = np.array([[category in row["targets"] for category in categories] for row in validation], dtype=bool)
    selection = pick_thresholds(validation_values, validation_labels, categories)
    calibration = {
        "version": 1, "artifact_metadata_sha256": digest(artifact / "metadata.json"),
        **selection,
        "thresholds": {str(SOURCE_CATEGORY): 0.5, **selection["thresholds"]},
        "validation_sha256": split_manifest["validation"]["sha256"],
        "train_sha256": split_manifest["train"]["sha256"],
        "source_dataset_sha256": source_hash,
        "selection_data": "validation only; no control or public test scores/labels used",
    }
    # Фиксируем решение на диске до первого inference и расчёта метрик control.
    save_json(artifact / "calibration.json", calibration)
    calibration_hash = digest(artifact / "calibration.json")
    calibrated = CalibratedPipeline(artifact_path=artifact)
    control = [rows[int(index)] for index in splits["control"]]
    control_values = probabilities(baseline, control, categories)
    control_expected = [row["targets"] for row in control]
    control_predictions = {
        name: predicted_sets(control_values, candidate["thresholds"], categories)
        for name, candidate in selection["candidates"].items()
    }
    actual_predictions, durations = [], []
    for row in control:
        inp = PipelineInput(description=row["description"], mcId=SOURCE_CATEGORY)
        tick = time.perf_counter()
        actual_predictions.append(set(calibrated.run(inp)))
        durations.append(time.perf_counter() - tick)
    if actual_predictions != control_predictions[selection["selected_method"]]:
        raise RuntimeError("Реальный calibrated run расходится с зафиксированными порогами")
    if digest(artifact / "calibration.json") != calibration_hash:
        raise RuntimeError("Пороги изменились после доступа к control")
    durations.sort()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label, "dataset_sha256": source_hash,
        "protocol": {
            "selection_split": "validation", "evaluation_split": "control",
            "public_test_used": False, "control_used_for_selection": False,
            "source_mc_id": SOURCE_CATEGORY, "unsupported_category": SOURCE_CATEGORY,
            "fixed_training": "Original pretrained RuBERT-tiny2; 3 epochs, last checkpoint; only train split.",
            "limitation": "Internal holdout from provisional automatic labels, not independent manual gold; not comparable with diagnostic119 F1.",
        },
        "grouping": grouping, "splits": split_manifest,
        "training": training, "calibration": calibration,
        "calibration_sha256_before_control": calibration_hash,
        "control_metrics": {name: metrics(control_expected, predictions) for name, predictions in control_predictions.items()},
        "selected_control_metrics": metrics(control_expected, actual_predictions),
        "inference": {
            "device": calibrated.device, "api_requests": 0,
            "control_run_calls": len(control), "run_outputs_match_saved_thresholds": True,
            "latency_ms": {"median": median(durations) * 1000, "p95_nearest_rank": durations[math.ceil(0.95 * len(durations)) - 1] * 1000, "mean": sum(durations) / len(durations) * 1000},
            "latency_scope": "Actual calibrated run(), sequential documents; excludes loading; warmed by probability collection.",
            "torch_threads": torch.get_num_threads(),
        },
        "observations": {
            name: [
                {"source_row_number": int(index) + 1,
                 "description_sha256": hashlib.sha256(rows[int(index)]["description"].encode()).hexdigest(),
                 "expected": sorted(rows[int(index)]["targets"]),
                 "probabilities": {str(category): float(value) for category, value in zip(categories, values[local], strict=True)}}
                for local, index in enumerate(splits[name])
            ]
            for name, values in (("validation", validation_values), ("control", control_values))
        },
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "src/subcategory_extraction/semantic_encoder_f1/threshold_search.py",
            ROOT / "src/subcategory_extraction/semantic_encoder/train.py",
            ROOT / "src/subcategory_extraction/semantic_encoder/model.py",
            ROOT / "src/subcategory_extraction/semantic_encoder/pipeline.py",
            ROOT / "src/subcategory_extraction/semantic_encoder_calibrated/pipeline.py",
        )},
        "total_seconds": time.perf_counter() - started,
    }
    save_json(report_path, report)
    logger.info("Validation выбрала %s: %s", selection["selected_method"], calibration["thresholds"])
    logger.info("Control: %s", {name: result["micro_f1"] for name, result in report["control_metrics"].items()})
    return report


def main() -> None:
    """Запустить один воспроизводимый опыт; повтор не изменяет прежние результаты."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="manual_train",
                        help="таблица PostgreSQL с обучающей разметкой (по умолчанию manual_train)")
    parser.add_argument("--output", type=Path, default=ROOT / "out/rubert-thresholds")
    parser.add_argument("--report", type=Path, default=ROOT / "out/rubert-thresholds/results.json")
    parser.add_argument("--replay", action="store_true", help="Пересчитать сохранённые ответы без обучения")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.replay:
        replay_recorded_report(Path(__file__).parent, "rubert_thresholds", args.report)
        return
    run(args.table, args.output.resolve(), args.report.resolve(), args.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
