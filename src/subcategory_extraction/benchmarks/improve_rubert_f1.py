"""Полный план RuBERT: validation-only отбор, локальный учитель, единая CPU-оценка."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import resource
import shutil
from statistics import median
import time

import numpy as np
import torch

from main import MICROCATEGORIES, PipelineConfig, PipelineInput, load_pipeline
from run import audit_overlap, calculate_metrics, prepare_rows, read_records
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.benchmarks.robust_thresholds import (
    binary_metrics, select_robust_thresholds, validation_groups,
)
from subcategory_extraction.benchmarks.train_encoder_f1 import fit_encoder, predict_logits
from subcategory_extraction.benchmarks.train_semantic_encoder import load_training_rows
from subcategory_extraction.semantic_encoder.model import MODEL_ID, MODEL_REVISION, select_device
from subcategory_extraction.semantic_encoder.pipeline import SubcategoryExtractionPipeline as Baseline
from subcategory_extraction.semantic_encoder_f1.model import DEFAULT_ARCHITECTURE, load_artifact
from subcategory_extraction.semantic_encoder_f1.pipeline import SubcategoryExtractionPipeline

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CATEGORIES = list(MICROCATEGORIES)
ADDITIONAL = [category for category in CATEGORIES if category != 101]
TEACHER_ID = "DeepPavlov/rubert-base-cased"
TEACHER_REVISION = "4036cab694767a299f2b9e6492909664d9414229"
REVIEW = HERE / "results/rubert-hard-negatives.json"
logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    """Зафиксировать вход/выход без чтения секретов."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    """Атомарно сохранить отчёт в принадлежащем опыту каталоге."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_memory() -> None:
    """Освободить промежуточные модели между независимыми кандидатами."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Вычислить те же FP32 sigmoid-вероятности, что production PyTorch на CPU."""
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).sigmoid().numpy().astype(np.float64)


def row_labels(rows: list[dict]) -> np.ndarray:
    """Построить булевы дополнительные метки в фиксированном порядке."""
    return np.array([[category in row["targets"] for category in ADDITIONAL] for row in rows], dtype=bool)


def candidate_probabilities(artifact: Path, rows: list[dict], device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Загрузить checkpoint без учителя и вычислить реальные logits."""
    actual_device = select_device(device)
    model, tokenizer, metadata = load_artifact(artifact, actual_device)
    values = predict_logits(model, tokenizer, rows, metadata["architecture"], actual_device, batch_size)
    del model, tokenizer
    release_memory()
    return sigmoid(values)[:, 1:]


def calibrate(artifact: Path, validation: list[dict], groups: np.ndarray, validation_sha256: str) -> dict:
    """Зафиксировать threshold-only crossfit до любого доступа к control inference."""
    values = candidate_probabilities(artifact, validation)
    calibration = {
        "version": 1, "artifact_metadata_sha256": digest(artifact / "metadata.json"),
        **select_robust_thresholds(values, row_labels(validation), ADDITIONAL, groups),
        "validation_sha256": validation_sha256,
    }
    save_json(artifact / "calibration.json", calibration)
    return calibration


def spec(name: str, *, architecture: dict | None = None, lr: float = 3e-5,
         seed: int = 42, reviewed: bool = False, distilled: bool = False) -> dict:
    """Описать воспроизводимого студента без данных control/test."""
    return {"name": name, "architecture": dict(architecture or DEFAULT_ARCHITECTURE),
            "learning_rate": lr, "seed": seed, "reviewed": reviewed, "distilled": distilled}


def train_candidate(
    specification: dict, output: Path, train_rows: list[dict], reviewed_rows: list[dict],
    weights: np.ndarray, validation: list[dict], groups: np.ndarray,
    validation_sha256: str, device: str, teacher_logits: np.ndarray | None = None,
) -> dict:
    """Обучить один заранее заданный вариант и настроить его только на validation."""
    artifact = output / specification["name"]
    result_path = artifact / "candidate.json"
    signature = {
        "specification": specification, "train_source_sha256": digest(output / "train.jsonl"),
        "validation_sha256": validation_sha256,
        "review_sha256": digest(REVIEW) if specification["reviewed"] else None,
        "teacher_logits_sha256": digest(output / "teacher-train-logits.npy") if specification["distilled"] else None,
    }
    # Повтор после прерванного внешнего процесса допускает только полностью
    # завершённый checkpoint с идентичной спецификацией, а не фиктивный fallback.
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature or digest(artifact / "metadata.json") != result["calibration"]["artifact_metadata_sha256"]:
            raise ValueError(f"Несовместимый завершённый кандидат {artifact}")
        return result
    if specification["distilled"] and teacher_logits is None:
        raise ValueError("Дистилляция требует настоящие train-only teacher logits")
    training = fit_encoder(
        reviewed_rows if specification["reviewed"] else train_rows, validation, artifact,
        model_id=MODEL_ID, model_revision=MODEL_REVISION,
        architecture=specification["architecture"], learning_rate=specification["learning_rate"],
        seed=specification["seed"], device=device,
        sample_weights=weights if specification["reviewed"] else None,
        teacher_logits=teacher_logits if specification["distilled"] else None,
    )
    release_memory()
    calibration = calibrate(artifact, validation, groups, validation_sha256)
    result = {
        "signature": signature, "specification": specification,
        "artifact": str(artifact.resolve()), "training": training,
        "calibration": calibration,
    }
    save_json(result_path, result)
    logger.info("%s: epoch%d, validation fixed %.6f, threshold-crossfit %.6f, %s", specification["name"], training["best_epoch"],
                training["best_validation_micro_f1"], calibration["selection_metric"]["micro_f1"], calibration["selected_method"])
    return result


def rank(result: dict) -> float:
    """Отбирать конфигурацию исключительно по threshold-crossfit validation micro-F1."""
    return result["calibration"]["selection_metric"]["micro_f1"]


def prepare(output: Path) -> dict:
    """Скопировать старое разбиение, не создавая удобный новый control после просмотра."""
    output.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "out/rubert-thresholds"
    manifest = json.loads((previous / "split-manifest.json").read_text(encoding="utf-8"))
    source = ROOT / "docs/improved_dataset/train.jsonl"
    if manifest["source_sha256"] != digest(source):
        raise ValueError("Прежнее разбиение не соответствует актуальному source train")
    for name in ("train", "validation", "control"):
        original = previous / f"{name}.jsonl"
        if digest(original) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Изменилось прежнее разбиение {name}")
        destination = output / original.name
        if destination.exists() and digest(destination) != digest(original):
            raise ValueError(f"Нельзя заменить существующее разбиение {destination}")
        if not destination.exists():
            shutil.copyfile(original, destination)
    save_json(output / "split-manifest.json", manifest)
    protocol = {
        "version": 1, "source_dataset_sha256": digest(source),
        "split": "reuse previous grouped3196/1043/612; no new split seed",
        "control_status": "previously observed internal control; not a new independent holdout",
        "selection": "validation only; checkpoint fixed-threshold F1, configuration threshold-only group-crossfit F1",
        "learning_rates": [1e-5, 3e-5, 5e-5], "max_epochs": 6, "min_epochs": 2, "patience": 2,
        "warmup_ratio": 0.1, "finalist_seeds": [42, 43, 44], "production_seed": 42,
        "teacher": {"model_id": TEACHER_ID, "revision": TEACHER_REVISION,
                    "architecture": {"pooling": "mean", "max_tokens": 256, "max_windows": 3, "strategy": "coverage"},
                    "learning_rate": 2e-5, "batch_size": 8, "max_epochs": 3, "min_epochs": 2, "patience": 1},
        "distillation": {"alpha": 0.5, "temperature": 2.0, "targets": "teacher train logits + reviewed hard labels"},
        "diagnostic": "all119 reviewed descriptions evaluated once after frozen selection; not independent manual gold",
        "cpu": "single compact student, no teacher/LLM/API dependency at inference",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Нельзя менять протокол запущенного опыта")
    save_json(protocol_path, protocol)
    return manifest


def train_plan(output: Path, device: str) -> dict:
    """Выполнить LR/checkpoint, окна, review, teacher/distillation и seed-финалистов."""
    from subcategory_extraction.benchmarks.review_hard_negatives import load_reviewed_rows

    manifest = prepare(output)
    train_rows, _ = load_training_rows(output / "train.jsonl")
    validation, validation_hash = load_training_rows(output / "validation.jsonl")
    reviewed, weights, review_summary = load_reviewed_rows(train_rows, REVIEW)
    groups = validation_groups(validation)
    all_results = []

    def fit(specification: dict, teacher_logits: np.ndarray | None = None) -> dict:
        result = train_candidate(specification, output, train_rows, reviewed, weights, validation,
                                 groups, validation_hash, device, teacher_logits)
        all_results.append(result)
        save_json(output / "training-progress.json", {"candidates": all_results, "review": review_summary})
        return result

    grid = [fit(spec(f"mean-lr-{lr:g}", lr=lr)) for lr in (1e-5, 3e-5, 5e-5)]
    best_mean = max(grid, key=rank)
    best_lr = best_mean["specification"]["learning_rate"]
    attention = fit(spec("coverage-attention", lr=best_lr, architecture={
        "pooling": "attention", "max_tokens": 512, "max_windows": 6, "strategy": "coverage",
    }))
    best_architecture = max([best_mean, attention], key=rank)["specification"]["architecture"]
    reviewed_result = fit(spec("reviewed-hard-negatives", architecture=best_architecture, lr=best_lr, reviewed=True))

    teacher_path = output / "teacher"
    teacher_report_path = teacher_path / "train-report.json"
    if teacher_report_path.exists():
        teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    else:
        teacher_report = fit_encoder(
            reviewed, validation, teacher_path, model_id=TEACHER_ID, model_revision=TEACHER_REVISION,
            architecture={"pooling": "mean", "max_tokens": 256, "max_windows": 3, "strategy": "coverage"},
            learning_rate=2e-5, batch_size=8, max_epochs=3, min_epochs=2, patience=1,
            sample_weights=weights, device=device,
        )
        release_memory()
    logits_path = output / "teacher-train-logits.npy"
    if logits_path.exists():
        teacher_logits = np.load(logits_path, allow_pickle=False)
    else:
        actual_device = select_device(device)
        teacher_model, teacher_tokenizer, teacher_metadata = load_artifact(teacher_path, actual_device)
        teacher_logits = predict_logits(teacher_model, teacher_tokenizer, reviewed, teacher_metadata["architecture"], actual_device, batch_size=8)
        np.save(logits_path, teacher_logits, allow_pickle=False)
        del teacher_model, teacher_tokenizer
        release_memory()
    distilled = fit(spec("distilled", architecture=best_architecture, lr=best_lr, reviewed=True, distilled=True), teacher_logits)
    selected = max(all_results, key=rank)
    winning_spec = selected["specification"]
    replicas = [selected]
    for seed in (43, 44):
        replica_spec = {**winning_spec, "name": f"finalist-seed-{seed}", "seed": seed}
        replicas.append(fit(replica_spec, teacher_logits if winning_spec["distilled"] else None))
    # Seed42 закреплён заранее: не выбираем самый удачный seed по validation.
    selected_path = output / "selected"
    if not selected_path.exists():
        shutil.copytree(Path(selected["artifact"]), selected_path)
    if digest(selected_path / "metadata.json") != selected["calibration"]["artifact_metadata_sha256"]:
        raise ValueError("Selected artifact не соответствует заранее выбранному seed42")
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(output / "protocol.json"),
        "split_manifest_sha256": digest(output / "split-manifest.json"), "split_manifest": manifest,
        "review": review_summary, "review_sha256": digest(REVIEW),
        "selection_data": "validation only; all control/test metrics still unavailable to this function",
        "selected_candidate": selected["specification"]["name"],
        "selected_artifact": str(selected_path.resolve()),
        "selected_metadata_sha256": digest(selected_path / "metadata.json"),
        "selected_calibration_sha256": digest(selected_path / "calibration.json"),
        "teacher": teacher_report, "teacher_logits_sha256": digest(logits_path),
        "candidates": all_results,
        "seed_validation_micro_f1": {str(result["specification"]["seed"]): result["training"]["best_validation_micro_f1"] for result in replicas},
        "seed_threshold_crossfit_micro_f1": {str(result["specification"]["seed"]): rank(result) for result in replicas},
    }
    save_json(output / "frozen-selection.json", frozen)
    logger.info("Выбор заморожен до оценки: %s, threshold-crossfit validation %.6f", frozen["selected_candidate"], rank(selected))
    return frozen


def predicted_sets(probabilities: np.ndarray, thresholds: dict) -> list[set[int]]:
    """Применить неизменяемые пороги к дополнительным категориям."""
    boundary = np.array([thresholds[str(category)] for category in ADDITIONAL])
    return [{category for category, include in zip(ADDITIONAL, row, strict=True) if include}
            for row in probabilities >= boundary]


def metrics(rows: list[dict], predicted: list[set[int]]) -> dict:
    """Сопоставить официальный sklearn micro-F1 и независимые счётчики."""
    expected = [row["targets"] for row in rows]
    official = calculate_metrics(expected, predicted)
    direct = score(expected, predicted)
    if any(official[key] != direct[key] for key in ("tp", "fp", "fn")) or abs(official["micro_f1"] - direct["micro_f1"]) > 1e-12:
        raise RuntimeError("Реализации метрики расходятся")
    return {**official, "by_category": direct["by_category"],
            "exact_match_rows": sum(truth == output for truth, output in zip(expected, predicted, strict=True)),
            "empty_target_rows": sum(not truth for truth in expected),
            "empty_target_false_positive_rows": sum(not truth and bool(output) for truth, output in zip(expected, predicted, strict=True))}


def baseline_predictions(artifact: Path, rows: list[dict]) -> list[set[int]]:
    """Запустить настоящий исходный RuBERT с порогом0.5 на CPU."""
    os.environ["SEMANTIC_ENCODER_DEVICE"] = "cpu"
    pipeline = Baseline(artifact_path=artifact)
    return [set(pipeline.run(PipelineInput(description=row["description"], mcId=101))) for row in rows]


def evaluate_plan(output: Path, report_path: Path) -> dict:
    """Один раз оценить уже зафиксированные варианты, ничего не переотбирая."""
    if report_path.exists():
        raise FileExistsError("Общая оценка уже существует; не повторяем подбор по просмотренному control/test")
    frozen_path = output / "frozen-selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_hash = digest(frozen_path)
    control, _ = load_training_rows(output / "control.jsonl")
    diagnostic_path = ROOT / "docs/improved_dataset/reviewed_test.parquet"
    diagnostic = [{"description": row.inp.description, "targets": set(row.expected)}
                  for row in prepare_rows(read_records(diagnostic_path), 101)]
    diagnostic_inputs = prepare_rows(read_records(diagnostic_path), 101)
    overlap = audit_overlap(diagnostic_inputs, ROOT / "docs/improved_dataset/train.jsonl")
    if overlap["overlapping_rows"]:
        raise ValueError("Диагностика пересекается с source train")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_selection_sha256_before_evaluation": frozen_hash,
        "protocol": json.loads((output / "protocol.json").read_text(encoding="utf-8")),
        "frozen_selection": frozen, "diagnostic_sha256": digest(diagnostic_path),
        "diagnostic_rows": len(diagnostic), "control_rows": len(control), "diagnostic_overlap": overlap,
        "results": {},
    }
    split_baseline = ROOT / "out/rubert-thresholds/encoder"
    report["results"]["original-split-baseline"] = {
        "control": metrics(control, baseline_predictions(split_baseline, control)),
        "diagnostic": metrics(diagnostic, baseline_predictions(split_baseline, diagnostic)),
        "training_scope": "same3196 rows, original fixed3epochs, last checkpoint, threshold0.5",
    }
    full_baseline = ROOT / "out/improved-dataset/semantic-encoder"
    report["results"]["original-full-train-baseline"] = {
        "diagnostic": metrics(diagnostic, baseline_predictions(full_baseline, diagnostic)),
        "control": None, "training_scope": "all4851 rows; control is training data, so no control F1 reported",
    }
    for result in frozen["candidates"]:
        name = result["specification"]["name"]
        artifact = Path(result["artifact"])
        calibration = json.loads((artifact / "calibration.json").read_text(encoding="utf-8"))
        if digest(artifact / "metadata.json") != calibration["artifact_metadata_sha256"]:
            raise ValueError(f"Порог не соответствует checkpoint {name}")
        candidate_result = {"specification": result["specification"], "thresholds": calibration["thresholds"], "threshold_method": calibration["selected_method"]}
        for split_name, rows in (("control", control), ("diagnostic", diagnostic)):
            values = candidate_probabilities(artifact, rows)
            candidate_result[split_name] = metrics(rows, predicted_sets(values, calibration["thresholds"]))
            candidate_result[f"{split_name}_fixed_0_5"] = metrics(rows, predicted_sets(values, {str(cid): 0.5 for cid in CATEGORIES}))
        report["results"][name] = candidate_result
        logger.info("Общая оценка %s: control %.6f, diagnostic119 %.6f", name, candidate_result["control"]["micro_f1"], candidate_result["diagnostic"]["micro_f1"])
    # Отдельно проверяем фактическую production поверхность: динамическая загрузка,
    # обязательный CPU, реальные run(), никакого учителя или API.
    os.environ["RUBERT_F1_ARTIFACT_PATH"] = str(output / "selected")
    os.environ["RUBERT_F1_DEVICE"] = "cpu"
    tick = time.perf_counter()
    pipeline = load_pipeline("subcategory_extraction", PipelineConfig(subcategory_extraction="semantic_encoder_f1", draft_generation=None))
    load_seconds = time.perf_counter() - tick
    actual, durations, observations = [], [], []
    for number, row in enumerate(diagnostic, start=1):
        tick = time.perf_counter()
        prediction = set(pipeline.run(PipelineInput(description=row["description"], mcId=101)))
        durations.append(time.perf_counter() - tick)
        actual.append(prediction)
        observations.append({"row_number": number, "description_sha256": hashlib.sha256(row["description"].encode()).hexdigest(),
                             "expected": sorted(row["targets"]), "predicted": sorted(prediction)})
    actual_metrics = metrics(diagnostic, actual)
    selected_result = report["results"][frozen["selected_candidate"]]
    if actual_metrics != selected_result["diagnostic"]:
        raise RuntimeError("Реальный production run расходится с зафиксированным checkpoint/порогами")
    durations.sort()
    if frozen_hash != digest(frozen_path):
        raise RuntimeError("Выбор изменился после доступа к оценке")
    report["selected"] = {"candidate": frozen["selected_candidate"], "diagnostic": actual_metrics,
                          "control": selected_result["control"], "metadata": pipeline.model_metadata}
    report["cpu"] = {"device": pipeline.device, "run_calls": len(diagnostic), "api_requests": 0,
                     "model_load_seconds": load_seconds,
                     "median_ms": median(durations) * 1000,
                     "p95_nearest_rank_ms": durations[math.ceil(0.95 * len(durations)) - 1] * 1000,
                     "mean_ms": sum(durations) / len(durations) * 1000,
                     "torch_threads": torch.get_num_threads(),
                     "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "rss_scope": "macOS process peak across evaluation of several models; not isolated student-only RAM",
                     "latency_scope": "sequential actual production run on119; excludes loading, includes first call"}
    report["diagnostic_observations"] = observations
    report["sources"] = {str(path.relative_to(ROOT)): digest(path) for path in (
        Path(__file__), HERE / "train_encoder_f1.py", HERE / "robust_thresholds.py", HERE / "review_hard_negatives.py",
        ROOT / "src/subcategory_extraction/semantic_encoder_f1/model.py", ROOT / "src/subcategory_extraction/semantic_encoder_f1/pipeline.py",
    )}
    save_json(report_path, report)
    logger.info("Selected %s: diagnostic119 micro-F1 %.6f, CPU median %.3f ms", frozen["selected_candidate"], actual_metrics["micro_f1"], report["cpu"]["median_ms"])
    return report


def main() -> None:
    """Выполнить весь план либо явно разделить обучение и однократную оценку."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out/rubert-f1-plan")
    parser.add_argument("--report", type=Path, default=HERE / "results/rubert-f1-plan.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("train", "all"):
        train_plan(args.output.resolve(), args.device)
    if args.stage in ("evaluate", "all"):
        evaluate_plan(args.output.resolve(), args.report.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
