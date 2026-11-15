"""Фрагменты, grounded teacher review, дистилляция и выбор CPU-ансамбля по OOF."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import torch

from main import MICROCATEGORIES
from run import normalize_description
from subcategory_extraction.semantic.semantic_encoder.model import select_device
from subcategory_extraction.semantic.semantic_encoder.train import load_training_rows
from subcategory_extraction.semantic.semantic_encoder_calibrated.train import split_rows
from subcategory_extraction.semantic.semantic_encoder_f1.robust_thresholds import binary_metrics, select_robust_thresholds, validation_groups
from subcategory_extraction.semantic.semantic_evidence.model import load_artifact, predict_logits
from subcategory_extraction.semantic.semantic_evidence.train import CLASSES, digest, fit, labels, release_memory, rows_digest, save_json

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/improved_dataset/train.jsonl"
DEFAULT_OUTPUT = ROOT / "out/semantic-evidence-f1"
SPECIFICATIONS = {
    "document": {"mode": "document", "max_tokens": 512, "stride": 510},
    "evidence": {"mode": "evidence", "max_tokens": 256, "stride": 192},
    "distilled": {"mode": "evidence", "max_tokens": 256, "stride": 192},
}
logger = logging.getLogger(__name__)


def select_rows(rows: list[dict], indices: np.ndarray) -> list[dict]:
    """Сохранить исходный порядок выделенной train-части."""
    return [rows[int(index)] for index in indices]


def probabilities(artifact: Path, rows: list[dict], device: str) -> np.ndarray:
    """Вычислить реальные предсказания сохранённого checkpoint."""
    actual = select_device(device)
    model, tokenizer, metadata = load_artifact(artifact, actual)
    logits = predict_logits(model, tokenizer, rows, metadata["architecture"], actual, batch_size=8)
    result = torch.from_numpy(logits.astype(np.float32)).sigmoid().numpy()
    del model, tokenizer
    release_memory()
    return result


def cached_probabilities(path: Path, artifact: Path, rows: list[dict], device: str) -> np.ndarray:
    """Переиспользовать только вероятности идентичных входов и весов."""
    stamp = {"rows_sha256": rows_digest(rows), "metadata_sha256": digest(artifact / "metadata.json"),
             "encoder_sha256": digest(artifact / "encoder/model.safetensors"), "head_sha256": digest(artifact / "head.safetensors")}
    manifest = path.with_suffix(".json")
    if path.exists():
        if json.loads(manifest.read_text()) != stamp:
            raise ValueError(f"Incompatible probability cache: {path}")
        values = np.load(path, allow_pickle=False)
    else:
        values = probabilities(artifact, rows, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
        save_json(manifest, stamp)
    if values.shape != (len(rows), len(CLASSES)) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid cached model probabilities")
    return values


def protect_user_decisions(rows: list[dict]) -> None:
    """Не позволить машинному review отменять индивидуальные ответы пользователя."""
    decisions = json.loads((ROOT / "docs/improved_dataset/decisions.json").read_text())
    entries = {entry["row"]: entry for entry in decisions["train"]
               if entry.get("review_type") in {"user_cli_review", "user_chat_review"}}
    protected = set()
    with (ROOT / "docs/innoglobalhack_avito/public-train-annotated.jsonl").open() as stream:
        for number, line in enumerate(stream, start=1):
            if number not in entries:
                continue
            record = json.loads(line)
            entry = entries[number]
            if record["item_id"] != entry["item_id"]:
                raise ValueError("Protected user decision source item changed")
            group_hash = hashlib.sha256(normalize_description(record["description"]).encode()).hexdigest()
            if entry.get("description_group_id", group_hash) != group_hash:
                raise ValueError("Protected user decision description changed")
            protected.add(group_hash)
    if len(protected) != len(entries):
        raise ValueError("Missing or duplicate protected user decision source")
    for row in rows:
        group_hash = hashlib.sha256(normalize_description(row["description"]).encode()).hexdigest()
        row["protected_user_decision"] = group_hash in protected


def prepare(output: Path) -> tuple[list[dict], list[dict], np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Воспроизвести прежний control, разделяя весь development на близкие группы."""
    all_rows, source_hash = load_training_rows(SOURCE)
    protect_user_decisions(all_rows)
    splits, grouping = split_rows(all_rows)
    development_indices = np.sort(np.concatenate((splits["train"], splits["validation"])))
    control_indices = splits["control"]
    groups = validation_groups(all_rows)
    if set(groups[development_indices]) & set(groups[control_indices]):
        raise RuntimeError("Near-duplicate components cross development/control")
    development = select_rows(all_rows, development_indices)
    control = select_rows(all_rows, control_indices)
    development_groups = groups[development_indices]
    folds = list(GroupKFold(n_splits=3).split(development_indices, groups=development_groups))
    protocol = {
        "version": 1, "source_sha256": source_hash, "policy_sha256": digest(ROOT / "docs/improved_dataset/annotation_policy.json"),
        "development_source_rows": (development_indices + 1).tolist(), "control_source_rows": (control_indices + 1).tolist(),
        "development_rows": len(development), "control_rows": len(control), "grouping": grouping,
        "control_status": "reused previously observed internal control, not independent holdout",
        "folds": [{"train": train.tolist(), "heldout": heldout.tolist()} for train, heldout in folds],
        "encoder_oof": "fixed five epochs, each heldout fold excluded from encoder training and checkpoint selection",
        "specifications": SPECIFICATIONS, "epochs": 5, "learning_rate": 5e-5, "seed": 42,
        "review_limit": 96, "review_scope": "development only; protected user decisions unchanged",
        "teacher": "fold-local document/evidence ensemble plus grounded high-confidence external training review",
        "teacher_leakage_guard": "student teacher predictions generated by models trained ONLY on the student's training fold",
        "distillation_alpha": 0.3,
        "selection": "group-crossfit calibration on encoder OOF; not fully nested ensemble CV",
        "test_policy": "both existing public test scenarios run only after all variants/weights/thresholds are frozen",
        "deployment": "CPU-only single student or weighted compact ensemble; no inference API",
        "target_f1": "0.92-0.93 is a goal, not a guaranteed result",
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Refusing to change an existing experimental protocol")
    save_json(path, protocol)
    return development, control, development_groups, folds


def tfidf_fit(rows: list[dict]) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Обучить тот же word TF-IDF baseline исключительно на train fold."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    features = vectorizer.fit_transform([row["description"] for row in rows])
    truth = labels(rows)
    weights = np.zeros((len(CLASSES), features.shape[1]))
    intercepts = np.full(len(CLASSES), -40.0)
    for column in range(1, len(CLASSES)):
        values = truth[:, column]
        if len(np.unique(values)) == 1:
            intercepts[column] = 40.0 if values[0] else -40.0
            continue
        classifier = LogisticRegression(C=4, max_iter=500, random_state=42).fit(features, values)
        weights[column] = classifier.coef_[0]
        intercepts[column] = classifier.intercept_[0]
    return vectorizer, weights, intercepts


def tfidf_probabilities(resource: tuple, rows: list[dict]) -> np.ndarray:
    """Оценить sparse baseline на неизменённых полных описаниях."""
    vectorizer, weights, intercepts = resource
    logits = vectorizer.transform([row["description"] for row in rows]) @ weights.T + intercepts
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def export_tfidf(resource: tuple, path: Path, training: dict) -> None:
    """Сохранить совместимый JSON baseline; исходные старые веса не переписывать."""
    vectorizer, weights, intercepts = resource
    save_json(path, {"version": 1, "categories": list(CLASSES), "trained_categories": list(CLASSES[1:]),
                     "threshold": 0.5, "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                     "idf": vectorizer.idf_.tolist(), "weights": weights.tolist(), "intercepts": intercepts.tolist(),
                     "training": training})


def calibrate(values: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict:
    """Подбирать пороги по OOF, без доступа к внутреннему control и двум test."""
    result = select_robust_thresholds(values[:, 1:], truth[:, 1:].astype(bool), list(CLASSES[1:]), groups)
    result["encoder_outputs_are_oof"] = True
    result["selection_data"] = "encoder OOF on development; no control/public-test values"
    result["method"] = "grouped_encoder_OOF_then_threshold_GroupKFold3_shrinkage"
    return result


def metric(values: np.ndarray, truth: np.ndarray, thresholds: dict) -> dict:
    """Считать micro-F1 только дополнительных категорий в одном порядке."""
    boundary = np.array([thresholds[str(c)] for c in CLASSES[1:]])
    return binary_metrics(truth[:, 1:].astype(bool), values[:, 1:] >= boundary)


def select_composition(oof: dict[str, np.ndarray], truth: np.ndarray, groups: np.ndarray) -> dict:
    """Сравнить одиночных студентов и заранее заданную небольшую сетку ансамбля."""
    names = list(SPECIFICATIONS)
    candidates = []
    compositions = [
        np.array([a, b, 4 - a - b], dtype=float) / 4
        for a in range(5) for b in range(5 - a)
    ] + [np.full(3, 1 / 3)]
    for weights in compositions:
        values = sum(weight * oof[name] for name, weight in zip(names, weights, strict=True))
        calibration = calibrate(values, truth, groups)
        candidates.append({"weights": {name: float(weight) for name, weight in zip(names, weights, strict=True) if weight},
                           "calibration": calibration})
    # Deterministic tie-break: fewer components, then insertion order.
    winner = max(candidates, key=lambda item: (item["calibration"]["selection_metric"]["micro_f1"], -len(item["weights"])))
    singles = {name: next(item for item in candidates if item["weights"] == {name: 1.0}) for name in names}
    ensemble = max((item for item in candidates if len(item["weights"]) > 1),
                   key=lambda item: item["calibration"]["selection_metric"]["micro_f1"])
    return {"selected": winner, "single_students": singles, "ensemble": ensemble, "candidates": candidates,
            "limitation": "encoder OOF is real, but weights/configuration selection is not an unbiased nested-CV estimate"}


def teacher_targets(rows: list[dict], report: dict, local: np.ndarray) -> np.ndarray:
    """Сохранить fold-local soft targets; grounded принятые review дают мягкую коррекцию."""
    result = local.copy()
    accepted = {
        entry["provenance"]["description_sha256"]: entry
        for entry in report["decisions"]
        if entry["action"] in {"teacher_proposal_applied_not_gold", "high_confidence_hard_case_agreement"}
    }
    for index, row in enumerate(rows):
        entry = accepted.get(hashlib.sha256(row["description"].encode()).hexdigest())
        if entry is not None and not row.get("protected_user_decision", False):
            targets = set(entry["after"])
            confidence = min(0.98, max(0.9, float(entry["confidence_min"])))
            result[index] = [confidence if category in targets else 1 - confidence for category in CLASSES]
    result[:, 0] = 0.0
    return result


def export_bundle(path: Path, artifacts: dict[str, Path], composition: dict, provenance: dict,
                  source_row_numbers: list[int]) -> None:
    """Экспортировать все runtime-компоненты рядом с selection.json, без ссылок на out."""
    if path.exists():
        existing = json.loads((path / "selection.json").read_text())
        if existing.get("provenance") != provenance:
            raise ValueError(f"Cannot replace an unrelated runtime bundle: {path}")
        return
    components = []
    path.mkdir(parents=True)
    for name, weight in composition["weights"].items():
        source = artifacts[name]
        destination = path / name
        destination.mkdir()
        for relative in ("encoder", "tokenizer"):
            shutil.copytree(source / relative, destination / relative)
        for relative in ("head.safetensors", "metadata.json"):
            shutil.copyfile(source / relative, destination / relative)
        components.append({"path": name, "weight": weight,
                           "metadata_sha256": digest(destination / "metadata.json"),
                           "head_sha256": digest(destination / "head.safetensors"),
                           "encoder_sha256": digest(destination / "encoder/model.safetensors")})
    save_json(path / "selection.json", {"version": 1, "classes": list(CLASSES), "components": components,
                                       "thresholds": composition["calibration"]["thresholds"], "provenance": provenance,
                                       "training": {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE),
                                                    "source_row_numbers": source_row_numbers}})


def train_plan(output: Path, device: str, stage: str, resume_review: Path | None = None,
               allow_partial_review: bool = False, artifact_path: Path | None = None) -> dict:
    """Обучить настоящие OOF-модели, затем заморозить выбор до контрольной оценки."""
    development, control, groups, folds = prepare(output)
    truth = labels(development)
    protocol_hash = digest(output / "protocol.json")
    source_row_numbers = json.loads((output / "protocol.json").read_text())["development_source_rows"]
    oof = {name: np.full((len(development), len(CLASSES)), np.nan, dtype=np.float32) for name in SPECIFICATIONS}
    baseline_oof = np.full_like(oof["document"], np.nan)
    for number, (training, heldout) in enumerate(folds):
        train_rows = select_rows(development, training)
        heldout_rows = select_rows(development, heldout)
        fold_path = output / f"fold-{number}"
        for name in ("document", "evidence"):
            artifact = fold_path / name
            fit(train_rows, artifact, SPECIFICATIONS[name], device=device,
                provenance={"protocol_sha256": protocol_hash, "fold": number, "heldout_rows_not_used": True})
            oof[name][heldout] = cached_probabilities(fold_path / f"{name}-heldout.npy", artifact, heldout_rows, device)
            save_json(output / "progress.json", {"stage": "base_oof", "fold": number, "component": name})
        baseline_oof[heldout] = tfidf_probabilities(tfidf_fit(train_rows), heldout_rows)
    for name in ("document", "evidence"):
        np.save(output / f"oof-{name}.npy", oof[name], allow_pickle=False)
    np.save(output / "oof-tfidf.npy", baseline_oof, allow_pickle=False)
    if stage == "base":
        return {"stage": "base_complete", "development_rows": len(development)}

    from dataset_annotation.semantic_evidence_review import apply_review, review_training_rows
    review_path = output / "teacher-review.json"
    if stage == "review":
        previous = None if resume_review is None else json.loads(resume_review.read_text())
        return review_training_rows(development, (oof["document"] + oof["evidence"]) / 2,
                                    review_path, limit=96, previous_report=previous)
    if review_path.exists():
        external_review = json.loads(review_path.read_text())
        if external_review.get("status") not in {"completed", "failed"}:
            raise ValueError("Interrupted teacher review must finish before training students")
    else:
        external_review = review_training_rows(development, (oof["document"] + oof["evidence"]) / 2,
                                               review_path, limit=96)
    partial_review = external_review.get("status") != "completed" or external_review.get("processed_rows") != 96
    if partial_review and (not allow_partial_review or external_review.get("completed_rows", 0) < 1):
        raise RuntimeError("Teacher queue is incomplete; finish review or explicitly accept --allow-partial-review")
    adjudication_path = ROOT / "src/dataset_annotation/semantic_evidence_adjudications.json"
    adjudication = json.loads(adjudication_path.read_text())
    if (adjudication["source_train_sha256"] != digest(SOURCE)
            or adjudication["independent_gold"] is not False):
        raise ValueError("Semantic adjudication source changed")
    reviewed, review_weights, review_summary = apply_review(
        development, external_review, adjudication["rejected_proposals"],
    )
    review_summary["adjudication_sha256"] = digest(adjudication_path)
    review_summary["requested_review_rows"] = 96
    review_summary["partial_review_explicitly_accepted"] = bool(partial_review and allow_partial_review)
    save_json(output / "applied-review.json", review_summary)
    # Independent review proposals never alter the original OOF evaluation labels.
    for number, (training, heldout) in enumerate(folds):
        fold_path = output / f"fold-{number}"
        train_rows = select_rows(development, training)
        reviewed_training = select_rows(reviewed, training)
        teacher = (cached_probabilities(fold_path / "document-train.npy", fold_path / "document", train_rows, device)
                   + cached_probabilities(fold_path / "evidence-train.npy", fold_path / "evidence", train_rows, device)) / 2
        teacher = teacher_targets(train_rows, review_summary, teacher)
        artifact = fold_path / "distilled"
        fit(reviewed_training, artifact, SPECIFICATIONS["distilled"], device=device,
            sample_weights=review_weights[training], teacher_probabilities=teacher,
            provenance={"protocol_sha256": protocol_hash, "fold": number, "heldout_rows_not_used": True,
                        "review_sha256": digest(output / "teacher-review.json"), "teacher_scope": "same training fold only"})
        oof["distilled"][heldout] = cached_probabilities(fold_path / "distilled-heldout.npy", artifact,
                                                       select_rows(development, heldout), device)
    np.save(output / "oof-distilled.npy", oof["distilled"], allow_pickle=False)
    if any(not np.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("Incomplete encoder OOF predictions")
    selection = select_composition(oof, truth, groups)
    base_teacher = (oof["document"] + oof["evidence"]) / 2
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
        **selection, "review_summary": review_summary,
        "teacher_oof_fixed_0.5": metric(base_teacher, truth, {str(c): 0.5 for c in CLASSES}),
        "tfidf_oof_fixed_0.5": metric(baseline_oof, truth, {str(c): 0.5 for c in CLASSES}),
        "test_evaluated": False,
    }
    frozen_path = output / "frozen-selection.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        frozen["frozen_at_utc"] = previous["frozen_at_utc"]
        if frozen != previous:
            raise ValueError("Cannot reselect an already frozen experiment")
    else:
        save_json(frozen_path, frozen)
    # Fit development-only models for the reused control; no test/holdout-driven reselection.
    artifacts = {}
    control_probabilities = {}
    development_teacher = []
    for name in ("document", "evidence"):
        artifact = output / "development" / name
        fit(development, artifact, SPECIFICATIONS[name], device=device,
            provenance={"protocol_sha256": protocol_hash, "scope": "development_only"})
        artifacts[name] = artifact
        control_probabilities[name] = cached_probabilities(output / f"control-{name}.npy", artifact, control, device)
        development_teacher.append(cached_probabilities(output / f"development-teacher-{name}.npy", artifact, development, device))
    development_teacher = teacher_targets(development, review_summary, sum(development_teacher) / 2)
    artifact = output / "development/distilled"
    fit(reviewed, artifact, SPECIFICATIONS["distilled"], device=device, sample_weights=review_weights,
        teacher_probabilities=development_teacher, provenance={"protocol_sha256": protocol_hash,
        "scope": "development_only", "review_sha256": digest(output / "teacher-review.json")})
    artifacts["distilled"] = artifact
    control_probabilities["distilled"] = cached_probabilities(output / "control-distilled.npy", artifact, control, device)
    control_truth = labels(control)
    control_results = {}
    variants = {**selection["single_students"], "ensemble": selection["ensemble"], "selected": selection["selected"]}
    for name, composition in variants.items():
        values = sum(weight * control_probabilities[component] for component, weight in composition["weights"].items())
        control_results[name] = metric(values, control_truth, composition["calibration"]["thresholds"])
        export_bundle(output / "bundles" / name, artifacts, composition,
                      {"protocol_sha256": protocol_hash, "frozen_selection_sha256": digest(output / "frozen-selection.json"),
                       "variant": name, "training_scope": "development_only", "selection": "encoder OOF only"},
                      source_row_numbers)
    resource = tfidf_fit(development)
    control_results["tfidf_fixed_0.5"] = metric(tfidf_probabilities(resource, control), control_truth, {str(c): 0.5 for c in CLASSES})
    export_tfidf(resource, output / "tfidf.json", {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE),
                                                "source_row_numbers": source_row_numbers, "development_only": True,
                                                "train_rows": len(development), "train_sha256": rows_digest(development)})
    # Production retains development-only artifacts: control never becomes training data.
    production_bundle = HERE / "artifacts" if artifact_path is None else artifact_path.resolve()
    export_bundle(production_bundle, artifacts, selection["selected"],
                  {"protocol_sha256": protocol_hash, "frozen_selection_sha256": digest(output / "frozen-selection.json"),
                   "variant": "selected", "training_scope": "development_only", "selection": "encoder OOF only"},
                  source_row_numbers)
    report = {"protocol_sha256": protocol_hash, "selection": selection, "control": control_results,
              "control_status": "previously observed internal control; not independent holdout",
              "review": review_summary, "teacher_review_status": external_review.get("status"),
              "production_bundle": str(production_bundle), "target_f1_is_not_guaranteed": True}
    save_json(output / "training-results.json", report)
    logger.info("Frozen selection: %s; reused-control results: %s", selection["selected"]["weights"], control_results)
    return report


def main() -> None:
    """Запустить фиксированный протокол; base позволяет выполнить review между стадиями."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact", type=Path, default=HERE / "artifacts",
                        help="Новый каталог итоговых весов; существующий чужой bundle не перезаписывается")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--stage", choices=("base", "review", "all"), default="all")
    parser.add_argument("--resume-review", type=Path, help="Сохранённый review для явного продолжения другим учителем")
    parser.add_argument("--allow-partial-review", action="store_true",
                        help="Явно принять сохранённую неполную проверку учителем; статус ошибки не скрывается")
    args = parser.parse_args()
    if args.resume_review is not None and args.stage != "review":
        parser.error("--resume-review допускается только со --stage review")
    result = train_plan(args.output.resolve(), args.device, args.stage, args.resume_review,
                        args.allow_partial_review, args.artifact)
    if args.stage == "review" and result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
