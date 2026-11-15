"""Дообучение encoder-кандидатов исключительно на train и validation."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    __version__ as transformers_version,
    get_linear_schedule_with_warmup,
)

from main import MICROCATEGORIES
from subcategory_extraction.semantic.semantic_encoder.model import (
    collate_documents,
    select_device,
)
from subcategory_extraction.semantic.semantic_encoder_f1.model import (
    WindowEncoder,
    save_artifact,
    tokenize_document,
    validate_architecture,
)

logger = logging.getLogger(__name__)
SOURCE_CATEGORY = 101
CLASSES = tuple(MICROCATEGORIES)


def _validate_rows(rows: list[dict], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} must contain at least one document")
    for index, row in enumerate(rows):
        if not isinstance(row.get("description"), str):
            raise ValueError(f"{name}[{index}] must have a string description")
        targets = row.get("targets")
        if not isinstance(targets, (set, frozenset)) or any(
            type(category) is not int or category not in CLASSES
            or category == SOURCE_CATEGORY for category in targets
        ):
            raise ValueError(f"{name}[{index}] must have additional-category targets excluding 101")


def _source_sha256(rows: list[dict]) -> str:
    """Хешировать упорядоченные описания и метки независимо от порядка set."""
    digest = hashlib.sha256()
    for row in rows:
        payload = {"description": row["description"], "targets": sorted(row["targets"])}
        digest.update(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _array_sha256(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def _prediction_batches(model, tokenizer, documents, device, batch_size):
    """Выдавать CPU-logits по порядку без сохранения графов устройства."""
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(documents), batch_size):
            stop = min(start + batch_size, len(documents))
            batch = collate_documents(tokenizer, documents[start:stop])
            values = model(batch.to(device))
            if values.shape != (stop - start, len(CLASSES)):
                raise ValueError("Encoder logits must have shape [documents, 6]")
            if not torch.isfinite(values).all().item():
                raise RuntimeError("Encoder produced non-finite prediction logits")
            yield values.detach().to(device="cpu", dtype=torch.float32).numpy()


def _predict_documents(model, tokenizer, documents, device, batch_size):
    if not documents:
        return np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(list(_prediction_batches(
        model, tokenizer, documents, device, batch_size,
    )), axis=0)


def predict_logits(
    model, tokenizer, rows: list[dict], architecture: dict,
    device: torch.device, batch_size: int = 16,
) -> np.ndarray:
    """Вычислить logits [N,6] с динамическим дополнением реальных окон."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    documents = [
        tokenize_document(tokenizer, row["description"], architecture)
        for row in rows
    ]
    return _predict_documents(model, tokenizer, documents, device, batch_size)


def _validation_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    included = [index for index, category in enumerate(CLASSES) if category != SOURCE_CATEGORY]
    predicted = logits[:, included] >= 0.0
    expected = targets[:, included].astype(bool)
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    denominator = 2 * tp + fp + fn
    return {
        "micro_f1": 2 * tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def fit_encoder(
    train_rows: list[dict], validation_rows: list[dict], artifact: Path, *,
    model_id: str, model_revision: str, architecture: dict,
    learning_rate: float = 3e-5, seed: int = 42, max_epochs: int = 6,
    min_epochs: int = 2, patience: int = 2, batch_size: int = 16,
    warmup_ratio: float = 0.1, device: str = "auto",
    sample_weights: np.ndarray | None = None,
    teacher_logits: np.ndarray | None = None,
    distillation_alpha: float = 0.5, temperature: float = 2.0,
) -> dict:
    """Обучить все веса и выбрать самый ранний лучший validation checkpoint.

    Веса примеров умножают среднюю по классам loss документа. Дистилляция
    использует только train-aligned soft targets. Validation использует
    исходные предварительные метки, не ручной gold; control/test не читаются.
    """
    started = time.perf_counter()
    artifact = Path(artifact).expanduser().resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact: {artifact}")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be nonempty")
    if not isinstance(model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision):
        raise ValueError("model_revision must be a pinned 40-character commit SHA")
    for name, value in (("batch_size", batch_size), ("max_epochs", max_epochs),
                        ("min_epochs", min_epochs), ("patience", patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_epochs > max_epochs:
        raise ValueError("min_epochs must not exceed max_epochs")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not math.isfinite(distillation_alpha) or not 0 <= distillation_alpha <= 1:
        raise ValueError("distillation_alpha must be in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    _validate_rows(train_rows, "train_rows")
    _validate_rows(validation_rows, "validation_rows")
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if sample_weights.shape != (len(train_rows),) or not np.all(
            np.isfinite(sample_weights) & (sample_weights > 0)
        ):
            raise ValueError("sample_weights must be finite, positive and train-aligned [N]")
    if teacher_logits is not None:
        teacher_logits = np.asarray(teacher_logits, dtype=np.float32)
        if teacher_logits.shape != (len(train_rows), len(CLASSES)) or not np.isfinite(teacher_logits).all():
            raise ValueError("teacher_logits must be finite and train-aligned [N, 6]")

    architecture = validate_architecture(architecture)
    training_device = select_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Reserve the directory once: subsequent checkpoint writes belong to this run.
    artifact.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        model_id, revision=model_revision,
    )
    model = WindowEncoder(encoder, len(CLASSES), architecture["pooling"]).to(training_device)
    model.requires_grad_(True)
    preparation_started = time.perf_counter()
    train_documents = [
        tokenize_document(tokenizer, row["description"], architecture) for row in train_rows
    ]
    validation_documents = [
        tokenize_document(tokenizer, row["description"], architecture) for row in validation_rows
    ]
    train_targets = torch.tensor([
        [float(category in row["targets"]) for category in CLASSES] for row in train_rows
    ], dtype=torch.float32)
    validation_targets = np.array([
        [float(category in row["targets"]) for category in CLASSES] for row in validation_rows
    ], dtype=np.float32)
    weights = torch.from_numpy(sample_weights) if sample_weights is not None else torch.ones(len(train_rows))
    # Compute the temperature-scaled teacher probabilities once, on CPU.
    teacher_probabilities = (
        torch.sigmoid(torch.from_numpy(teacher_logits) / temperature)
        if teacher_logits is not None else None
    )

    def collate(indices):
        batch = collate_documents(tokenizer, [train_documents[index] for index in indices])
        soft = teacher_probabilities[indices] if teacher_probabilities is not None else None
        return batch, train_targets[indices], weights[indices], soft

    loader = DataLoader(
        range(len(train_rows)), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    train_hash = _source_sha256(train_rows)
    validation_hash = _source_sha256(validation_rows)
    positives = {str(category): sum(category in row["targets"] for row in train_rows) for category in CLASSES}
    trained_classes = [category for category in CLASSES if positives[str(category)] > 0]
    training_config = {
        "seed": seed, "learning_rate": learning_rate, "batch_size": batch_size,
        "max_epochs": max_epochs, "min_epochs": min_epochs, "patience": patience,
        "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 1.0,
        "scheduler": "linear_warmup_linear_decay", "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps, "planned_steps": total_steps,
        "sample_weighting": "mean(weight * per_document_mean_class_loss)",
        "sample_weights_sha256": _array_sha256(sample_weights),
        "distillation": teacher_logits is not None,
        "teacher_logits_sha256": _array_sha256(teacher_logits),
        "distillation_alpha": distillation_alpha, "temperature": temperature,
        "selection_metric": "validation micro-F1 at 0.5, excluding source category 101",
        "tie_break": "earliest epoch", "train_sha256": train_hash,
        "validation_sha256": validation_hash,
    }
    metadata_base = {
        "version": 1, "model_id": model_id, "model_revision": model_revision,
        "classes": list(CLASSES), "architecture": architecture,
        "trained_classes": trained_classes,
        "unsupported_classes": [category for category in CLASSES if category not in trained_classes],
        "positive_counts": positives, "source_category": SOURCE_CATEGORY,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "train_sha256": train_hash, "validation_sha256": validation_hash,
        "source_hash_format": "sha256 of ordered canonical JSON description+sorted targets, newline separated",
        "training_device": str(training_device), "inference_device": "cpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__), "transformers_version": transformers_version,
    }
    parameters = {
        "encoder_trainable": sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad),
        "output_trainable": sum(parameter.numel() for parameter in model.output.parameters() if parameter.requires_grad),
        "total_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    epochs = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_metadata = None
    best_logits = None
    best_metrics = None
    completed_steps = 0
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_totals = torch.zeros(3, device=training_device)
        for batch, targets, batch_weights, soft_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = targets.to(training_device)
            batch_weights = batch_weights.to(training_device)
            logits = model(batch.to(training_device))
            hard_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
            soft_loss = torch.zeros_like(hard_loss)
            document_loss = hard_loss
            if soft_targets is not None:
                soft_loss = F.binary_cross_entropy_with_logits(
                    logits / temperature, soft_targets.to(training_device), reduction="none",
                ).mean(dim=1)
                document_loss = (1 - distillation_alpha) * hard_loss + distillation_alpha * temperature ** 2 * soft_loss
            loss = (document_loss * batch_weights).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            completed_steps += 1
            with torch.no_grad():
                loss_totals += torch.stack((
                    loss.detach(), (hard_loss * batch_weights).mean(),
                    (soft_loss * batch_weights).mean(),
                )) * len(targets)
        mean_loss, hard_bce, soft_bce = (loss_totals.cpu() / len(train_rows)).tolist()
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation_logits = _predict_documents(
            model, tokenizer, validation_documents, training_device, batch_size,
        )
        metrics = _validation_metrics(validation_logits, validation_targets)
        validation_bce = float(np.mean(
            np.logaddexp(0.0, validation_logits) - validation_targets * validation_logits,
        ))
        validation_seconds = time.perf_counter() - validation_started
        checkpoint_seconds = 0.0
        improved = metrics["micro_f1"] > best_f1
        if improved:
            best_f1 = metrics["micro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_logits = validation_logits
            best_metrics = metrics
            best_metadata = {
                **metadata_base,
                "training": {**training_config, "selected_epoch": epoch, "completed_steps": completed_steps},
                "selected_epoch": epoch, "best_validation_micro_f1": best_f1,
            }
            checkpoint_started = time.perf_counter()
            save_artifact(model, tokenizer, artifact, best_metadata)
            checkpoint_seconds = time.perf_counter() - checkpoint_started
        else:
            stale_epochs += 1
        epochs.append({
            "epoch": epoch, "train_loss": mean_loss,
            "train_weighted_hard_bce_loss": hard_bce,
            "train_weighted_distillation_bce_loss": soft_bce,
            "validation_bce_loss": validation_bce,
            "validation_micro_f1": metrics["micro_f1"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "selected": improved, "stale_epochs": stale_epochs,
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "completed_steps": completed_steps,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "seconds": time.perf_counter() - epoch_started,
        })
        logger.info(
            "encoder_f1 epoch %s/%s: loss %.6f, validation F1 %.6f (TP=%s FP=%s FN=%s)%s",
            epoch, max_epochs, mean_loss, metrics["micro_f1"], metrics["tp"], metrics["fp"],
            metrics["fn"], " [best]" if improved else "",
        )
        if epoch >= min_epochs and stale_epochs >= patience:
            break

    report = {
        "artifact_path": str(artifact), "metadata": best_metadata,
        "configuration": training_config, "parameters": parameters,
        "source_hashes": {"train": train_hash, "validation": validation_hash},
        "epochs": epochs, "best_epoch": best_epoch,
        "best_validation_micro_f1": best_f1, "best_validation_metrics": best_metrics,
        "best_validation_logits": best_logits.tolist(),
        "validation_logit_classes": list(CLASSES),
        "stop_reason": "early_stopping" if len(epochs) < max_epochs else "max_epochs",
        "completed_steps": completed_steps, "preparation_seconds": preparation_seconds,
        "total_seconds": time.perf_counter() - started,
        "evaluation": "Validation-only checkpoint selection; no control or diagnostic rows loaded",
    }
    (artifact / "train-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report
