"""Фиксированное обучение студентов: внешний OOF fold не выбирает checkpoint."""

from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from main import MICROCATEGORIES
from subcategory_extraction.semantic.semantic_encoder.model import MODEL_ID, MODEL_REVISION, collate_documents, select_device
from subcategory_extraction.semantic.semantic_evidence.model import EvidenceEncoder, save_artifact, tokenize_document, validate_architecture

CLASSES = tuple(MICROCATEGORIES)
logger = logging.getLogger(__name__)


def save_json(path: Path, value: dict) -> None:
    """Атомарно сохранить принадлежащий текущему опыту JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Хешировать артефакт без зависимости от даты файла."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_digest(rows: list[dict]) -> str:
    """Привязать обучение к точным описаниям и упорядоченным исходным меткам."""
    values = [{"description": row["description"], "targets": sorted(row["targets"])} for row in rows]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def release_memory() -> None:
    """Освободить промежуточные модели до загрузки следующей."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def labels(rows: list[dict]) -> np.ndarray:
    """Построить бинарные метки с неизменным порядком категорий."""
    return np.asarray([[category in row["targets"] for category in CLASSES] for row in rows], dtype=np.float32)


def fit(
    rows: list[dict], artifact: Path, architecture: dict, *, seed: int = 42,
    epochs: int = 5, batch_size: int = 8, learning_rate: float = 5e-5,
    device: str = "auto", sample_weights: np.ndarray | None = None,
    teacher_probabilities: np.ndarray | None = None, alpha: float = 0.3,
    model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION,
    provenance: dict | None = None,
) -> dict:
    """Обучить фиксированные пять эпох без чтения validation/control/test.

    Soft-target teacher обязан быть обучен исключительно на той же train-части.
    Это проверяется вызывающим OOF-планом и сохраняется в provenance. Тексты и
    метки внешнего fold не передаются в эту функцию ни для checkpoint, ни для loss.
    """
    if not rows or any(not isinstance(row.get("description"), str) or not isinstance(row.get("targets"), set)
                       or row["targets"] - (set(CLASSES) - {101}) for row in rows):
        raise ValueError("Expected nonempty train rows with additional-category target sets")
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid learning rate or distillation alpha")
    architecture = validate_architecture(architecture)
    weights = np.ones(len(rows), dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
    if weights.shape != (len(rows),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expected positive finite train-aligned sample weights")
    soft = None if teacher_probabilities is None else np.asarray(teacher_probabilities, dtype=np.float32)
    if soft is not None and (soft.shape != (len(rows), len(CLASSES)) or not np.isfinite(soft).all()
                             or (soft < 0).any() or (soft > 1).any()):
        raise ValueError("Expected train-aligned teacher probabilities [N,6]")
    signature = {
        "train_sha256": rows_digest(rows), "architecture": architecture, "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "model_id": model_id, "model_revision": model_revision,
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "teacher_sha256": None if soft is None else hashlib.sha256(soft.tobytes()).hexdigest(),
        "alpha": alpha if soft is not None else 0.0, "provenance": provenance or {},
    }
    artifact = Path(artifact)
    report_path = artifact / "train-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("signature") != signature:
            raise ValueError(f"Cannot reuse incompatible completed training: {artifact}")
        for relative, expected in report["artifact_sha256"].items():
            if digest(artifact / relative) != expected:
                raise ValueError(f"Completed training artifact changed: {artifact / relative}")
        return report
    if artifact.exists():
        raise FileExistsError(f"Incomplete artifact must not be silently reused: {artifact}")
    started = time.perf_counter()
    actual_device = select_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_id, revision=model_revision, trust_remote_code=False)
    model = EvidenceEncoder(encoder, len(CLASSES), mode=architecture["mode"]).to(actual_device)
    model.requires_grad_(True)
    documents = [tokenize_document(tokenizer, row["description"], architecture) for row in rows]
    targets = torch.from_numpy(labels(rows))
    row_weights = torch.from_numpy(weights)
    soft_targets = None if soft is None else torch.from_numpy(soft)

    def collate(indices: list[int]) -> tuple:
        return (collate_documents(tokenizer, [documents[index] for index in indices]),
                targets[indices], row_weights[indices],
                None if soft_targets is None else soft_targets[indices])

    loader = DataLoader(range(len(rows)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total = torch.zeros((), device=actual_device)
        for batch, hard, weight, teacher in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.to(actual_device))
            hard_loss = F.binary_cross_entropy_with_logits(logits, hard.to(actual_device), reduction="none").mean(1)
            loss = hard_loss
            if teacher is not None:
                soft_loss = F.binary_cross_entropy_with_logits(logits, teacher.to(actual_device), reduction="none").mean(1)
                loss = (1 - alpha) * hard_loss + alpha * soft_loss
            loss = (loss * weight.to(actual_device)).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += loss.detach() * len(hard)
        result = {"epoch": epoch, "loss": total.item() / len(rows), "seconds": time.perf_counter() - epoch_started}
        history.append(result)
        logger.info("%s epoch %d/%d loss %.6f (%.1fs)", artifact.name, epoch, epochs, result["loss"], result["seconds"])
    metadata = {
        "version": 1, "module": "semantic_evidence", "classes": list(CLASSES),
        "architecture": architecture, "trained_classes": [c for c in CLASSES if any(c in r["targets"] for r in rows)],
        "model_id": model_id, "model_revision": model_revision,
        "training": {**signature, "rows": len(rows), "device": str(actual_device),
                     "checkpoint_selection": "last fixed epoch; no external fold labels", "history": history},
        "provenance": provenance or {}, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    model.eval()
    save_artifact(model, tokenizer, artifact, metadata)
    report = {
        "signature": signature, "history": history, "rows": len(rows),
        "total_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "artifact_sha256": {name: digest(artifact / name) for name in
                            ("metadata.json", "head.safetensors", "encoder/model.safetensors")},
    }
    save_json(report_path, report)
    del model, encoder, tokenizer, optimizer, scheduler, loader
    release_memory()
    return report
