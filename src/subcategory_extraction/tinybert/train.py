"""Пять фиксированных эпох supervised fine-tuning настоящего Huawei TinyBERT."""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from subcategory_extraction.tinybert.model import (
    CLASSES,
    DROPOUT,
    MODEL_ID,
    MODEL_REVISION,
    THRESHOLD,
    TOKENIZATION,
    DocumentWindows,
    collate_documents,
    load_original_model,
    resolve_device,
    tokenize_document,
)
from subcategory_extraction.tinybert.pipeline import DEFAULT_ARTIFACT, ROOT, file_sha256

LOGGER = logging.getLogger(__name__)
TRAINING_SETTINGS = {
    "epochs": 5,
    "seed": 42,
    "optimizer": "AdamW",
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "batch_size_documents": 8,
    "drop_last": False,
    "loss": "BCEWithLogitsLoss",
    "gradient_clip_norm": 1.0,
    "encoder_frozen": False,
    "shuffle": True,
    "num_workers": 0,
    "checkpoint_selection": "fixed_final_epoch_only",
    "validation": None,
    "early_stopping": False,
    "parameter_search": False,
    "teacher_distillation": False,
}


def read_training_data(path: Path) -> tuple[list[str], list[list[float]], dict[str, Any]]:
    """Прочитать только descriptions и reviewed-метки из подготовленного JSONL."""
    descriptions: list[str] = []
    labels: list[list[float]] = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("description"), str):
                raise ValueError(f"Нет текстового description: {path}:{line_number}")
            found = row.get("found_microcategories")
            if not isinstance(found, list):
                raise ValueError(f"Нет found_microcategories: {path}:{line_number}")
            categories = set()
            for category in found:
                if (
                    not isinstance(category, dict)
                    or type(category.get("mcId")) is not int
                    or category["mcId"] not in CLASSES
                ):
                    raise ValueError(f"Некорректная метка: {path}:{line_number}: {category}")
                categories.add(category["mcId"])
            descriptions.append(row["description"])
            labels.append([float(category in categories) for category in CLASSES])
    if not descriptions:
        raise ValueError(f"Обучающая выборка пуста: {path}")
    provenance_path = path.with_name(f"{path.stem}-provenance.json")
    provenance = None
    provenance_hash = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_hash = file_sha256(provenance_path)
    return descriptions, labels, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": len(descriptions),
        "provenance": provenance,
        "provenance_sha256": provenance_hash,
        "input_field": "description",
        "title_used": False,
        "label_field": "found_microcategories[].mcId",
        "label_provenance": "human-reviewed derived labels, not organizer ground truth",
    }


class TrainingCollator:
    """Собрать батч целых документов без фиктивных окон."""

    def __init__(self, tokenizer: BertTokenizerFast) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self, examples: list[tuple[DocumentWindows, list[float]]],
    ) -> tuple[dict[str, Tensor], tuple[int, ...], Tensor]:
        batch, counts = collate_documents(self.tokenizer, [windows for windows, _ in examples])
        labels = torch.tensor([labels for _, labels in examples], dtype=torch.float32)
        return batch, counts, labels


def fingerprint(module: nn.Module) -> str:
    """Хешировать фактические имена, формы и значения всех тензоров модуля."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}\n".encode("utf-8"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    """Включить завершённые MPS-вычисления в реальные замеры времени."""
    if device.type == "mps":
        torch.mps.synchronize()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Записать человекочитаемый JSON без NaN/Infinity."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(dataset: Path, artifact: Path, requested_device: str) -> dict[str, Any]:
    """Дообучить все параметры и атомарно опубликовать только финальный артефакт."""
    started = perf_counter()
    device = resolve_device(requested_device)
    if artifact.exists():
        raise FileExistsError(f"Артефакт уже существует: {artifact}; укажите новый --artifact")
    random.seed(TRAINING_SETTINGS["seed"])
    torch.manual_seed(TRAINING_SETTINGS["seed"])
    if device.type == "mps":
        torch.mps.manual_seed(TRAINING_SETTINGS["seed"])
    descriptions, labels, training_data = read_training_data(dataset)
    model, tokenizer, loading_info = load_original_model()
    before = {
        "model": fingerprint(model),
        "encoder": fingerprint(model.encoder),
        "head": fingerprint(model.classifier),
    }
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "encoder": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "head": sum(parameter.numel() for parameter in model.classifier.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    encoded = [tokenize_document(tokenizer, description) for description in descriptions]
    tokenization_report = {
        "documents": len(encoded),
        "windows": sum(len(document) for document in encoded),
        "min_windows_per_document": min(map(len, encoded)),
        "max_windows_per_document": max(map(len, encoded)),
        "tokens_including_specials_and_overlap": sum(
            len(window["input_ids"]) for document in encoded for window in document
        ),
        "unknown_tokens_including_overlap": sum(
            window["input_ids"].count(tokenizer.unk_token_id)
            for document in encoded for window in document
        ),
    }
    del descriptions
    positive_counts = {
        str(category): int(sum(row[index] for row in labels))
        for index, category in enumerate(CLASSES)
    }
    trained_classes = [category for category in CLASSES if 0 < positive_counts[str(category)] < len(labels)]
    unsupported_classes = [category for category in CLASSES if category not in trained_classes]
    generator = torch.Generator().manual_seed(TRAINING_SETTINGS["seed"])
    loader = DataLoader(
        list(zip(encoded, labels, strict=True)),
        batch_size=TRAINING_SETTINGS["batch_size_documents"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
        collate_fn=TrainingCollator(tokenizer),
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_SETTINGS["learning_rate"],
        weight_decay=TRAINING_SETTINGS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    epochs = []
    optimization_started = perf_counter()
    total_steps = 0
    for epoch in range(1, TRAINING_SETTINGS["epochs"] + 1):
        synchronize(device)
        epoch_started = perf_counter()
        loss_sum = 0.0
        seen_documents = 0
        steps = 0
        for batch, window_counts, targets in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch, window_counts)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Нечисловой loss TinyBERT на эпохе {epoch}, шаге {steps + 1}")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING_SETTINGS["gradient_clip_norm"], error_if_nonfinite=True,
            )
            optimizer.step()
            documents = targets.shape[0]
            loss_sum += loss.detach().item() * documents
            seen_documents += documents
            steps += 1
        synchronize(device)
        epoch_report = {
            "epoch": epoch,
            "loss": loss_sum / seen_documents,
            "seconds": perf_counter() - epoch_started,
            "documents": seen_documents,
            "optimizer_steps": steps,
        }
        epochs.append(epoch_report)
        total_steps += steps
        LOGGER.info("Эпоха %d/5: loss=%.6f, %.2f с, %d документов", epoch, epoch_report["loss"], epoch_report["seconds"], seen_documents)
    optimization_seconds = perf_counter() - optimization_started
    model.to("cpu").eval()
    after = {
        "model": fingerprint(model),
        "encoder": fingerprint(model.encoder),
        "head": fingerprint(model.classifier),
    }
    update_proof = {
        component: {"before_sha256": before[component], "after_sha256": after[component], "changed": before[component] != after[component]}
        for component in before
    }
    if not all(proof["changed"] for proof in update_proof.values()):
        raise RuntimeError(f"Не подтверждено обновление энкодера и головы: {update_proof}")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }
    metadata = {
        "version": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "classes": list(CLASSES),
        "threshold": THRESHOLD,
        "dropout": DROPOUT,
        "tokenization": TOKENIZATION,
        "tokenizer": {"class": "BertTokenizerFast", "do_lower_case": True, "source": MODEL_ID, "revision": MODEL_REVISION},
        "pooling": {
            "tokens": "masked_mean_including_special_tokens_excluding_padding",
            "documents": "length_weighted_mean_of_all_native_windows",
            "overlap": "counted_twice",
            "window_limit": None,
        },
        "checkpoint": "final_epoch_5",
        "positive_counts": positive_counts,
        "trained_classes": trained_classes,
        "unsupported_classes": unsupported_classes,
        "unsupported_reasons": {
            str(category): "no_positive_examples" if positive_counts[str(category)] == 0 else "no_negative_examples"
            for category in unsupported_classes
        },
        "training": TRAINING_SETTINGS,
        "training_data": training_data,
        "device": str(device),
        "requested_device": requested_device,
        "versions": versions,
        "original_loading_info": loading_info,
        "parameter_counts": parameter_counts,
        "encoder_update_proof": update_proof["encoder"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Reviewed labels are human-derived, not organizer ground truth.",
            "Native original WordPiece tokenizer is retained; Cyrillic may be heavily fragmented.",
            "Classes with no positive or no negative examples are unsupported; their sigmoid scores are not replaced or hidden.",
            "No new teacher distillation, validation, early stopping, or test-label checkpoint selection.",
            "Overlapping tokens and special tokens contribute once per window to document pooling.",
        ],
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{artifact.name}-", dir=artifact.parent))
    try:
        model.encoder.save_pretrained(staging / "encoder", safe_serialization=True)
        tokenizer.save_pretrained(staging / "tokenizer")
        save_file(model.classifier.state_dict(), str(staging / "head.safetensors"))
        report = {
            "version": 1,
            "status": "completed",
            "epochs": epochs,
            "optimizer_steps": total_steps,
            "optimization_seconds": optimization_seconds,
            "wall_seconds": perf_counter() - started,
            "wall_time_scope": "data/model loading through final weight serialization; excludes JSON finalization and atomic publication",
            "parameter_counts": parameter_counts,
            "parameter_update_proof": update_proof,
            "training_data": training_data,
            "tokenization": tokenization_report,
            "device": str(device),
            "versions": versions,
            "original_loading_info": loading_info,
            "checkpoint": "final_epoch_5",
        }
        write_json(staging / "train-report.json", report)
        metadata["artifact_files"] = {
            file.relative_to(staging).as_posix(): file_sha256(file)
            for file in sorted(staging.rglob("*")) if file.is_file()
        }
        write_json(staging / "metadata.json", metadata)
        os.replace(staging, artifact)
    except BaseException:
        shutil.rmtree(staging)
        raise
    LOGGER.info("Сохранён финальный TinyBERT: %s; wall %.2f с", artifact, report["wall_seconds"])
    return report


def main() -> None:
    """Запустить фиксированное обучение без доступа к меткам итогового теста."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "out/tinybert-training.jsonl")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args.dataset.expanduser().resolve(), args.artifact.expanduser().resolve(), args.device)


if __name__ == "__main__":
    main()
