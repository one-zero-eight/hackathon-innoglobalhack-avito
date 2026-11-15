"""Дообучение реального RuBERT-tiny2 на всём отфильтрованном train без holdout."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Any

from safetensors.torch import save_file
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.semantic.semantic_encoder.model import (
    ARTIFACT_VERSION,
    DEFAULT_ARTIFACT,
    MODEL_ID,
    MODEL_REVISION,
    THRESHOLD,
    TOKENIZATION,
    TRAINING,
    SemanticEncoder,
    collate_documents,
    select_device,
    tokenize_document,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = ROOT / "out/hybrid-training.jsonl"
SOURCE_CATEGORY = 101


def load_training_rows(dataset: Path) -> tuple[list[dict[str, Any]], str]:
    """Прочитать готовый filtered train, не обращаясь к test и его меткам."""
    rows = []
    digest = hashlib.sha256()
    with dataset.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            inp = PipelineInput(
                description=record["description"], mcId=record.get("mcId", SOURCE_CATEGORY),
            )
            if inp.mc_id != SOURCE_CATEGORY:
                raise ValueError(f"Строка {line_number}: ожидается официальный train с mcId=101")
            targets = {target["mcId"] for target in record["found_microcategories"]}
            if (
                any(type(category) is not int or category not in MICROCATEGORIES for category in targets)
                or inp.mc_id in targets
            ):
                raise ValueError(f"Строка {line_number}: некорректные дополнительные категории")
            rows.append({"description": inp.description, "targets": targets})
    if not rows:
        raise ValueError("Обучающий dataset пуст")
    return rows, digest.hexdigest()


def train(dataset: Path, artifact: Path, requested_device: str) -> dict[str, Any]:
    """Обучить encoder и multilabel-голову ровно три эпохи и сохранить реальные веса."""
    started = time.perf_counter()
    seed = TRAINING["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(requested_device)
    rows, train_sha256 = load_training_rows(dataset)
    classes = list(MICROCATEGORIES)
    positives = {category: sum(category in row["targets"] for row in rows) for category in classes}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
    )
    model = SemanticEncoder(encoder, len(classes)).to(device)
    model.requires_grad_(True)
    training_data = [
        (
            tokenize_document(tokenizer, row["description"]),
            [float(category in row["targets"]) for category in classes],
        )
        for row in rows
    ]

    def collate(records: list[tuple]) -> tuple:
        """Объединить реальные окна документов и соответствующие multilabel-метки."""
        batch = collate_documents(tokenizer, [record[0] for record in records])
        targets = torch.tensor([record[1] for record in records], dtype=torch.float32)
        return batch, targets

    loader = DataLoader(
        training_data, batch_size=TRAINING["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    epoch_reports = []
    logger.info("semantic_encoder: %s документов, устройство %s, %s эпохи", len(rows), device, TRAINING["epochs"])
    for epoch in range(TRAINING["epochs"]):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = targets.to(device)
            logits = model(batch.to(device))
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Нечисловой train loss в эпохе {epoch + 1}")
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * targets.shape[0]
        mean_loss = total_loss.item() / len(rows)
        epoch_reports.append({
            "epoch": epoch + 1,
            "train_bce_loss": mean_loss,
            "seconds": time.perf_counter() - epoch_started,
        })
        logger.info("Эпоха %s/%s: train BCE %.6f", epoch + 1, TRAINING["epochs"], mean_loss)

    artifact = artifact.expanduser().resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.encoder.save_pretrained(artifact / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(artifact / "tokenizer")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()},
        str(artifact / "head.safetensors"),
    )
    trained_classes = [category for category in classes if positives[category] > 0]
    metadata = {
        "version": ARTIFACT_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "classes": classes,
        "threshold": THRESHOLD,
        "tokenization": TOKENIZATION,
        "training": TRAINING,
        "trained_classes": trained_classes,
        "unsupported_classes": [category for category in classes if category not in trained_classes],
        "positive_counts": {str(category): count for category, count in positives.items()},
        "source_category": SOURCE_CATEGORY,
        "train_path": str(dataset.expanduser().resolve()),
        "train_sha256": train_sha256,
        "train_rows": len(rows),
        "training_device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
    }
    report = {
        "artifact_path": str(artifact),
        "metadata": metadata,
        "epochs": epoch_reports,
        "encoder_trainable_parameters": sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad),
        "head_trainable_parameters": sum(parameter.numel() for parameter in model.head.parameters() if parameter.requires_grad),
        "total_seconds": time.perf_counter() - started,
        "evaluation": "No validation/holdout/test; all supplied training rows used in all epochs",
    }
    (artifact / "train-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (artifact / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    logger.info("Сохранён артефакт %s; категории без train positives: %s", artifact, metadata["unsupported_classes"])
    return report


def main() -> None:
    """Запустить фиксированный вариант обучения без подбора гиперпараметров."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    train(args.dataset, args.artifact, args.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
