"""Обучаемая агрегация окон и строго локальные артефакты плана улучшения F1."""

from itertools import groupby
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from main import MICROCATEGORIES
from subcategory_extraction.semantic.semantic_encoder.model import (
    DocumentBatch,
    collate_documents,
    normalize_description,
    select_device,
)

ARTIFACT_VERSION = 1
DEFAULT_ARCHITECTURE = {
    "pooling": "mean", "max_tokens": 512, "max_windows": 3, "strategy": "legacy",
}


def validate_architecture(architecture: dict[str, Any]) -> dict[str, Any]:
    """Проверить ограниченные настройки; coverage по умолчанию использует шесть окон."""
    if not isinstance(architecture, dict) or set(architecture) - set(DEFAULT_ARCHITECTURE):
        raise ValueError("Unknown architecture settings")
    settings = {**DEFAULT_ARCHITECTURE, **architecture}
    if settings["strategy"] not in ("legacy", "coverage"):
        raise ValueError("strategy must be legacy or coverage")
    if settings["strategy"] == "coverage" and "max_windows" not in architecture:
        settings["max_windows"] = 6
    if settings["pooling"] not in ("mean", "attention"):
        raise ValueError("pooling must be mean or attention")
    if type(settings["max_tokens"]) is not int or not 4 <= settings["max_tokens"] <= 512:
        raise ValueError("max_tokens must be an integer in [4, 512]")
    if type(settings["max_windows"]) is not int or not 1 <= settings["max_windows"] <= 32:
        raise ValueError("max_windows must be an integer in [1, 32]")
    if settings["strategy"] == "legacy" and settings["max_windows"] > 3:
        raise ValueError("legacy supports at most first/middle/last windows")
    if settings["strategy"] == "coverage" and settings["max_windows"] < 2:
        raise ValueError("coverage requires at least two windows to retain both endpoints")
    return settings


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase, description: str, architecture: dict[str, Any],
) -> list[dict[str, list[int]]]:
    """Сохранить исходное содержание кроме общей NFKC/нормализации пробелов.

    Legacy повторяет прежние окна начала/середины/конца по512 токенов.
    Coverage разбивает текст на смежные непересекающиеся фрагменты: при
    достаточном бюджете сохраняются все токены. Сверх бюджета равномерно
    выбираются фрагменты, включая первый/последний; между ними возможны
    пропуски. Служебные токены входят в max_tokens и в счётчики pooling.
    """
    settings = validate_architecture(architecture)
    capacity = settings["max_tokens"] - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("Special tokens do not fit in max_tokens")
    tokens = tokenizer.encode(
        normalize_description(description), add_special_tokens=False,
        truncation=False, verbose=False,
    )
    budget = settings["max_windows"]
    if settings["strategy"] == "legacy":
        last = max(0, len(tokens) - capacity)
        positions = [0] if budget == 1 else sorted({
            index * last // (budget - 1) for index in range(budget)
        })
    else:
        chunk_count = max(1, (len(tokens) + capacity - 1) // capacity)
        if chunk_count <= budget:
            chunks = range(chunk_count)
        else:
            chunks = [index * (chunk_count - 1) // (budget - 1) for index in range(budget)]
        positions = [index * capacity for index in chunks]
    return [
        dict(tokenizer.prepare_for_model(
            tokens[start:start + capacity], add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=True,
            return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
        ))
        for start in positions
    ]


class WindowEncoder(nn.Module):
    """Взвешенное по токенам среднее либо attention окон отдельно для каждой метки."""

    def __init__(self, encoder: PreTrainedModel, class_count: int, pooling: str) -> None:
        super().__init__()
        if pooling not in ("mean", "attention"):
            raise ValueError("pooling must be mean or attention")
        if type(class_count) is not int or class_count <= 0:
            raise ValueError("class_count must be a positive integer")
        self.encoder = encoder
        self.pooling = pooling
        self.output = nn.ModuleDict({
            "classifier": nn.Linear(encoder.config.hidden_size, class_count),
        })
        if pooling == "attention":
            attention = nn.Linear(encoder.config.hidden_size, class_count)
            nn.init.zeros_(attention.weight)
            nn.init.zeros_(attention.bias)
            self.output["attention"] = attention

    def forward(self, batch: DocumentBatch) -> Tensor:
        """Объединить реальные окна без фиктивных окон и отрыва градиентов."""
        hidden = self.encoder(**batch.inputs).last_hidden_state
        mask = batch.inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        token_sums = (hidden * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp_min(1)
        if self.pooling == "attention":
            window_means = token_sums / token_counts
            window_logits = self.output["classifier"](window_means)
            attention_logits = self.output["attention"](window_means) + token_counts.log()
        documents = []
        start = 0
        for _, group in groupby(batch.document_indices):
            end = start + sum(1 for _ in group)
            if self.pooling == "mean":
                documents.append(
                    token_sums[start:end].sum(dim=0) / token_counts[start:end].sum(),
                )
            else:
                weights = attention_logits[start:end].softmax(dim=0)
                documents.append((weights * window_logits[start:end]).sum(dim=0))
            start = end
        result = torch.stack(documents)
        return self.output["classifier"](result) if self.pooling == "mean" else result


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Отвергнуть несовместимую схему, порядок меток или архитектуру."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    if type(metadata.get("version")) is not int or metadata["version"] != ARTIFACT_VERSION:
        raise ValueError("Unsupported encoder artifact version")
    for key in ("model_id", "model_revision"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata requires a nonempty {key}")
    classes = metadata.get("classes")
    if (
        not isinstance(classes, list)
        or any(type(category) is not int for category in classes)
        or classes != list(MICROCATEGORIES)
    ):
        raise ValueError("Artifact classes must match all microcategories in order")
    architecture = metadata.get("architecture")
    settings = validate_architecture(architecture)
    if set(architecture) != set(DEFAULT_ARCHITECTURE) or architecture != settings:
        raise ValueError("Artifact architecture must contain all canonical settings")
    if not isinstance(metadata.get("training"), dict):
        raise ValueError("Artifact requires training provenance")
    if "trained_classes" in metadata:
        trained = metadata["trained_classes"]
        if (
            not isinstance(trained, list)
            or any(type(category) is not int or category not in MICROCATEGORIES for category in trained)
            or len(trained) != len(set(trained))
            or 101 in trained
        ):
            raise ValueError("Invalid trained_classes; category 101 has no positive training support")
    return settings


def _validate_model(model: WindowEncoder, settings: dict[str, Any]) -> None:
    if model.pooling != settings["pooling"]:
        raise ValueError("Model pooling disagrees with architecture metadata")
    if model.output["classifier"].out_features != len(MICROCATEGORIES):
        raise ValueError("Model classifier must include all microcategories")
    positions = getattr(model.encoder.config, "max_position_embeddings", None)
    if positions is not None and settings["max_tokens"] > positions:
        raise ValueError("max_tokens exceeds encoder positional capacity")


def save_artifact(
    model: WindowEncoder, tokenizer: PreTrainedTokenizerBase,
    artifact: Path, metadata: dict[str, Any],
) -> None:
    """Сохранить новый лучший checkpoint в каталоге текущего опыта."""
    settings = _validate_metadata(metadata)
    _validate_model(model, settings)
    artifact = Path(artifact)
    artifact.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(artifact / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(artifact / "tokenizer")
    save_file({
        key: tensor.detach().cpu().contiguous()
        for key, tensor in model.output.state_dict().items()
    }, str(artifact / "head.safetensors"))
    (artifact / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_artifact(
    artifact: Path, device: torch.device,
) -> tuple[WindowEncoder, PreTrainedTokenizerBase, dict[str, Any]]:
    """Загрузить только локальные safetensors; calibration проверяет production pipeline."""
    artifact = Path(artifact).expanduser().resolve()
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    settings = _validate_metadata(metadata)
    tokenizer = AutoTokenizer.from_pretrained(
        artifact / "tokenizer", local_files_only=True, trust_remote_code=False,
    )
    encoder = AutoModel.from_pretrained(
        artifact / "encoder", local_files_only=True, use_safetensors=True,
        trust_remote_code=False,
    )
    model = WindowEncoder(encoder, len(metadata["classes"]), settings["pooling"])
    _validate_model(model, settings)
    model.output.load_state_dict(load_file(str(artifact / "head.safetensors")), strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, metadata
