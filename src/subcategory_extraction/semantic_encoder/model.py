"""Общая архитектура, окна токенов и локальная загрузка encoder-модели."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
import json
from pathlib import Path
from typing import Any
import unicodedata

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from main import MICROCATEGORIES

MODEL_ID = "cointegrated/rubert-tiny2"
MODEL_REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "out/semantic-encoder"
THRESHOLD = 0.5
TOKENIZATION = {
    "normalization": "NFKC; collapse whitespace; preserve case and ё",
    "max_tokens": 512,
    "max_windows": 3,
    "window_selection": "first/middle/last; deduplicate positions",
    "pooling": "mean over all attention-masked tokens of real windows, including special tokens",
    "padding": "longest real window in document batch",
}
TRAINING = {
    "epochs": 3,
    "seed": 42,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "batch_size": 16,
    "optimizer": "AdamW",
    "loss": "BCEWithLogitsLoss; unweighted mean over documents and classes",
    "fine_tune_encoder": True,
    "validation": None,
    "threshold_selection": None,
}


def normalize_description(text: str) -> str:
    """Применить одну нормализацию при обучении и инференсе."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def select_device(requested: str) -> torch.device:
    """Выбрать CPU/MPS, не подменяя явно запрошенный недоступный backend."""
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("SEMANTIC_ENCODER_DEVICE/--device: ожидается auto, cpu или mps")
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError("MPS запрошен, но недоступен в текущем PyTorch/на устройстве")
    return torch.device("mps" if requested == "mps" or requested == "auto" and available else "cpu")


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase, description: str,
) -> list[dict[str, list[int]]]:
    """Выбрать реальные окна начала, середины и конца без обрезания до префикса."""
    tokens = tokenizer.encode(
        normalize_description(description), add_special_tokens=False, truncation=False,
        verbose=False,
    )
    capacity = TOKENIZATION["max_tokens"] - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("Служебные токены не помещаются в окно encoder-модели")
    last = max(0, len(tokens) - capacity)
    positions = sorted({0, last // 2, last})
    return [
        dict(tokenizer.prepare_for_model(
            tokens[start:start + capacity], add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=True,
            return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
        ))
        for start in positions
    ]


@dataclass(frozen=True)
class DocumentBatch:
    """Плоский batch реальных окон и их индексов исходных документов."""

    inputs: dict[str, Tensor]
    document_indices: tuple[int, ...]

    def to(self, device: torch.device) -> "DocumentBatch":
        """Перенести только входные тензоры на вычислительное устройство."""
        return DocumentBatch(
            {key: value.to(device) for key, value in self.inputs.items()},
            self.document_indices,
        )


def collate_documents(
    tokenizer: PreTrainedTokenizerBase, documents: list[list[dict[str, list[int]]]],
) -> DocumentBatch:
    """Дополнить только реальные окна до общей длины, без фиктивных документов."""
    if not documents or any(not windows for windows in documents):
        raise ValueError("Batch должен содержать документы с хотя бы одним окном")
    windows = [window for document in documents for window in document]
    indices = tuple(index for index, document in enumerate(documents) for _ in document)
    padded = tokenizer.pad(windows, padding=True, return_tensors="pt")
    return DocumentBatch(dict(padded), indices)


class SemanticEncoder(nn.Module):
    """Дообучаемый transformer, среднее по документу и линейная multilabel-голова."""

    def __init__(self, encoder: PreTrainedModel, class_count: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, class_count)

    def forward(self, batch: DocumentBatch) -> Tensor:
        """Сохранить градиенты всех реальных окон при объединении документа."""
        hidden = self.encoder(**batch.inputs).last_hidden_state
        mask = batch.inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        token_sums = (hidden * mask).sum(dim=1)
        token_counts = mask.sum(dim=1)
        pooled = []
        start = 0
        for _, group in groupby(batch.document_indices):
            end = start + sum(1 for _ in group)
            pooled.append(token_sums[start:end].sum(dim=0) / token_counts[start:end].sum().clamp_min(1))
            start = end
        return self.head(torch.stack(pooled))


@dataclass(frozen=True)
class EncoderResource:
    """Один неизменяемый контейнер локальных inference-ресурсов."""

    tokenizer: PreTrainedTokenizerBase
    model: SemanticEncoder
    metadata: dict[str, Any]
    device: torch.device


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Отвергнуть несовместимую архитектуру и неподдерживаемые настройки."""
    if (
        metadata.get("version") != ARTIFACT_VERSION
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("classes") != list(MICROCATEGORIES)
        or metadata.get("threshold") != THRESHOLD
        or metadata.get("tokenization") != TOKENIZATION
        or metadata.get("training") != TRAINING
    ):
        raise ValueError("Несовместимый артефакт semantic_encoder")
    trained = metadata.get("trained_classes")
    if (
        not isinstance(trained, list)
        or any(type(category) is not int or category not in MICROCATEGORIES for category in trained)
        or len(trained) != len(set(trained))
    ):
        raise ValueError("Некорректный список обученных категорий semantic_encoder")


@lru_cache(maxsize=2)
def _load_resource(
    artifact: str, signature: tuple[tuple[str, int, int], ...], device_name: str,
) -> EncoderResource:
    """Загрузить локальные веса единожды на неизменённую версию артефакта."""
    path = Path(artifact)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(
        path / "encoder", local_files_only=True, use_safetensors=True,
    )
    model = SemanticEncoder(encoder, len(metadata["classes"]))
    model.head.load_state_dict(load_file(str(path / "head.safetensors")), strict=True)
    device = select_device(device_name)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return EncoderResource(tokenizer, model, metadata, device)


def load_resource(artifact: Path, requested_device: str) -> EncoderResource:
    """Кэшировать по абсолютному пути, версии конфигурации/весов и устройству."""
    path = artifact.expanduser().resolve()
    required = (
        "metadata.json", "head.safetensors", "encoder/config.json", "encoder/model.safetensors",
        "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
    )
    signature = []
    for relative in required:
        file = path / relative
        if not file.is_file():
            raise ValueError(
                f"Нет локального артефакта semantic_encoder: {file}. Обучение: "
                "PYTHONPATH=src uv run python -m "
                "subcategory_extraction.semantic_encoder.train "
                "--dataset out/hybrid-training.jsonl --device auto"
            )
        stat = file.stat()
        signature.append((relative, stat.st_mtime_ns, stat.st_size))
    device = select_device(requested_device)
    return _load_resource(str(path), tuple(signature), str(device))
