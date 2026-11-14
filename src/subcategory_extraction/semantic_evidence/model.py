"""Обучаемое внимание к токенам, полные окна и локальные артефакты модели."""

from copy import deepcopy
from itertools import groupby
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from main import MICROCATEGORIES
from subcategory_extraction.semantic_encoder.model import (
    DocumentBatch,
    collate_documents,
    normalize_description,
)

ARTIFACT_VERSION = 1
MODULE_NAME = "semantic_evidence"
DEFAULT_ARCHITECTURE = {"mode": "evidence", "max_tokens": 256, "stride": 192}


def validate_architecture(architecture: dict[str, Any]) -> dict[str, Any]:
    """Проверить конфигурацию; stride означает шаг начала соседних окон."""
    if not isinstance(architecture, dict) or set(architecture) != set(DEFAULT_ARCHITECTURE):
        raise ValueError("Архитектура должна задавать mode, max_tokens и stride")
    mode = architecture["mode"]
    maximum = architecture["max_tokens"]
    stride = architecture["stride"]
    if (
        mode not in ("evidence", "document")
        or type(maximum) is not int or maximum not in (256, 512)
        or type(stride) is not int or stride not in (192, 510)
        or stride > maximum - 2
    ):
        raise ValueError("Неподдерживаемая архитектура semantic_evidence")
    return {"mode": mode, "max_tokens": maximum, "stride": stride}


def tokenize_document(
    tokenizer: PreTrainedTokenizerBase,
    description: str,
    architecture: dict[str, Any],
) -> list[dict[str, list[int]]]:
    """Покрыть весь текст окнами без пропуска хвоста и ограничения их количества."""
    configuration = validate_architecture(architecture)
    tokens = tokenizer.encode(
        normalize_description(description), add_special_tokens=False,
        truncation=False, verbose=False,
    )
    capacity = configuration["max_tokens"] - tokenizer.num_special_tokens_to_add(pair=False)
    step = configuration["stride"]
    if capacity <= 0 or step > capacity:
        raise ValueError("Служебные токены не позволяют покрыть текст выбранным шагом")
    last = max(0, len(tokens) - capacity)
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return [
        dict(tokenizer.prepare_for_model(
            tokens[start:start + capacity], add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=True,
            return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
        ))
        for start in positions
    ]


class _LabelwiseLinear(nn.Module):
    """Применить свою линейную проекцию к представлению каждой категории."""

    def __init__(self, hidden_size: int, class_count: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(class_count, hidden_size))
        self.bias = nn.Parameter(torch.empty(class_count))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, representations: Tensor) -> Tensor:
        """Вернуть по одному скаляру на категорию без матрицы лишних выходов."""
        return (representations * self.weight).sum(dim=-1) + self.bias


class EvidenceEncoder(nn.Module):
    """Дообучить encoder и объединить локальные свидетельства с контекстом."""

    def __init__(
        self, encoder: PreTrainedModel, class_count: int, mode: str = "evidence",
    ) -> None:
        super().__init__()
        if mode not in ("evidence", "document") or class_count <= 0:
            raise ValueError("Некорректный режим или число категорий")
        self.encoder = encoder
        self.encoder.requires_grad_(True)
        self.mode = mode
        self.class_count = class_count
        hidden_size = encoder.config.hidden_size
        if mode == "document":
            self.output = nn.ModuleDict({"classifier": nn.Linear(hidden_size, class_count)})
        else:
            self.output = nn.ModuleDict({
                "attention": nn.Linear(hidden_size, class_count, bias=False),
                "gate": _LabelwiseLinear(2 * hidden_size, class_count),
                "classifier": _LabelwiseLinear(hidden_size, class_count),
                "global": nn.Linear(hidden_size, class_count),
            })

    def forward(self, batch: DocumentBatch) -> Tensor:
        """Нормировать внимание по всем реальным токенам документа, а не по окнам."""
        hidden = self.encoder(**batch.inputs).last_hidden_state
        mask = batch.inputs["attention_mask"].bool()
        logits = []
        start = 0
        for _, group in groupby(batch.document_indices):
            end = start + sum(1 for _ in group)
            # Удаляем только padding: special tokens сохраняют валидность пустого текста.
            tokens = hidden[start:end][mask[start:end]]
            if tokens.shape[0] == 0:
                raise ValueError("Документ должен содержать хотя бы служебные токены")
            global_context = tokens.mean(dim=0)
            if self.mode == "document":
                logits.append(self.output["classifier"](global_context))
            else:
                attention = self.output["attention"](tokens).transpose(0, 1)
                evidence = attention.softmax(dim=-1) @ tokens
                context = global_context.unsqueeze(0).expand(self.class_count, -1)
                gate = self.output["gate"](torch.cat((evidence, context), dim=-1)).sigmoid()
                combined = gate.unsqueeze(-1) * evidence + (1 - gate.unsqueeze(-1)) * context
                # Независимая глобальная ветвь может подавить даже сильное локальное упоминание.
                logits.append(
                    self.output["classifier"](combined) + self.output["global"](global_context)
                )
            start = end
        return torch.stack(logits)


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Проверить версию, происхождение обучения и совместимость классов."""
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("version")) is not int
        or metadata.get("version") != ARTIFACT_VERSION
        or metadata.get("module") != MODULE_NAME
        or metadata.get("classes") != list(MICROCATEGORIES)
        or not isinstance(metadata.get("training"), dict)
        or not isinstance(metadata.get("provenance"), dict)
    ):
        raise ValueError("Несовместимые метаданные semantic_evidence или нет provenance")
    architecture = validate_architecture(metadata.get("architecture"))
    trained = metadata.get("trained_classes")
    if (
        not isinstance(trained, list)
        or any(type(category) is not int or category not in MICROCATEGORIES for category in trained)
        or len(trained) != len(set(trained))
    ):
        raise ValueError("Некорректный список обученных категорий semantic_evidence")
    return architecture


def save_artifact(
    model: EvidenceEncoder, tokenizer: PreTrainedTokenizerBase,
    path: Path | str, metadata: dict[str, Any],
) -> None:
    """Сохранить локальные safetensors, токенизатор и проверенное происхождение."""
    canonical = deepcopy(metadata)
    canonical.update({
        "version": ARTIFACT_VERSION, "module": MODULE_NAME, "classes": list(MICROCATEGORIES),
    })
    canonical["architecture"] = _validate_metadata(canonical)
    if canonical["architecture"]["mode"] != model.mode or model.class_count != len(MICROCATEGORIES):
        raise ValueError("Метаданные не соответствуют сохраняемой модели")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder", safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(path / "tokenizer")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.output.state_dict().items()},
        str(path / "head.safetensors"),
    )
    (path / "metadata.json").write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def load_artifact(
    path: Path | str, device: torch.device | str,
) -> tuple[EvidenceEncoder, PreTrainedTokenizerBase, dict[str, Any]]:
    """Загрузить только локальные encoder/head веса, не исполняя remote code."""
    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    architecture = _validate_metadata(metadata)
    if not (path / "encoder/model.safetensors").is_file():
        raise ValueError("Не найдены нешардированные safetensors encoder-модели")
    tokenizer = AutoTokenizer.from_pretrained(
        path / "tokenizer", local_files_only=True, trust_remote_code=False,
    )
    encoder = AutoModel.from_pretrained(
        path / "encoder", local_files_only=True, use_safetensors=True, trust_remote_code=False,
    )
    if architecture["max_tokens"] > encoder.config.max_position_embeddings:
        raise ValueError("Окна превышают позиционную ёмкость encoder-модели")
    model = EvidenceEncoder(encoder, len(MICROCATEGORIES), architecture["mode"])
    model.output.load_state_dict(load_file(str(path / "head.safetensors")), strict=True)
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("Артефакт содержит нечисловые или бесконечные веса")
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer, metadata


def predict_logits(
    model: EvidenceEncoder, tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]], architecture: dict[str, Any],
    device: torch.device | str, batch_size: int = 8,
) -> np.ndarray:
    """Получить document-level logits для тренера и OOF без изменения режима модели."""
    configuration = validate_architecture(architecture)
    if configuration["mode"] != model.mode or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Режим архитектуры или batch_size не соответствует модели")
    if not rows:
        return np.empty((0, model.class_count), dtype=np.float32)
    previous_training = model.training
    model.eval()
    predictions = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                documents = [
                    tokenize_document(tokenizer, row["description"], configuration)
                    for row in rows[start:start + batch_size]
                ]
                batch = collate_documents(tokenizer, documents).to(torch.device(device))
                predictions.append(model(batch).float().cpu().numpy())
    finally:
        model.train(previous_training)
    return np.concatenate(predictions, axis=0)
