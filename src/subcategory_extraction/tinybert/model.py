"""Исходный TinyBERT и единая обработка всех окон документа."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from transformers import BertConfig, BertModel, BertTokenizerFast

from main import MICROCATEGORIES

MODEL_ID = "huawei-noah/TinyBERT_General_4L_312D"
MODEL_REVISION = "34707a33cd59a94ecde241ac209bf35103691b43"
CLASSES = tuple(MICROCATEGORIES)
THRESHOLD = 0.5
DROPOUT = 0.1
TOKENIZATION = {
    "max_length": 512,
    "stride": 64,
    "truncation": True,
    "return_overflowing_tokens": True,
    "padding": False,
    "add_special_tokens": True,
}
ENCODER_DIMENSIONS = {
    "num_hidden_layers": 4,
    "hidden_size": 312,
    "intermediate_size": 1200,
    "num_attention_heads": 12,
    "vocab_size": 30522,
    "max_position_embeddings": 512,
}
Window = dict[str, list[int]]
DocumentWindows = list[Window]


def validate_encoder_config(config: BertConfig) -> None:
    """Отклонить другую архитектуру вместо исходного TinyBERT."""
    if config.model_type != "bert" or any(
        getattr(config, name) != value for name, value in ENCODER_DIMENSIONS.items()
    ):
        raise ValueError("Архитектура не соответствует Huawei TinyBERT General 4L/312D")


def validate_tokenizer(tokenizer: BertTokenizerFast) -> None:
    """Проверить основные свойства штатного WordPiece-токенизатора."""
    if (
        not tokenizer.is_fast
        or not tokenizer.do_lower_case
        or len(tokenizer) != ENCODER_DIMENSIONS["vocab_size"]
        or tokenizer.padding_side != "right"
        or tokenizer.truncation_side != "right"
        or tokenizer.pad_token_id != 0
        or tokenizer.unk_token_id != 100
        or tokenizer.cls_token_id != 101
        or tokenizer.sep_token_id != 102
    ):
        raise ValueError("Артефакт содержит несовместимый токенизатор TinyBERT")


def resolve_device(requested: str) -> torch.device:
    """Выбрать CPU/MPS; явный недоступный MPS не заменять молча CPU."""
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Устройство TinyBERT должно быть auto, cpu или mps")
    mps_available = torch.backends.mps.is_available()
    if requested == "mps" and not mps_available:
        raise ValueError("Запрошен MPS, но он недоступен в этой установке PyTorch")
    return torch.device("mps" if requested == "mps" or (requested == "auto" and mps_available) else "cpu")


def tokenize_document(tokenizer: BertTokenizerFast, description: str) -> DocumentWindows:
    """Сохранить все штатные окна описания, включая перекрытия и спецтокены."""
    encoded = tokenizer(description, **TOKENIZATION)
    return [
        {name: encoded[name][index] for name in tokenizer.model_input_names}
        for index in range(len(encoded["input_ids"]))
    ]


def collate_documents(
    tokenizer: BertTokenizerFast, documents: Sequence[DocumentWindows],
) -> tuple[dict[str, Tensor], tuple[int, ...]]:
    """Дополнить только реальные окна и сохранить границы документов."""
    counts = tuple(len(document) for document in documents)
    if not counts or any(count == 0 for count in counts):
        raise ValueError("Каждый документ должен содержать хотя бы одно окно")
    windows = [window for document in documents for window in document]
    batch = tokenizer.pad(windows, padding=True, return_tensors="pt")
    return dict(batch), counts


class TinyBertClassifier(nn.Module):
    """Дообучаемый энкодер и многометочная голова без неиспользуемого pooler."""

    def __init__(self, encoder: BertModel) -> None:
        super().__init__()
        validate_encoder_config(encoder.config)
        if encoder.pooler is not None:
            raise ValueError("TinyBERT должен загружаться без pooler")
        self.encoder = encoder
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(CLASSES))

    def forward(self, batch: dict[str, Tensor], window_counts: tuple[int, ...]) -> Tensor:
        """Взвесить окна числом непустых токенов, сохранив градиенты без scatter."""
        if sum(window_counts) != batch["input_ids"].shape[0]:
            raise ValueError("Границы документов не соответствуют числу окон")
        states = self.encoder(**batch, return_dict=True).last_hidden_state
        mask = batch["attention_mask"].to(dtype=states.dtype)
        window_sums = torch.bmm(mask.unsqueeze(1), states).squeeze(1)
        window_lengths = mask.sum(dim=1)
        document_sums = torch.stack([
            windows.sum(dim=0) for windows in window_sums.split(window_counts, dim=0)
        ])
        document_lengths = torch.stack([
            lengths.sum() for lengths in window_lengths.split(window_counts, dim=0)
        ])
        pooled = document_sums / document_lengths.unsqueeze(1)
        return self.classifier(self.dropout(pooled))


def load_original_model() -> tuple[TinyBertClassifier, BertTokenizerFast, dict[str, Any]]:
    """Загрузить закреплённый оригинал безопасным torch weights_only-путём."""
    encoder, loading_info = BertModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        add_pooling_layer=False,
        use_safetensors=False,
        weights_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise ValueError(f"Не удалось полностью загрузить энкодер TinyBERT: {loading_info}")
    unexpected = loading_info.get("unexpected_keys", [])
    if any(not key.startswith(("cls.", "fit_denses.", "pooler.", "bert.pooler.")) for key in unexpected):
        raise ValueError(f"Неожиданные веса исходного TinyBERT: {unexpected}")
    tokenizer = BertTokenizerFast.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        do_lower_case=True,
        trust_remote_code=False,
    )
    validate_tokenizer(tokenizer)
    model = TinyBertClassifier(encoder)
    model.requires_grad_(True)
    return model, tokenizer, loading_info
