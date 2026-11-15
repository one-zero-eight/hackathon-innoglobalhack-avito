"""CPU-инференс локального финального TinyBERT без загрузок из сети."""

import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import BertModel, BertTokenizerFast

from main import BasePipeline, PipelineInput
from subcategory_extraction.tinybert.model import (
    CLASSES,
    DROPOUT,
    MODEL_ID,
    MODEL_REVISION,
    THRESHOLD,
    TOKENIZATION,
    TinyBertClassifier,
    collate_documents,
    resolve_device,
    tokenize_document,
    validate_tokenizer,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = ROOT / "out/tinybert"
REQUIRED_FILES = (
    "metadata.json", "train-report.json", "head.safetensors",
    "encoder/config.json", "encoder/model.safetensors",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
)


def file_sha256(path: Path) -> str:
    """Вычислить SHA-256 файла без копирования всего файла в память."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Инвалидировать кэш после изменения любого файла артефакта."""
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"Нет полного локального артефакта TinyBERT: {path}; отсутствуют {missing}. "
            "Сначала запустите python -m subcategory_extraction.tinybert.train"
        )
    signature = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            stat = file.stat()
            signature.append((file.relative_to(path).as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _validate_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Отклонить незавершённые, несовместимые или повреждённые артефакты."""
    if (
        metadata.get("version") != 1
        or metadata.get("model_id") != MODEL_ID
        or metadata.get("revision") != MODEL_REVISION
        or metadata.get("classes") != list(CLASSES)
        or metadata.get("threshold") != THRESHOLD
        or metadata.get("tokenization") != TOKENIZATION
        or metadata.get("dropout") != DROPOUT
        or metadata.get("checkpoint") != "final_epoch_5"
    ):
        raise ValueError("Несовместимый формат или настройки артефакта TinyBERT")
    training = metadata.get("training", {})
    if training.get("epochs") != 5 or training.get("encoder_frozen") is not False:
        raise ValueError("Ожидается финальное дообучение всех слоёв TinyBERT за пять эпох")
    data = metadata.get("training_data", {})
    counts = metadata.get("positive_counts", {})
    rows = data.get("rows")
    if not isinstance(rows, int) or rows <= 0 or any(
        not isinstance(counts.get(str(category)), int)
        or not 0 <= counts[str(category)] <= rows for category in CLASSES
    ):
        raise ValueError("Некорректные размеры или счётчики обучающей выборки TinyBERT")
    trained = [category for category in CLASSES if 0 < counts[str(category)] < rows]
    if (
        metadata.get("trained_classes") != trained
        or metadata.get("unsupported_classes") != [category for category in CLASSES if category not in trained]
    ):
        raise ValueError("Несогласованная поддержка классов TinyBERT")
    files = metadata.get("artifact_files")
    if not isinstance(files, dict) or not set(REQUIRED_FILES).difference({"metadata.json"}).issubset(files):
        raise ValueError("Отсутствует манифест файлов TinyBERT")
    for relative, expected_hash in files.items():
        file = path / relative
        if not file.resolve().is_relative_to(path) or not file.is_file() or file_sha256(file) != expected_hash:
            raise ValueError(f"Нарушена целостность артефакта TinyBERT: {relative}")


@lru_cache(maxsize=2)
def _load_resources(
    artifact: str, device: str, signature: tuple[tuple[str, int, int], ...],
) -> tuple[TinyBertClassifier, BertTokenizerFast, dict[str, Any]]:
    """Разделять готовые eval-ресурсы только для той же версии файлов и устройства."""
    path = Path(artifact)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    _validate_metadata(metadata, path)
    tokenizer = BertTokenizerFast.from_pretrained(
        path / "tokenizer", local_files_only=True, trust_remote_code=False,
    )
    validate_tokenizer(tokenizer)
    encoder, loading_info = BertModel.from_pretrained(
        path / "encoder",
        add_pooling_layer=False,
        use_safetensors=True,
        weights_only=True,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Неполный или несовместимый энкодер TinyBERT: {loading_info}")
    model = TinyBertClassifier(encoder)
    head = load_file(str(path / "head.safetensors"), device="cpu")
    if not all(torch.isfinite(value).all().item() for value in head.values()):
        raise ValueError("Голова TinyBERT содержит нечисловые веса")
    model.classifier.load_state_dict(head, strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    return model, tokenizer, metadata


class SubcategoryExtractionPipeline(BasePipeline):
    """Выделить услуги настоящим дообученным TinyBERT из локального артефакта."""

    name = "tinybert"

    def __init__(self) -> None:
        path = Path(os.getenv("TINYBERT_ARTIFACT_PATH", str(DEFAULT_ARTIFACT))).expanduser()
        self.artifact_path = (path if path.is_absolute() else ROOT / path).resolve()
        self.device = str(resolve_device(os.getenv("TINYBERT_DEVICE", "cpu")))
        self._model, self._tokenizer, metadata = _load_resources(
            str(self.artifact_path), self.device, _artifact_signature(self.artifact_path),
        )
        self.model_metadata = deepcopy(metadata)
        self.categories = list(CLASSES)
        self.threshold = float(metadata["threshold"])
        self.trained_categories = list(metadata["trained_classes"])
        self.unsupported_categories = list(metadata["unsupported_classes"])

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Вернуть реальные sigmoid-оценки всех классов, не скрывая неподдержанные."""
        document = tokenize_document(self._tokenizer, inp.description)
        batch, counts = collate_documents(self._tokenizer, [document])
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        with torch.inference_mode():
            values = self._model(batch, counts).sigmoid()[0].cpu().tolist()
        return dict(zip(self.categories, values, strict=True))

    def run(self, inp: PipelineInput) -> list[int]:
        """Применить фиксированный порог и исключить только исходную категорию."""
        return [
            category for category, probability in self.probabilities(inp).items()
            if category != inp.mc_id and probability >= self.threshold
        ]
