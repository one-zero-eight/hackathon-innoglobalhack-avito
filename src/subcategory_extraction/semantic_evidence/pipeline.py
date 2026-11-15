"""CPU-only runtime выбранной композиции локальных evidence/document-моделей."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch

from main import BasePipeline, MICROCATEGORIES, PipelineInput
from subcategory_extraction.semantic_evidence.model import (
    ARTIFACT_VERSION,
    collate_documents,
    load_artifact,
    tokenize_document,
)

DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts"
_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _inside_bundle(path: Path, root: Path) -> Path:
    """Не позволить относительному пути или symlink выйти за пределы bundle."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Путь артефакта выходит за пределы bundle: {path}")
    return resolved


def _sha256(path: Path) -> str:
    """Потоково вычислить отпечаток настоящего файла весов или метаданных."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    """Отличить конечное JSON-число от bool, NaN и бесконечности."""
    return type(value) in (int, float) and math.isfinite(value)


def _load_selection(root: Path) -> tuple[dict[str, Any], list[tuple[Path, float]]]:
    """Проверить схему композиции и хеши всех выбранных локальных весов."""
    manifest = _inside_bundle(root / "selection.json", root)
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(selection, dict)
        or type(selection.get("version")) is not int
        or selection.get("version") != ARTIFACT_VERSION
        or selection.get("classes") != list(MICROCATEGORIES)
    ):
        raise ValueError("Несовместимая схема selection.json semantic_evidence")
    thresholds = selection.get("thresholds")
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != {str(category) for category in MICROCATEGORIES}
        or any(not _number(value) or not 0 <= value <= 1 for value in thresholds.values())
    ):
        raise ValueError("selection.json должен задавать конечный порог [0,1] для каждой категории")
    components = selection.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("selection.json должен содержать хотя бы один компонент")
    validated = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("Некорректный компонент selection.json")
        name = component.get("path")
        weight = component.get("weight")
        if (
            not isinstance(name, str) or not name
            or Path(name).is_absolute() or ".." in Path(name).parts
            or not _number(weight) or weight <= 0
        ):
            raise ValueError("Некорректный относительный путь или вес компонента")
        path = _inside_bundle(root / name, root)
        if not path.is_dir():
            raise ValueError(f"Нет каталога компонента semantic_evidence: {path}")
        # Проверяем также symlink конфигураций и tokenizer, загружаемых transformers.
        for resource in path.rglob("*"):
            _inside_bundle(resource, root)
        for field, relative_file in (
            ("metadata_sha256", "metadata.json"),
            ("head_sha256", "head.safetensors"),
            ("encoder_sha256", "encoder/model.safetensors"),
        ):
            expected = component.get(field)
            if not isinstance(expected, str) or _HASH_PATTERN.fullmatch(expected) is None:
                raise ValueError(f"Некорректный {field} компонента {name}")
            file_path = _inside_bundle(path / relative_file, root)
            if not file_path.is_file() or _sha256(file_path) != expected.lower():
                raise ValueError(f"Отпечаток {relative_file} компонента {name} не совпадает")
        validated.append((path, float(weight)))
    if not math.isclose(math.fsum(weight for _, weight in validated), 1.0, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("Сумма весов компонентов selection.json должна равняться 1")
    return selection, validated


class SubcategoryExtractionPipeline(BasePipeline):
    """Усреднить вероятности выбранных моделей без API и внешних учителей."""

    name = "semantic_evidence"

    def __init__(self, artifact_path: Path | str | None = None) -> None:
        if artifact_path is None:
            artifact_path = os.getenv("SEMANTIC_EVIDENCE_ARTIFACT_PATH", str(DEFAULT_ARTIFACT))
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        selection, components = _load_selection(self.artifact_path)
        self.categories = tuple(MICROCATEGORIES)
        self.thresholds = {int(key): float(value) for key, value in selection["thresholds"].items()}
        self._components = []
        component_metadata = []
        trained = set()
        for path, weight in components:
            model, tokenizer, metadata = load_artifact(path, torch.device("cpu"))
            self._components.append((model, tokenizer, metadata["architecture"], weight))
            component_metadata.append(metadata)
            trained.update(metadata["trained_classes"])
        self._trained_categories = frozenset(trained)
        self._metadata = deepcopy(selection)
        self._metadata["component_metadata"] = component_metadata
        self._metadata["trained_classes"] = [category for category in self.categories if category in trained]

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Вернуть независимую копию selection и provenance каждого компонента."""
        return deepcopy(self._metadata)

    @property
    def device(self) -> str:
        """Всегда сообщать фактическое production-устройство CPU."""
        return "cpu"

    @property
    def trained_categories(self) -> frozenset[int]:
        """Сообщить категории с обучающими положительными примерами в композиции."""
        return self._trained_categories

    @property
    def unsupported_categories(self) -> frozenset[int]:
        """Сообщить категории без положительных примеров во всех компонентах."""
        return frozenset(self.categories) - self.trained_categories

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Взвесить sigmoid вероятности, исключив только исходную категорию."""
        probabilities = torch.zeros(len(self.categories), dtype=torch.float64)
        with torch.inference_mode():
            for model, tokenizer, architecture, weight in self._components:
                windows = tokenize_document(tokenizer, inp.description, architecture)
                batch = collate_documents(tokenizer, [windows])
                probabilities.add_(model(batch).sigmoid()[0].to(torch.float64), alpha=weight)
        return {
            category: float(score)
            for category, score in zip(self.categories, probabilities.tolist(), strict=True)
            if category != inp.mc_id
        }

    def run(self, inp: PipelineInput) -> list[int]:
        """Применить зафиксированные TRAIN/OOF-пороги выбранной композиции."""
        return [
            category for category, probability in self.probabilities(inp).items()
            if probability >= self.thresholds[category]
        ]
