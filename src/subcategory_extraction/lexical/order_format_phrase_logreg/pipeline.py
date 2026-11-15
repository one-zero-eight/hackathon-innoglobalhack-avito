"""Логрег на фрагментах с применением ограничений к найденным категориям."""

import gzip
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer

from main import PipelineInput
from ..order_format_logistic_regression.pipeline import HOMOGLYPHS
from ..snippet_logistic_regression.pipeline import (
    SubcategoryExtractionPipeline as CategoryPipeline,
    features,
    identity,
)

ARTIFACT = Path(__file__).with_name("artifacts") / "model.json.gz"
THRESHOLD = 0.5
BOUNDARY = re.compile(r"\n+|(?<=[.!?])\s+")
CATEGORY_TERMS = {
    102: re.compile(r"штукатур"),
    103: re.compile(r"плит[коч]|керамогранит"),
    104: re.compile(r"сантех|разводк\w+ (?:воды|труб)|водоснабж"),
    105: re.compile(r"электри[кч]|электромонтаж|проводк"),
    106: re.compile(r"покле\w+|окле\w+|обо[иея]|маляр|покраск"),
}
ANY_SERVICE = re.compile(r"отдельн\w+ (?:заказ|вид\w* работ|работ|этап)|разов\w+ заявк")


def phrases(description: str) -> list[str]:
    """Выделить фразы из всего объявления, сохраняя противопоставления."""
    return [part.strip() for part in BOUNDARY.split(description) if part.strip()]


def phrase_features(phrase: str) -> list[str]:
    """Получить признаки n-грамм целой фразы, независимо от ключей."""
    return features([(phrase.casefold().translate(HOMOGLYPHS), "")], "ngrams")


class OrderFormatClassifier:
    """Оценить фразы локально и сохранить цитаты, повлиявшие на решение."""

    def __init__(self, model: dict | None = None) -> None:
        if model is None:
            with gzip.open(ARTIFACT, "rt", encoding="utf-8") as stream:
                model = json.load(stream)
        self.vectorizer = CountVectorizer(
            analyzer=identity, binary=True, vocabulary=model["vocabulary"],
        )
        self.labels = model["classes"]
        self.weights = np.asarray(model["weights"])
        self.intercept = np.asarray(model["intercept"])
        if (not {"neutral", "block_all"} <= set(self.labels)
                or set(self.labels) - {"neutral", "allow", "block_all", "block_category"}
                or self.weights.shape != (len(self.labels), len(model["vocabulary"]))
                or self.intercept.shape != (len(self.labels),)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all()):
            raise ValueError("Некорректный артефакт фразового логрега")

    def assess(self, description: str, categories: list[int]) -> tuple[list[int], list[dict]]:
        """Применить общий запрет, адресные запреты и явно названные исключения."""
        parts = phrases(description)
        if not categories or not parts:
            return categories, []
        matrix = self.vectorizer.transform([phrase_features(part) for part in parts])
        scores = softmax(matrix @ self.weights.T + self.intercept, axis=1)
        all_categories = set(categories)
        blocked: set[int] = set()
        allowed: set[int] = set()
        evidence = []
        for part, row in zip(parts, scores, strict=True):
            best = int(row.argmax())
            label, score = self.labels[best], float(row[best])
            if label == "neutral" or score < THRESHOLD:
                continue
            normalized = part.casefold().translate(HOMOGLYPHS)
            named = {cid for cid in categories if cid in CATEGORY_TERMS and CATEGORY_TERMS[cid].search(normalized)}
            affected = all_categories if label == "block_all" else named
            if label == "allow":
                if not affected and ANY_SERVICE.search(normalized):
                    affected = all_categories
                allowed.update(affected)
            else:
                blocked.update(affected)
            if affected:
                evidence.append({"text": part, "label": label, "score": score, "mc_ids": sorted(affected)})
        return [cid for cid in categories if cid not in blocked or cid in allowed], evidence


class SubcategoryExtractionPipeline(CategoryPipeline):
    """Применить проверку формата заказа к результату логрега категорий."""

    name = "order_format_phrase_logreg"

    def __init__(self) -> None:
        super().__init__()
        self.order_format = OrderFormatClassifier()

    def run(self, inp: PipelineInput) -> list[int]:
        categories = super().run(inp)
        return self.order_format.assess(inp.description, categories)[0]
