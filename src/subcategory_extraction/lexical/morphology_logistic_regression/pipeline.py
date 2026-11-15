"""Логистическая регрессия на окнах вокруг словоформ."""

import gzip
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from main import BasePipeline, PipelineInput
from ..snippet_logistic_regression.pipeline import features, identity

ARTIFACTS = Path(__file__).with_name("artifacts")
MODEL_NAMES = ("avito-case-train", "yandex-youla-decision1")
DEFAULT_MODEL = MODEL_NAMES[0]
TOKEN = re.compile(r"[^\W_]+(?:[-‐‑][^\W_]+)*")
BOUNDARY = re.compile(r"[.!?;\n]+")
WINDOW_WIDTH = 11


def normalize_token(text: str) -> str:
    """Привести токен к ключу индекса."""
    return unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").replace("‐", "-").replace("‑", "-")


def windows(text: str, index: dict[str, list[list[str]]]) -> list[tuple[str, str]]:
    """Открыть окно в пределах предложения и сохранить канонический якорь."""
    found: dict[tuple[str, str], None] = {}
    for part in BOUNDARY.split(text.lower()):
        tokens = list(TOKEN.finditer(part))
        for i, token in enumerate(tokens):
            matches = index.get(normalize_token(token.group()))
            if not matches:
                continue
            left = max(0, min(i - WINDOW_WIDTH // 2, len(tokens) - WINDOW_WIDTH))
            right = min(left + WINDOW_WIDTH, len(tokens))
            snippet = part[tokens[left].start():tokens[right - 1].end()]
            for _, lemma in matches:
                found[snippet, lemma] = None
    return list(found)


class SubcategoryExtractionPipeline(BasePipeline):
    """Оценить объединённые признаки окон по каждой категории."""

    name = "morphology_logistic_regression"

    def __init__(self) -> None:
        name = os.getenv("MORPH_LOGREG_MODEL", DEFAULT_MODEL)
        if name not in MODEL_NAMES:
            raise ValueError(f"Неизвестный вариант морфологического логрега: {name}")
        with gzip.open(ARTIFACTS / f"{name}.json.gz", "rt", encoding="utf-8") as file:
            model = json.load(file)
        self.index = model["index"]
        self.categories = model["categories"]
        self.vectorizer = CountVectorizer(analyzer=identity, binary=True, vocabulary=model["vocabulary"])
        self.weights = np.asarray(model["weights"])
        self.cutoffs = np.asarray(model["cutoffs"])

    def run(self, inp: PipelineInput) -> list[int]:
        snippets = windows(inp.description, self.index)
        if not snippets:
            return []
        row = self.vectorizer.transform([features(snippets, "pairs")])
        scores = (row @ self.weights.T)[0]
        return [cid for cid, score, cutoff in zip(self.categories, scores, self.cutoffs, strict=True)
                if cid != inp.mc_id and score >= cutoff]
