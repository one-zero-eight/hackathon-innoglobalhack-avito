"""Классификация по бинарным n-граммам в окнах вокруг ключей."""

import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from main import BasePipeline, MICROCATEGORIES, PipelineInput

ARTIFACT = Path(__file__).with_name("artifacts") / "model.json.gz"
WORD = re.compile(r"\w+")
WORD_GRAMS = CountVectorizer(ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b").build_analyzer()
CHAR_GRAMS = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5)).build_analyzer()


def windows(text: str, anchors: set[str], width: int = 7, shift: int = 0) -> list[tuple[str, str]]:
    """Выделить окна, сохраняя ключевую фразу и знаки препинания."""
    text = text.lower()
    tokens = list(WORD.finditer(text))
    found = {}
    for start in range(len(tokens)):
        for size in (1, 2):
            phrase_tokens = tokens[start:start + size]
            if len(phrase_tokens) != size:
                continue
            phrase = " ".join(token.group() for token in phrase_tokens)
            if phrase not in anchors:
                continue
            left = max(0, min(start - (width - size) // 2 + shift, len(tokens) - width))
            snippet = text[tokens[left].start():tokens[min(left + width, len(tokens)) - 1].end()]
            found[snippet, phrase] = None
    return list(found)


STOP_MODES = ("before", "halfway", "after")


def stopped_windows(
    text: str, ranked: dict[int, list[str]], width: int, mode: str, *, fixed_dictionary: bool = False,
) -> list[tuple[str, str]]:
    """Обрезать окно у соседних ключей с непересекающимися категориями."""
    owners: dict[str, set[int]] = {}
    for cid, terms in ranked.items():
        for term in terms:
            owners.setdefault(term, set()).add(cid)
    text = text.lower()
    tokens = list(WORD.finditer(text))
    sizes = sorted({len(term.split()) for term in owners})
    matches: dict[int, list[tuple[int, str]]] = {}
    for start in range(len(tokens)):
        for size in sizes:
            end = start + size
            if size <= width and end <= len(tokens):
                phrase = " ".join(token.group() for token in tokens[start:end])
                if fixed_dictionary:
                    if BOUNDARY.search(text[tokens[start].start():tokens[end - 1].end()]):
                        continue
                    phrase = phrase.replace("ё", "е")
                if phrase in owners:
                    matches.setdefault(start, []).append((end, phrase))

    found = {}
    for start, anchors in matches.items():
        for end, phrase in anchors:
            left = max(0, min(start - (width - (end - start)) // 2, len(tokens) - width))
            right = min(left + width, len(tokens))
            previous, following = left, right
            for other_start in range(max(0, left - max(sizes) + 1), right):
                for other_end, other in matches.get(other_start, ()):
                    if not owners[phrase].isdisjoint(owners[other]):
                        continue
                    if left < other_end <= start:
                        previous = max(previous, other_end)
                    elif end <= other_start < right:
                        following = min(following, other_start)
            if previous > left:
                left = {"before": previous, "halfway": (previous + start) // 2, "after": start}[mode]
            if following < right:
                right = {"before": following, "halfway": (end + following) // 2, "after": end}[mode]
            snippet = text[tokens[left].start():tokens[right - 1].end()]
            found[snippet, phrase] = None
    return list(found)


def features(snippets: list[tuple[str, str]], mode: str) -> list[str]:
    """Извлечь признаки внутри каждого окна."""
    result = set()
    for snippet, anchor in snippets:
        if mode == "keys":
            result.add(anchor)
            continue
        words = WORD_GRAMS(snippet)
        result.update("w:" + word for word in words)
        result.update("c:" + gram for gram in CHAR_GRAMS(snippet))
        if mode == "pairs":
            result.update(f"a:{anchor}|{word}" for word in words if " " in word)
    return sorted(result)


BOUNDARY = re.compile(r"[.!?;,:\n]+")


def ad_features(
    text: str, anchors: set[str], ranked: dict[int, list[str]], width: int, mode: str,
) -> tuple[list[str], bool]:
    """Собрать общие и, при необходимости, признаки своей категории."""
    parts = BOUNDARY.split(text) if mode in {"split", "category"} else [text]
    if mode in STOP_MODES:
        snippets = stopped_windows(text, ranked, width, mode)
    else:
        snippets = [item for part in parts for item in windows(part, anchors, width)]
    shared = features(snippets, "pairs")
    if mode == "priority":
        hits = {anchor for _, anchor in snippets}
        shared.extend(f"{cid}:top10" for cid, terms in ranked.items() if hits.intersection(terms[:10]))
    if mode == "category":
        for cid, terms in ranked.items():
            own = set(terms)
            shared.extend(f"{cid}:{feature}" for feature in features(
                [item for item in snippets if item[1] in own], "pairs",
            ))
    return shared, bool(snippets)


def identity(values: list[str]) -> list[str]:
    """Передать готовые признаки в векторизатор."""
    return values


class SubcategoryExtractionPipeline(BasePipeline):
    """Вернуть категории, прошедшие пороги логистической регрессии."""

    name = "snippet_logistic_regression"

    def __init__(self) -> None:
        with gzip.open(os.getenv("SNIPPET_LOGISTIC_ARTIFACT") or ARTIFACT, "rt", encoding="utf-8") as file:
            model = json.load(file)
        self.categories = model["categories"]
        self.anchors = set(model["anchors"])
        self.mode, self.width, self.shift = model["policy"]
        self.ranked = {int(cid): terms for cid, terms in model.get("ranked", {}).items()}
        self.vectorizer = CountVectorizer(analyzer=identity, binary=True, vocabulary=model["vocabulary"])
        self.weights = np.asarray(model["weights"])
        self.cutoff = np.asarray(model["cutoff"])
        if (not self.categories or set(self.categories) - MICROCATEGORIES.keys()
            or self.weights.shape != (len(self.categories), len(model["vocabulary"]))
            or self.cutoff.shape != (len(self.categories),)
            or not np.isfinite(self.weights).all() or not np.isfinite(self.cutoff).all()):
            raise ValueError("Некорректный артефакт логистической регрессии")

    def run(self, inp: PipelineInput) -> list[int]:
        if self.ranked:
            row, hit = ad_features(inp.description, self.anchors, self.ranked, self.width, self.mode)
        else:
            snippets = windows(inp.description, self.anchors, self.width, self.shift)
            row, hit = features(snippets, self.mode), bool(snippets)
        if not hit:
            return []
        scores = self.vectorizer.transform([row]) @ self.weights.T
        return [cid for cid, score, cutoff in zip(self.categories, scores[0], self.cutoff, strict=True)
                if cid != inp.mc_id and score >= cutoff]
